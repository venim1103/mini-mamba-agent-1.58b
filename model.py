# Copyright 2026 venim1103
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    raise ImportError("Please install mamba-ssm>=2.0: pip install mamba-ssm")

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

from einops import rearrange

# ==========================================
# 1. Custom Triton Kernel for 1.58-bit Weights
# ==========================================
@triton.jit
def _ternary_quant_kernel(w_ptr, output_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    w = tl.load(w_ptr + offsets, mask=mask).to(tl.float32)
    w_scaled = w / scale
    # Triton does not expose tl.math.round on all versions; emulate nearest-int rounding.
    w_quant = tl.where(w_scaled >= 0, tl.floor(w_scaled + 0.5), tl.ceil(w_scaled - 0.5))
    w_quant = tl.maximum(w_quant, -1.0)
    w_quant = tl.minimum(w_quant, 1.0)
    tl.store(output_ptr + offsets, w_quant, mask=mask)

class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        scale = weight.abs().mean().clamp(min=1e-5)
        output = torch.empty_like(weight)
        n_elements = weight.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _ternary_quant_kernel[grid](weight, output, n_elements, scale.item(), BLOCK_SIZE=1024)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def weight_quant(weight):
    return TernaryQuantizeSTE.apply(weight)

def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    quantized_x = torch.round(x * scale)
    quantized_x = torch.clamp(quantized_x, -128, 127)
    return ((quantized_x - (x * scale)).detach() + (x * scale)) / scale

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)

    def forward(self, x):
        x_norm = self.norm(x)
        x_quant = activation_quant(x_norm)
        w_quant = weight_quant(self.weight)
        return F.linear(x_quant, w_quant, self.bias)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(norm_x + self.eps))


class AttentionBlock(nn.Module):
    """Lightweight attention block with GQA (4 KV heads) for hybrid architecture.
    
    As recommended in Nemotron-H §2.1, a small percentage of attention layers
    (evenly dispersed) alongside Mamba-2 layers improves retrieval-heavy tasks.
    """
    
    def __init__(self, dim, n_kv_heads=4):
        super().__init__()
        self.dim = dim
        self.n_heads = dim // 64  # head_dim = 64
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // self.n_heads
        
        # GQA: fewer KV heads than Q heads
        self.q_proj = BitLinear(dim, dim)
        self.k_proj = BitLinear(dim, self.n_kv_heads * self.head_dim)
        self.v_proj = BitLinear(dim, self.n_kv_heads * self.head_dim)
        self.o_proj = BitLinear(dim, dim)
        
        self.norm = RMSNorm(dim)
        
    def forward(self, hidden_states, seq_idx=None):
        x = self.norm(hidden_states)
        
        # Project to Q, K, V
        q = self.q_proj(x).view(x.shape[0], x.shape[1], self.n_heads, self.head_dim)
        k = self.k_proj(x).view(x.shape[0], x.shape[1], self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(x.shape[0], x.shape[1], self.n_kv_heads, self.head_dim)
        
        # Expand K, V to all heads (GQA)
        if self.n_kv_heads < self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)
        
        # Simple causal attention (no flash attention for compatibility)
        # Scale
        q = q * (self.head_dim ** -0.5)
        
        # Attention scores
        attn = torch.einsum("bqhd,bkhd->bhqk", q, k)
        
        # Causal mask
        seqlen = x.shape[1]
        mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v)
        out = out.contiguous().view(x.shape[0], x.shape[1], self.dim)
        
        return hidden_states + self.o_proj(out)

# ==========================================
# 2. BitMamba Block & Architecture
# ==========================================
class BitMambaBlock(nn.Module):
    """Mamba-2 SSD block with 1.58-bit (ternary) heavy projections.

    Architecture follows the Mamba-2 head-based formulation from the SSD paper
    (Dao & Gu, 2024) and the Nemotron-H design (§2.1). Key differences from
    Mamba-1:
      - A is a scalar per head (not a diagonal matrix per state)
      - B, C are shared across heads via ngroups (analogous to GQA)
      - No dt_rank decomposition — dt is produced directly per head
      - Uses the mamba_chunk_scan_combined Triton kernel with native seq_idx
    """

    def __init__(self, dim, d_state=128, d_conv=4, expand=2, headdim=64, ngroups=1, chunk_size=256):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.d_inner = int(expand * dim)
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size

        assert self.d_inner % headdim == 0, f"d_inner ({self.d_inner}) must be divisible by headdim ({headdim})"

        self.norm = RMSNorm(dim)

        # 1.58-bit Heavy Projections (Saves VRAM!)
        # in_proj: produces [z, x, B, C, dt] in one shot
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = BitLinear(self.dim, d_in_proj, bias=False)
        self.out_proj = BitLinear(self.d_inner, self.dim, bias=False)

        # FP16/FP32 Recurrent Core (Maintains Stability!)
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim, out_channels=conv_dim, bias=True,
            kernel_size=d_conv, groups=conv_dim, padding=d_conv - 1
        )

        # Mamba-2: A is a scalar per head (stored in log-space for stability)
        A = torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.nheads))

        # Initialize dt_bias so initial dt after softplus is in [dt_min, dt_max]
        dt_min, dt_max = 0.001, 0.1
        dt_init_floor = 1e-4
        inv_dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        inv_dt = torch.clamp(inv_dt, min=dt_init_floor)
        # Inverse of softplus: x = log(exp(val) - 1)
        self.dt_bias.data = inv_dt + torch.log(-torch.expm1(-inv_dt))

        # Output gate normalization (as used in Mamba-2 reference)
        self.out_norm = RMSNorm(self.d_inner)

    def forward(self, hidden_states, seq_idx=None):
        batch, seqlen, _ = hidden_states.shape
        h = self.norm(hidden_states)

        # Single projection produces z, x, B, C, dt
        zxbcdt = self.in_proj(h)

        # Split: z (gate) | xBC (conv input) | dt (timestep)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Causal conv1d on x, B, C (not z — z is the gate)
        if causal_conv1d_fn is not None:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2),  # (B, conv_dim, L)
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                activation="silu",
            ).transpose(1, 2)  # back to (B, L, conv_dim)
        else:
            xBC = self.conv1d(xBC.transpose(1, 2))[..., :seqlen].transpose(1, 2)
            xBC = F.silu(xBC)

        # Split convolved result into x, B, C
        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1
        )

        A = -torch.exp(self.A_log.float())  # (nheads,)

        # Reshape for Mamba-2 SSD kernel
        # x: (B, L, nheads, headdim), dt: (B, L, nheads)
        # B: (B, L, ngroups, d_state), C: (B, L, ngroups, d_state)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim),
            dt_bias=self.dt_bias,
            dt_softplus=True,
            seq_idx=seq_idx,
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        # Output gate normalization + projection
        y = self.out_norm(y)
        out = self.out_proj(y)
        return hidden_states + out

class BitMambaLLM(nn.Module):
    def __init__(self, vocab_size=64000, dim=1024, n_layers=40, d_state=128, expand=2,
                 headdim=64, ngroups=1, chunk_size=256, use_checkpoint=False,
                 use_attn=False, attn_pct=0.08):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_checkpoint = use_checkpoint
        self.use_attn = use_attn
        self.attn_pct = attn_pct
        
        # Compute attention layer indices (evenly dispersed as in Nemotron-H)
        if use_attn and attn_pct > 0:
            n_attn = max(1, int(n_layers * attn_pct))
            step = n_layers // n_attn
            self.attn_indices = set(range(step // 2, n_layers, step))
        else:
            self.attn_indices = set()
        
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i in self.attn_indices:
                # Lightweight attention layer (GQA with 4 KV heads)
                self.layers.append(AttentionBlock(dim, n_kv_heads=4))
            else:
                self.layers.append(BitMambaBlock(
                    dim, d_state=d_state, expand=expand,
                    headdim=headdim, ngroups=ngroups, chunk_size=chunk_size,
                ))
            
        self.norm = RMSNorm(dim)
        self.output = BitLinear(dim, vocab_size, bias=False)

    def forward(self, input_ids, seq_idx=None):
        from torch.utils.checkpoint import checkpoint
        x = self.tok_embeddings(input_ids)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, seq_idx, use_reentrant=False)
            else:
                x = layer(x, seq_idx=seq_idx)
        x = self.norm(x)
        return self.output(x)

    def forward_hidden(self, input_ids, seq_idx=None):
        """Return pre-logit hidden states (G2: for chunked cross-entropy)."""
        from torch.utils.checkpoint import checkpoint
        x = self.tok_embeddings(input_ids)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, seq_idx, use_reentrant=False)
            else:
                x = layer(x, seq_idx=seq_idx)
        return self.norm(x)


def chunked_cross_entropy(hidden, output_proj, targets, chunk_size=1024, ignore_index=-100):
    """Compute cross-entropy without materializing the full logits tensor.

    Instead of computing logits for the entire sequence at once (which for
    [BS, 16384, 64000] costs ~3.9 GB), this applies the output projection
    in chunks of ``chunk_size`` tokens along the sequence dimension.

    Args:
        hidden:      (BS, seq_len, dim) — pre-logit hidden states from forward_hidden
        output_proj: nn.Module — the output head (model.output)
        targets:     (BS, seq_len)  — target token ids
        chunk_size:  int — number of tokens per chunk (default 1024)
        ignore_index: int — label to ignore in loss (default -100)

    Returns:
        Scalar loss (mean cross-entropy over all tokens).
    """
    bs, seq_len, _ = hidden.shape
    total_loss = 0.0
    valid_tokens = (targets != ignore_index).sum().clamp(min=1)

    for i in range(0, seq_len, chunk_size):
        end = min(i + chunk_size, seq_len)
        chunk_logits = output_proj(hidden[:, i:end, :])          # (BS, chunk, vocab)
        chunk_targets = targets[:, i:end].reshape(-1)             # (BS * chunk,)
        total_loss += F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.size(-1)),
            chunk_targets,
            reduction='sum',
            ignore_index=ignore_index,
        )

    return total_loss / valid_tokens

