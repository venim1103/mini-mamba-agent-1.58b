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
from typing import Optional, Set

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
    # Direct ternary quantization: {-1, 0, 1} with symmetric 0.5 threshold.
    w_quant = tl.where(w_scaled >= 0.5, 1.0, tl.where(w_scaled <= -0.5, -1.0, 0.0),)
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
        if not x.is_cuda or not self.weight.is_cuda:
            raise RuntimeError(
                "BitLinear requires CUDA tensors because ternary quantization uses a Triton kernel. "
                "Move model and inputs to GPU, e.g. model.to('cuda') and x = x.to('cuda')."
            )
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

# ==========================================
# 3. Lightweight GQA Attention Block (Nemotron-H §2.1)
# ==========================================
class BitAttentionBlock(nn.Module):
    """Lightweight grouped-query attention block with ternary projections.

    Dispersed among Mamba blocks (~8% of layers) to provide precise
    retrieval from context — critical for tool-name/parameter recall.
    Uses GQA with fewer KV heads to keep parameter cost low.
    """

    def __init__(self, dim, n_heads=16, n_kv_heads=4, max_seq_len=16384):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads  # GQA repetition factor

        self.norm = RMSNorm(dim)

        # Ternary projections for Q, K, V, O  (consistent with BitMamba design)
        self.wq = BitLinear(dim, n_heads * self.head_dim, bias=False)
        self.wk = BitLinear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = BitLinear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = BitLinear(n_heads * self.head_dim, dim, bias=False)

        # RoPE frequency buffer
        freqs = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("rope_freqs", freqs, persistent=False)

    def _apply_rope(self, x, seq_len):
        """Apply rotary positional embedding in-place."""
        t = torch.arange(seq_len, device=x.device, dtype=self.rope_freqs.dtype)
        freqs = torch.outer(t, self.rope_freqs)  # (L, head_dim/2)
        cos = freqs.cos()[None, None, :, :]  # (1, 1, L, head_dim/2)
        sin = freqs.sin()[None, None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    def forward(self, hidden_states, seq_idx=None):
        bsz, seqlen, _ = hidden_states.shape
        h = self.norm(hidden_states)

        q = self.wq(h).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE
        q = self._apply_rope(q, seqlen)
        k = self._apply_rope(k, seqlen)

        # GQA: expand KV heads to match Q heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return hidden_states + self.wo(attn_out)


def _compute_attention_indices(n_layers: int, attn_pct: float = 0.08) -> Set[int]:
    """Evenly disperse attention layers among Mamba layers.

    Places ~attn_pct of layers as attention, evenly spaced through the stack.
    Avoids layer 0 (let embedding settle) and the last layer.
    """
    n_attn = max(1, round(n_layers * attn_pct))
    if n_attn >= n_layers:
        return set(range(n_layers))
    # Evenly space starting from roughly 1/(n_attn+1) through the stack
    step = n_layers / (n_attn + 1)
    indices = {int(round(step * (i + 1))) for i in range(n_attn)}
    # Clamp to valid range [1, n_layers-2]
    indices = {max(1, min(i, n_layers - 2)) for i in indices}
    return indices


class BitMambaLLM(nn.Module):
    def __init__(self, vocab_size=64000, dim=1024, n_layers=40, d_state=128, expand=2,
                 headdim=64, ngroups=1, chunk_size=256,
                 use_attn=True, attn_pct=0.08, n_attn_heads=16, n_attn_kv_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_embeddings = nn.Embedding(vocab_size, dim)

        attn_indices = _compute_attention_indices(n_layers, attn_pct) if use_attn else set()
        self.attn_indices = attn_indices

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i in attn_indices:
                self.layers.append(BitAttentionBlock(
                    dim, n_heads=n_attn_heads, n_kv_heads=n_attn_kv_heads,
                ))
            else:
                self.layers.append(BitMambaBlock(
                    dim, d_state=d_state, expand=expand,
                    headdim=headdim, ngroups=ngroups, chunk_size=chunk_size,
                ))

        self.norm = RMSNorm(dim)
        self.output = BitLinear(dim, vocab_size)
        self.output.weight = self.tok_embeddings.weight

    def forward(self, input_ids, seq_idx=None):
        x = self.tok_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, seq_idx=seq_idx)
        x = self.norm(x)
        return self.output(x)

