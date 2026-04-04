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
from contextlib import nullcontext
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    mamba_chunk_scan_combined = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from einops import rearrange


def maybe_autocast(device=None, amp_dtype=None):
    if device is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    elif isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device)

    if device_type != "cuda":
        return nullcontext()

    if amp_dtype is None:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16

    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _mamba_chunk_scan_combined_fallback(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D,
    z=None,
    dt_bias=None,
    dt_softplus=True,
    seq_idx=None,
    return_final_states=False,
):
    del chunk_size
    bsz, seqlen, nheads, headdim = x.shape
    ngroups = B.shape[2]
    d_state = B.shape[3]

    heads_per_group = nheads // ngroups
    Bh = B.repeat_interleave(heads_per_group, dim=2).to(torch.float32)
    Ch = C.repeat_interleave(heads_per_group, dim=2).to(torch.float32)

    x_f = x.to(torch.float32)
    dt_f = dt.to(torch.float32)
    A_f = A.to(torch.float32)
    D_f = D.to(torch.float32)

    if dt_bias is not None:
        dt_f = dt_f + dt_bias.to(torch.float32).view(1, 1, -1)
    if dt_softplus:
        dt_f = F.softplus(dt_f)

    state = x_f.new_zeros((bsz, nheads, headdim, d_state))
    outputs = []

    for t in range(seqlen):
        if seq_idx is not None and t > 0:
            reset = (seq_idx[:, t] != seq_idx[:, t - 1]).view(bsz, 1, 1, 1)
            state = torch.where(reset, torch.zeros_like(state), state)

        x_t = x_f[:, t, :, :]
        dt_t = dt_f[:, t, :]
        B_t = Bh[:, t, :, :]
        C_t = Ch[:, t, :, :]

        dA = torch.exp(dt_t * A_f.view(1, -1))
        dB = dt_t.unsqueeze(-1) * B_t

        state = dA.unsqueeze(-1).unsqueeze(-1) * state + torch.einsum("bhn,bhp->bhpn", dB, x_t)
        y_t = torch.einsum("bhpn,bhn->bhp", state, C_t) + D_f.view(1, -1, 1) * x_t

        if z is not None:
            y_t = y_t * F.silu(z[:, t, :, :].to(torch.float32))

        outputs.append(y_t)

    y = torch.stack(outputs, dim=1).to(x.dtype)
    if return_final_states:
        return y, state.to(x.dtype)
    return y


if mamba_chunk_scan_combined is None:
    mamba_chunk_scan_combined = _mamba_chunk_scan_combined_fallback

# ==========================================
# 1. Custom Triton Kernel for 1.58-bit Weights
# ==========================================
if HAS_TRITON:
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
        if HAS_TRITON and weight.is_cuda:
            output = torch.empty_like(weight)
            n_elements = weight.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            _ternary_quant_kernel[grid](weight, output, n_elements, scale.item(), BLOCK_SIZE=1024)
            return output
        w_scaled = weight / scale
        w_quant = torch.where(w_scaled >= 0, torch.floor(w_scaled + 0.5), torch.ceil(w_scaled - 0.5))
        return torch.clamp(w_quant, -1.0, 1.0)

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
        self.register_buffer('_cached_quant_weight', None, persistent=False)
        self._cached_quant_weight_version = -1

    def _clear_inference_cache(self):
        self._cached_quant_weight = None
        self._cached_quant_weight_version = -1

    def _get_quantized_weight(self):
        if self.training:
            return weight_quant(self.weight)

        weight_version = getattr(self.weight, '_version', None)
        cache_is_stale = (
            self._cached_quant_weight is None
            or self._cached_quant_weight.device != self.weight.device
            or self._cached_quant_weight.dtype != self.weight.dtype
            or self._cached_quant_weight_version != weight_version
        )
        if cache_is_stale:
            with torch.no_grad():
                self._cached_quant_weight = weight_quant(self.weight).detach()
            self._cached_quant_weight_version = weight_version
        return self._cached_quant_weight

    def prepare_for_inference(self):
        self.eval()
        self._get_quantized_weight()
        return self

    def train(self, mode=True):
        if mode:
            self._clear_inference_cache()
        return super().train(mode)

    def forward(self, x):
        x_norm = self.norm(x)
        x_quant = activation_quant(x_norm)
        w_quant = self._get_quantized_weight()
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

        # Use PyTorch SDPA to avoid explicitly materializing a full causal mask.
        # Shapes for SDPA are (B, H, L, D).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if seq_idx is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
            out = out.transpose(1, 2)
        else:
            # Enforce document boundaries from seq_idx by attending only within
            # each contiguous segment. This prevents cross-document leakage.
            batch_size, _, seqlen, _ = q.shape
            out = hidden_states.new_empty(batch_size, seqlen, self.n_heads, self.head_dim)
            for b in range(batch_size):
                seg = seq_idx[b]
                changes = torch.nonzero(seg[1:] != seg[:-1], as_tuple=False).flatten() + 1
                boundaries = torch.cat([
                    torch.tensor([0], device=seg.device, dtype=torch.long),
                    changes.to(torch.long),
                    torch.tensor([seqlen], device=seg.device, dtype=torch.long),
                ])

                for i in range(boundaries.numel() - 1):
                    start = boundaries[i].item()
                    end = boundaries[i + 1].item()
                    if end <= start:
                        continue
                    q_seg = q[b:b + 1, :, start:end, :]
                    k_seg = k[b:b + 1, :, start:end, :]
                    v_seg = v[b:b + 1, :, start:end, :]
                    out_seg = F.scaled_dot_product_attention(
                        q_seg, k_seg, v_seg, attn_mask=None, dropout_p=0.0, is_causal=True
                    )
                    out[b:b + 1, start:end, :, :] = out_seg.transpose(1, 2)

        out = out.contiguous().view(x.shape[0], x.shape[1], self.dim)
        
        return hidden_states + self.o_proj(out)

    def prefill(self, hidden_states):
        """Full-sequence attention, returns output and KV cache for decoding."""
        x = self.norm(hidden_states)
        bsz, seqlen, _ = x.shape

        q = self.q_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        if self.n_kv_heads < self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k_exp = k.repeat_interleave(repeat_factor, dim=2)
            v_exp = v.repeat_interleave(repeat_factor, dim=2)
        else:
            k_exp, v_exp = k, v

        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k_exp.transpose(1, 2), v_exp.transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        cache = {"k": k, "v": v}
        return hidden_states + self.o_proj(out), cache

    def step(self, hidden_states, cache):
        """Single-token decode step with KV cache."""
        x = self.norm(hidden_states)
        bsz = x.shape[0]

        q = self.q_proj(x).view(bsz, 1, self.n_heads, self.head_dim)
        k_new = self.k_proj(x).view(bsz, 1, self.n_kv_heads, self.head_dim)
        v_new = self.v_proj(x).view(bsz, 1, self.n_kv_heads, self.head_dim)

        cache["k"] = torch.cat([cache["k"], k_new], dim=1)
        cache["v"] = torch.cat([cache["v"], v_new], dim=1)

        k, v = cache["k"], cache["v"]
        if self.n_kv_heads < self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)

        # Single query attending to all cached keys — no causal mask needed
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=False,
        ).transpose(1, 2).contiguous().view(bsz, 1, self.dim)

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
        if causal_conv1d_fn is not None and xBC.is_cuda:
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

    def prefill(self, hidden_states):
        """Full-sequence Mamba-2 scan, returns output and (conv_state, ssm_state) for decoding."""
        batch, seqlen, _ = hidden_states.shape
        h = self.norm(hidden_states)

        zxbcdt = self.in_proj(h)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Save conv state: last d_conv inputs before convolution
        xBC_t = xBC.transpose(1, 2)  # (batch, conv_dim, seqlen)
        if seqlen >= self.d_conv:
            conv_state = xBC_t[:, :, -self.d_conv:].clone()
        else:
            conv_state = F.pad(xBC_t, (self.d_conv - seqlen, 0)).clone()

        # Apply conv1d
        if causal_conv1d_fn is not None and xBC_t.is_cuda:
            xBC = causal_conv1d_fn(
                xBC_t,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                activation="silu",
            ).transpose(1, 2)
        else:
            xBC = self.conv1d(xBC_t)[..., :seqlen].transpose(1, 2)
            xBC = F.silu(xBC)

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1
        )

        A = -torch.exp(self.A_log.float())

        y, ssm_state = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt, A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim),
            dt_bias=self.dt_bias,
            dt_softplus=True,
            return_final_states=True,
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.out_norm(y)
        out = self.out_proj(y)

        cache = {"conv_state": conv_state, "ssm_state": ssm_state}
        return hidden_states + out, cache

    def step(self, hidden_states, cache):
        """Single-token Mamba-2 step with cached conv + SSM state.

        Uses official causal_conv1d_update and selective_state_update Triton
        kernels when available (matching the Mamba-2 reference implementation).
        The SSD chunk-scan kernel and sequential step use algebraically
        equivalent but numerically distinct accumulation orders, so step
        outputs will have small relative differences (~0.5%) vs a full
        forward pass.  This is inherent to the SSD formulation and matches
        the upstream mamba-ssm behaviour.
        """
        dtype = hidden_states.dtype
        h = self.norm(hidden_states)  # (bsz, 1, dim)

        zxbcdt = self.in_proj(h).squeeze(1)  # (bsz, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        conv_state = cache["conv_state"]

        if causal_conv1d_update is not None and xBC.is_cuda:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                activation="silu",
            )
        else:
            conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            conv_state[:, :, -1] = xBC
            cache["conv_state"] = conv_state
            xBC = (conv_state * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias
            xBC = F.silu(xBC)

        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1
        )

        A = -torch.exp(self.A_log.float())
        ssm_state = cache["ssm_state"]  # (bsz, nheads, headdim, d_state)

        if selective_state_update is not None and ssm_state.is_cuda:
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            # selective_state_update expects per-element A/dt/D with stride-0
            # broadcasting along headdim (detected via tie_hdim optimisation).
            A_ssm = A.view(self.nheads, 1, 1).expand(self.nheads, self.headdim, self.d_state)
            dt_ssm = dt.unsqueeze(-1).expand(-1, -1, self.headdim)
            dt_bias_ssm = self.dt_bias.unsqueeze(-1).expand(-1, self.headdim)
            D_ssm = self.D.unsqueeze(-1).expand(-1, self.headdim)
            y = selective_state_update(
                ssm_state, x, dt_ssm, A_ssm, B, C,
                D=D_ssm, z=z,
                dt_bias=dt_bias_ssm, dt_softplus=True,
            )
            y = rearrange(y, "b h p -> b (h p)")
        else:
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim).float()
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups).float()
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups).float()
            heads_per_group = self.nheads // self.ngroups
            B = B.repeat_interleave(heads_per_group, dim=1)
            C = C.repeat_interleave(heads_per_group, dim=1)

            dt_act = F.softplus(dt.float() + self.dt_bias.float())
            dA = torch.exp(dt_act * A)
            dB = dt_act.unsqueeze(-1) * B

            ssm_state = dA.unsqueeze(-1).unsqueeze(-1) * ssm_state + torch.einsum("bhn,bhp->bhpn", dB, x)
            cache["ssm_state"] = ssm_state

            y = torch.einsum("bhpn,bhn->bhp", ssm_state, C)
            y = y + self.D.float().unsqueeze(-1) * x
            z = rearrange(z, "b (h p) -> b h p", p=self.headdim).float()
            y = y * F.silu(z)
            y = rearrange(y, "b h p -> b (h p)")

        y = y.unsqueeze(1).to(dtype)  # (bsz, 1, d_inner)
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

    def _backbone(self, input_ids, seq_idx=None):
        """Shared backbone: embedding → layers → norm. Used by both forward and forward_hidden."""
        from torch.utils.checkpoint import checkpoint
        x = self.tok_embeddings(input_ids)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, seq_idx, use_reentrant=False)
            else:
                x = layer(x, seq_idx=seq_idx)
        return self.norm(x)

    def forward(self, input_ids, seq_idx=None, targets=None):
        hidden_states = self._backbone(input_ids, seq_idx)
        
        if targets is not None:
            return chunked_cross_entropy(hidden_states, self.output, targets, return_stats=True)
            
        return self.output(hidden_states)

    def forward_hidden(self, input_ids, seq_idx=None):
        """Return pre-logit hidden states (G2: for chunked cross-entropy)."""
        return self._backbone(input_ids, seq_idx)

    def prepare_for_inference(self):
        self.eval()
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.prepare_for_inference()
        return self

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=512, temperature=0.7,
                 do_sample=True, eos_token_id=None):
        """O(n) autoregressive generation with cached SSM/KV states.

        Prefills the prompt in one pass, then decodes one token at a time
        using per-layer conv/SSM state (Mamba blocks) and KV cache (attention blocks).
        """
        self.prepare_for_inference()

        # --- Prefill: process the full prompt ---
        x = self.tok_embeddings(input_ids)
        caches = []
        for layer in self.layers:
            x, cache = layer.prefill(x)
            caches.append(cache)
        x = self.norm(x)
        logits = self.output(x[:, -1:, :])  # (B, 1, vocab)

        # --- Decode: one token at a time ---
        generated = input_ids
        for _ in range(max_new_tokens):
            if do_sample and temperature > 0:
                next_token = torch.multinomial(
                    F.softmax(logits[0, -1, :] / temperature, dim=-1), num_samples=1
                )
            else:
                next_token = logits[0, -1, :].argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            # Single-token forward through cached layers
            x = self.tok_embeddings(next_token.unsqueeze(0))  # (1, 1, dim)
            for i, layer in enumerate(self.layers):
                x = layer.step(x, caches[i])
            x = self.norm(x)
            logits = self.output(x)  # (1, 1, vocab)

        return generated


def chunked_cross_entropy(hidden, output_proj, targets, chunk_size=1024, ignore_index=-100, return_stats=False):
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
        Scalar loss (mean cross-entropy over all tokens), or a tuple of
        ``(total_loss_sum, valid_tokens)`` when ``return_stats=True``.
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

    if return_stats:
        return total_loss, valid_tokens
    return total_loss / valid_tokens

