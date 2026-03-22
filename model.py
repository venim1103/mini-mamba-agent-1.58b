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
    from mamba_ssm.ops.triton.ssd_combined import mamba_inner_fn
except ImportError:
    raise ImportError("Please install mamba-ssm: pip install mamba-ssm")

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
    w_quant = tl.math.round(w_scaled)
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

# ==========================================
# 2. BitMamba Block & Architecture
# ==========================================
class BitMambaBlock(nn.Module):
    def __init__(self, dim, d_state=128, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_inner = int(expand * dim)
        self.d_state = d_state
        self.dt_rank = math.ceil(self.dim / 16)
        
        self.norm = RMSNorm(dim)

        # 1.58-bit Heavy Projections (Saves VRAM!)
        self.in_proj = BitLinear(self.dim, self.d_inner * 2, bias=False)
        self.x_proj = BitLinear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.out_proj = BitLinear(self.d_inner, self.dim, bias=False)

        # FP16/FP32 Recurrent Core (Maintains Stability!)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner, bias=True, 
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Stored in log-space for stability
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) 
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))

    def forward(self, hidden_states, seq_idx=None):
        batch, seqlen, _ = hidden_states.shape
        h = self.norm(hidden_states)

        xz = self.in_proj(h)
        x, z = xz.chunk(2, dim=-1)

        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seqlen] 
        x = x_conv.transpose(1, 2)
        x = F.silu(x)

        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        A = -torch.exp(self.A_log.float())
        
        # seq_idx ensures state recurrence is flushed at document boundaries!
        y = mamba_inner_fn(
            x, dt, A, B, C, self.D.float(), z, 
            dt_bias=self.dt_proj.bias, 
            dt_softplus=True,
            seq_idx=seq_idx 
        )

        out = self.out_proj(y)
        return hidden_states + out

class BitMambaLLM(nn.Module):
    def __init__(self, vocab_size=64000, dim=1024, n_layers=40, d_state=128, expand=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(BitMambaBlock(dim, d_state=d_state, expand=expand))
            
        self.norm = RMSNorm(dim)
        self.output = BitLinear(dim, vocab_size)
        self.output.weight = self.tok_embeddings.weight 

    def forward(self, input_ids, seq_idx=None):
        x = self.tok_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, seq_idx=seq_idx)
        x = self.norm(x)
        return self.output(x)

