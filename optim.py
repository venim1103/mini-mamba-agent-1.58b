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
import math

# G11: Optional 8-bit AdamW for ~0.46 GB optimizer state savings (Parent model)
try:
    import bitsandbytes as bnb
    Adam8bit = bnb.optim.Adam8bit
except ImportError:
    Adam8bit = None

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, ns_steps=3):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None: continue
                
                g = p.grad.data.float()
                p.grad = None  # Free BF16 gradient memory (G5)
                
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                
                a, b, c = (3.4445, -4.7750, 2.0315)
                X = buf / (buf.norm(keepdim=True) + 1e-8)  # New tensor, buf stays intact (G4)
                for _ in range(ns_steps):
                    A = X @ X.T
                    B = b * A + c * A @ A
                    X = a * X + B @ X
                    
                p.data.add_(X.type_as(p.data), alpha=-lr)

def setup_mamba_optimizers(model, config, use_8bit=True):
    muon_params, adam_params, mamba_sensitive_params = [], [], []
    
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        
        # ISOLATION: The sensitive continuous Mamba parameters
        if any(key in name for key in ['A_log', 'D', 'dt_bias', 'dt_proj']):
            mamba_sensitive_params.append(p)
        # Muon handles the 2D BitLinear weights (ndim == 2 excludes 3D conv1d weights)
        elif p.ndim == 2 and 'weight' in name and 'norm' not in name and 'tok_embeddings' not in name:
            muon_params.append(p)
        # AdamW handles biases, norms, and embeddings
        else:
            adam_params.append(p)

    # G11: Use 8-bit Adam when bitsandbytes is available (~0.46 GB savings)
    AdamCls = Adam8bit if (use_8bit and Adam8bit is not None) else torch.optim.AdamW

    muon_opt = Muon(muon_params, lr=config['peak_lr'])
    adam_opt = AdamCls(adam_params, lr=config['peak_lr'], weight_decay=0.01)
    
    # The dedicated optimizer for the State Space Core (Fixed low LR, 0 Weight Decay)
    mamba_core_opt = AdamCls(
        mamba_sensitive_params, lr=config['peak_lr'] * 0.1, weight_decay=0.0 
    )

    return muon_opt, adam_opt, mamba_core_opt

class FGWSD_Scheduler:
    def __init__(self, muon_opt, adam_opt, mamba_opt, total_steps, config):
        self.opts = [muon_opt, adam_opt, mamba_opt]
        self.peak_lr = config['peak_lr']
        self.end_lr = config['end_lr']
        self.phases = config['phases']
        self.total_steps = total_steps
        
        self.step_boundaries = []
        current_step = 0
        for p in self.phases:
            current_step += int(self.total_steps * p['pct'])
            self.step_boundaries.append(current_step)

    def get_lr_and_ctx(self, step):
        for i, boundary in enumerate(self.step_boundaries):
            if step < boundary:
                phase = self.phases[i]
                start_step = self.step_boundaries[i-1] if i > 0 else 0
                progress = (step - start_step) / (boundary - start_step)
                
                if i == 0: lr = self.end_lr + progress * (self.peak_lr - self.end_lr)
                elif i in [1, 2]: lr = self.peak_lr
                else:
                    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                    lr = self.end_lr + (self.peak_lr - self.end_lr) * cosine_decay
                    
                return lr, phase['ctx'], f"Phase_{i+1}"
        return self.end_lr, self.phases[-1]['ctx'], "Complete"

    def step(self, current_step):
        lr, ctx, phase_name = self.get_lr_and_ctx(current_step)
        
        # Update Adam and Muon
        for opt in self.opts[:2]:
            for group in opt.param_groups: group['lr'] = lr
            
        # Update Mamba core (keep it 10x lower than the main LR)
        for group in self.opts[2].param_groups: group['lr'] = lr * 0.1
            
        return lr, ctx, phase_name

