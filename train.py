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
import torch.nn.functional as F
import os
import time
import wandb
from model import BitMambaLLM
from data import create_dataloaders
from optim import setup_mamba_optimizers, FGWSD_Scheduler

MODE = "scout" 

if MODE == "scout":
    MODEL_CONFIG = dict(vocab_size=64000, dim=512, n_layers=24, d_state=64, expand=2)
    TOTAL_STEPS = 100_000 
    PEAK_LR = 4.5e-4 
else:
    MODEL_CONFIG = dict(vocab_size=64000, dim=1024, n_layers=40, d_state=128, expand=2)
    TOTAL_STEPS = 1_000_000 
    PEAK_LR = 3.0e-4 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2             
GRAD_ACCUM_STEPS = 8       
SAVE_EVERY = 5000
CHECKPOINT_DIR = f"checkpoints/bitmamba_{MODE}"

TRAIN_CONFIG = [
    {"name": "formal_logic", "path": "local_data/train/logic", "format": "parquet", "weight": 0.35},
    {"name": "code",         "path": "local_data/train/code",  "format": "json",    "weight": 0.30},
    {"name": "web",          "path": "local_data/train/web",   "format": "parquet", "weight": 0.20},
    {"name": "tool_use",     "path": "local_data/train/tools", "format": "json",    "weight": 0.15}
]

# UPDATED: Max Context capped at 16k (16384) to safely fit BS=2 on RTX 3090
CURRICULUM_CONFIG = {
    "peak_lr": PEAK_LR, "end_lr": 1.5e-6,
    "phases": [
        {"pct": 0.05, "ctx": 2048},   
        {"pct": 0.40, "ctx": 4096},   
        {"pct": 0.35, "ctx": 8192},   
        {"pct": 0.20, "ctx": 16384}    
    ]
}

def create_seq_idx_batch(cu_seqlens_padded, n_segs, seqlen):
    """Create per-batch-element seq_idx from padded cu_seqlens.

    Args:
        cu_seqlens_padded: (batch_size, max_n_segs) — padded with -1 sentinels
        n_segs:            (batch_size,)            — real lengths per element
        seqlen:            int — current context length

    Returns:
        seq_idx: (batch_size, seqlen) — int32 tensor on DEVICE
    """
    batch_size = cu_seqlens_padded.shape[0]
    seq_idx = torch.zeros(batch_size, seqlen, dtype=torch.int32, device=DEVICE)
    for b in range(batch_size):
        n = n_segs[b].item()
        cu = cu_seqlens_padded[b, :n]
        # Filter to boundaries within the current (possibly truncated) context
        valid = cu[cu <= seqlen]
        if len(valid) == 0 or valid[-1] != seqlen:
            valid = torch.cat([valid, torch.tensor([seqlen], dtype=torch.int32)])
        for i in range(len(valid) - 1):
            start, end = valid[i].item(), valid[i + 1].item()
            seq_idx[b, start:end] = i
    return seq_idx

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    wandb.init(project="Agentic-1.58b-Model", name=f"run-bitmamba-{MODE}", config=CURRICULUM_CONFIG)
    
    print(f"Initializing {MODE.upper()} BitMamba Model...")
    model = BitMambaLLM(**MODEL_CONFIG).to(DEVICE)
    
    muon_opt, adam_opt, mamba_core_opt = setup_mamba_optimizers(model, CURRICULUM_CONFIG)
    scheduler = FGWSD_Scheduler(muon_opt, adam_opt, mamba_core_opt, TOTAL_STEPS, CURRICULUM_CONFIG)
    
    train_loader, _ = create_dataloaders(TRAIN_CONFIG, tokenizer_path="custom_agentic_tokenizer", max_seq_len=16384, batch_size=BATCH_SIZE)
    model.train()
    data_iter = iter(train_loader)
    total_tokens, t0 = 0, time.time()
    
    for step in range(TOTAL_STEPS):
        current_lr, current_ctx, phase_name = scheduler.step(step)
        for opt in [muon_opt, adam_opt, mamba_core_opt]: opt.zero_grad()
        accumulated_loss = 0.0
        
        for _ in range(GRAD_ACCUM_STEPS):
            try: x, y, cu_seqlens, n_segs = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y, cu_seqlens, n_segs = next(data_iter)
            
            x, y = x.to(DEVICE)[:, :current_ctx], y.to(DEVICE)[:, :current_ctx]
            seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, current_ctx)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x, seq_idx=seq_idx)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1)) / GRAD_ACCUM_STEPS
            loss.backward()
            accumulated_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        for opt in [muon_opt, adam_opt, mamba_core_opt]: opt.step()
        
        if step % 10 == 0:
            t1 = time.time()
            dt, t0 = t1 - t0, t1
            tokens_this_step = BATCH_SIZE * GRAD_ACCUM_STEPS * current_ctx
            total_tokens += tokens_this_step 
            print(f"Step {step:06d} | {phase_name:<15} | Loss: {accumulated_loss:.4f} | Tok/s: {tokens_this_step/dt:.0f}")
            wandb.log({"Train/Loss": accumulated_loss, "System/Total_Tokens": total_tokens}, step=step)

        if step > 0 and step % SAVE_EVERY == 0:
            torch.save({'step': step, 'model_state_dict': model.state_dict()}, os.path.join(CHECKPOINT_DIR, f"step_{step:06d}.pt"))

    wandb.finish()

if __name__ == "__main__":
    main()

