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
from model import BitMambaLLM, chunked_cross_entropy
from data import create_dataloaders
from optim import setup_mamba_optimizers, FGWSD_Scheduler

MODE = "scout" 

if MODE == "scout":
    MODEL_CONFIG = dict(vocab_size=64000, dim=512, n_layers=24, d_state=64, expand=2, use_checkpoint=True)
    TOTAL_STEPS = 100_000 
    PEAK_LR = 4.5e-4 
elif MODE == "upscaled":
    # Continued pre-training after SOLAR upscaling (MiniPuzzle-inspired)
    # Lower LR since model already has pretrained weights; shorter run
    MODEL_CONFIG = dict(vocab_size=64000, dim=1024, n_layers=64, d_state=128, expand=2, 
                       use_checkpoint=True, use_attn=True, attn_pct=0.08)
    TOTAL_STEPS = 20_000  # Short continued pretrain (5-10B tokens equivalent)
    PEAK_LR = 1.0e-4  # Lower LR for continued training
else:
    MODEL_CONFIG = dict(vocab_size=64000, dim=1024, n_layers=40, d_state=128, expand=2, use_checkpoint=True)
    TOTAL_STEPS = 1_000_000 
    PEAK_LR = 3.0e-4 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2             
GRAD_ACCUM_STEPS = 8       
SAVE_EVERY = 5000

# G10: Use FP16 + GradScaler on Ampere (RTX 3090) for 2x throughput.
# Set to torch.bfloat16 on Ada Lovelace (RTX 4090) where BF16 == FP16 speed.
AMP_DTYPE = torch.float16
CHECKPOINT_DIR = f"checkpoints/bitmamba_{MODE}"

# FG-WSD: Data quality progression per phase (Nanbeige4-3B §2.2.2)
# Keep LR flat while progressively increasing data quality
# Warmup: Mixed | Stable 1: Web-heavy | Stable 2: Code/Logic-heavy | Decay: HQ reasoning only
# Synthetic data integrated in Stable 2 and Decay (Nemotron-H §2.3, Nanbeige4-3B §2.2)
#
# PRE-TRAINING: Generate synthetic data first using synth_data.py:
#   python synth_data.py --strategy diverse_qa --input local_data/train/web --output local_data/synth/web_qa
#   python synth_data.py --strategy distill --input local_data/train/code --output local_data/synth/code_distill
#   python synth_data.py --strategy extract --input local_data/train/web --output local_data/synth/knowledge_extract
#   python synth_data.py --strategy rephrase --input local_data/train/web --output local_data/synth/web_rephrased
#
TRAIN_CONFIGS = {
    "Phase_1": [  # Warmup - diverse mixed data
        {"name": "formal_logic", "path": "local_data/train/logic", "format": "parquet", "weight": 0.25},
        {"name": "code",         "path": "local_data/train/code",  "format": "json",    "weight": 0.25},
        {"name": "web",          "path": "local_data/train/web",   "format": "parquet", "weight": 0.30},
        {"name": "tool_use",     "path": "local_data/train/tools", "format": "json",    "weight": 0.20}
    ],
    "Phase_2": [  # Stable 1 - heavy on web/diversity
        {"name": "formal_logic", "path": "local_data/train/logic", "format": "parquet", "weight": 0.20},
        {"name": "code",         "path": "local_data/train/code",  "format": "json",    "weight": 0.20},
        {"name": "web",          "path": "local_data/train/web",   "format": "parquet", "weight": 0.45},
        {"name": "tool_use",     "path": "local_data/train/tools", "format": "json",    "weight": 0.15}
    ],
    "Phase_3": [  # Stable 2 - heavy on code/logic + synthetic CoT (Nanbeige4-3B)
        {"name": "formal_logic", "path": "local_data/train/logic", "format": "parquet", "weight": 0.30},
        {"name": "code",         "path": "local_data/train/code",  "format": "json",    "weight": 0.25},
        {"name": "synth_qa",     "path": "local_data/synth/web_qa", "format": "json",    "weight": 0.20},
        {"name": "synth_distill","path": "local_data/synth/code_distill","format": "json","weight": 0.15},
        {"name": "web",          "path": "local_data/train/web",   "format": "parquet", "weight": 0.05},
        {"name": "tool_use",     "path": "local_data/train/tools", "format": "json",    "weight": 0.05}
    ],
    "Phase_4": [  # Decay - 100% high-quality reasoning/synthetic (Nemotron-H)
        {"name": "formal_logic", "path": "local_data/train/logic", "format": "parquet", "weight": 0.25},
        {"name": "code",         "path": "local_data/train/code",  "format": "json",    "weight": 0.20},
        {"name": "synth_qa",     "path": "local_data/synth/web_qa", "format": "json",    "weight": 0.20},
        {"name": "synth_distill","path": "local_data/synth/code_distill","format": "json","weight": 0.15},
        {"name": "synth_extract","path": "local_data/synth/knowledge_extract","format": "json","weight": 0.10},
        {"name": "synth_rephrase","path": "local_data/synth/web_rephrased","format": "json","weight": 0.10}
    ],
}

# Legacy single config (used for backward compatibility if needed)
TRAIN_CONFIG = TRAIN_CONFIGS["Phase_2"]

# UPDATED: Fixed context at 8192 for first 80% (stable phases), only expand to 16384 in decay
# Per Nanbeige4-3B and Nemotron-H: expanding context during stable training ruins dynamics
CURRICULUM_CONFIG = {
    "peak_lr": PEAK_LR, "end_lr": 1.5e-6,
    "phases": [
        {"pct": 0.05, "ctx": 8192},   # warmup
        {"pct": 0.40, "ctx": 8192},   # stable 1
        {"pct": 0.35, "ctx": 8192},   # stable 2
        {"pct": 0.20, "ctx": 16384}   # decay (extend context here!)
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
    
    # Load upscaled checkpoint if in continued pretraining mode
    if MODE == "upscaled":
        print("Loading upscaled checkpoint for continued pre-training...")
        # Look for upscaled checkpoint in default location
        upscale_ckpt = "checkpoints/upscaled/step_000000_1B_mamba.pt"
        if os.path.exists(upscale_ckpt):
            ckpt = torch.load(upscale_ckpt, map_location=DEVICE)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded upscaled weights from {upscale_ckpt}")
        else:
            print(f"Warning: Upscaled checkpoint not found at {upscale_ckpt}")
            print("Please run: python upscale.py first")
    
    # G12: torch.compile the shared backbone so both forward() and forward_hidden() benefit
    model._backbone = torch.compile(model._backbone, mode="reduce-overhead")
    
    muon_opt, adam_opt, mamba_core_opt = setup_mamba_optimizers(model, CURRICULUM_CONFIG)
    scheduler = FGWSD_Scheduler(muon_opt, adam_opt, mamba_core_opt, TOTAL_STEPS, CURRICULUM_CONFIG)
    
    # Start with Phase_1 (warmup) data config for FG-WSD
    initial_ctx = CURRICULUM_CONFIG['phases'][0]['ctx']
    train_loader, tokenizer = create_dataloaders(
        TRAIN_CONFIGS["Phase_1"], tokenizer_path="custom_agentic_tokenizer", 
        max_seq_len=initial_ctx, batch_size=BATCH_SIZE
    )
    model.train()
    data_iter = iter(train_loader)
    total_tokens, t0 = 0, time.time()
    previous_ctx = initial_ctx
    previous_phase = "Phase_1"  # Track phase changes for FG-WSD data quality progression
    # G10: GradScaler for FP16 (no-op when using BF16)
    scaler = torch.amp.GradScaler(enabled=(AMP_DTYPE == torch.float16))
    
    for step in range(TOTAL_STEPS):
        current_lr, current_ctx, phase_name = scheduler.step(step)
        
        # FG-WSD: Recreate DataLoader when phase or context window changes
        need_reload = False
        if phase_name != previous_phase and phase_name != "Complete" and previous_phase is not None:
            need_reload = True
            print(f"  [FG-WSD] Phase changed to {phase_name}: data quality updated")
            wandb.log({"System/FG_WSD_Phase": phase_name}, step=step)
        if current_ctx != previous_ctx:
            need_reload = True
        
        if need_reload:
            train_loader, _ = create_dataloaders(
                TRAIN_CONFIGS.get(phase_name, TRAIN_CONFIG), tokenizer_path="custom_agentic_tokenizer",
                max_seq_len=current_ctx, batch_size=BATCH_SIZE
            )
            data_iter = iter(train_loader)
            previous_ctx = current_ctx
        
        previous_phase = phase_name
        
        for opt in [muon_opt, adam_opt, mamba_core_opt]: opt.zero_grad()
        accumulated_loss = 0.0
        
        for _ in range(GRAD_ACCUM_STEPS):
            try: x, y, cu_seqlens, n_segs = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y, cu_seqlens, n_segs = next(data_iter)
            
            x, y = x.to(DEVICE)[:, :current_ctx], y.to(DEVICE)[:, :current_ctx]
            seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, current_ctx)
            
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE):
                # G2: Chunked cross-entropy avoids materializing full [BS, ctx, vocab] logits
                hidden = model.forward_hidden(x, seq_idx=seq_idx)
                loss = chunked_cross_entropy(hidden, model.output, y) / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            
        for opt in [muon_opt, adam_opt, mamba_core_opt]: scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        for opt in [muon_opt, adam_opt, mamba_core_opt]: scaler.step(opt)
        scaler.update()
        
        # Fix #7: Count tokens every step, not just every 10 steps
        tokens_this_step = BATCH_SIZE * GRAD_ACCUM_STEPS * current_ctx
        total_tokens += tokens_this_step
        
        if step % 10 == 0:
            t1 = time.time()
            dt, t0 = t1 - t0, t1
            print(f"Step {step:06d} | {phase_name:<15} | Loss: {accumulated_loss:.4f} | Tok/s: {tokens_this_step/dt:.0f}")
            wandb.log({"Train/Loss": accumulated_loss, "System/Total_Tokens": total_tokens}, step=step)

        if step > 0 and step % SAVE_EVERY == 0:
            torch.save({'step': step, 'model_state_dict': model.state_dict()}, os.path.join(CHECKPOINT_DIR, f"step_{step:06d}.pt"))

    wandb.finish()

if __name__ == "__main__":
    main()

