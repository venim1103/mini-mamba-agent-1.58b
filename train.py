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

import os
# Set CUDA allocator config to reduce fragmentation on 16GB GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
import time
import wandb
from model import BitMambaLLM, maybe_autocast
from data import create_dataloaders
from optim import setup_mamba_optimizers, FGWSD_Scheduler
from dist_utils import (
    setup_distributed, cleanup_distributed, is_main_process,
    wrap_model_ddp, unwrap_model, barrier,
)

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # overridden by setup_distributed()
BATCH_SIZE = 2             
GRAD_ACCUM_STEPS = 8       
SAVE_EVERY = 5000

# G10: Use FP16 + GradScaler on Ampere (RTX 3090) for 2x throughput.
# Set to torch.bfloat16 on Ada Lovelace (RTX 4090) where BF16 == FP16 speed.
AMP_DTYPE = torch.float16
CHECKPOINT_DIR = f"checkpoints/bitmamba_{MODE}"
# Static divisor to keep loss_sum in a safe range for the FP16 GradScaler.
# Must be a fixed constant — do not use current_ctx here, as it changes between
# FG-WSD phases and would cause logged loss values to jump discontinuously.
SAFE_DIVISOR = 16384.0 * BATCH_SIZE

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
        {"name": "code",         "path": "local_data/train/code",  "format": "parquet", "weight": 0.25},
        {"name": "web",          "path": "local_data/train/web",   "format": "parquet", "weight": 0.30},
        {"name": "tool_use",     "path": "local_data/train/tools", "format": "parquet", "weight": 0.20}
    ],
    "Phase_2": [  # Stable 1 - heavy on web/diversity
        {"name": "formal_logic", "path": "local_data/train/logic", "format": "parquet", "weight": 0.20},
        {"name": "code",         "path": "local_data/train/code",  "format": "parquet", "weight": 0.20},
        {"name": "web",          "path": "local_data/train/web",   "format": "parquet", "weight": 0.45},
        {"name": "tool_use",     "path": "local_data/train/tools", "format": "parquet", "weight": 0.15}
    ],
    "Phase_3": [  # Stable 2 - heavy on code/logic + synthetic CoT (Nanbeige4-3B)
        {"name": "formal_logic", "path": "local_data/train/logic", "format": "parquet", "weight": 0.30},
        {"name": "code",         "path": "local_data/train/code",  "format": "parquet", "weight": 0.25},
        {"name": "synth_qa",     "path": "local_data/synth/web_qa", "format": "json",    "weight": 0.20},
        {"name": "synth_distill","path": "local_data/synth/code_distill","format": "json","weight": 0.15},
        {"name": "web",          "path": "local_data/train/web",   "format": "parquet", "weight": 0.05},
        {"name": "tool_use",     "path": "local_data/train/tools", "format": "parquet", "weight": 0.05}
    ],
    "Phase_4": [  # Decay - 100% high-quality reasoning/synthetic (Nemotron-H)
        {"name": "formal_logic", "path": "local_data/train/logic", "format": "parquet", "weight": 0.25},
        {"name": "code",         "path": "local_data/train/code",  "format": "parquet", "weight": 0.20},
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


def run_training_steps(model, raw_model, optimizers, scheduler,
                       train_loader, scaler, total_steps,
                       checkpoint_dir, device, world_size=1):
    """Inner training loop, decoupled from DDP setup, wandb init, and torch.compile.

    Args:
        model:          DDP-wrapped (or bare) model for forward passes
        raw_model:      Unwrapped model for state_dict saves and output head access
        optimizers:     Tuple of (muon_opt, adam_opt, mamba_core_opt)
        scheduler:      FGWSD_Scheduler instance
        train_loader:   Initial DataLoader (recreated internally on phase change)
        scaler:         torch.amp.GradScaler (disabled on CPU)
        total_steps:    Number of optimizer steps to run
        checkpoint_dir: Directory to write step checkpoints into
        device:         String device identifier ('cpu' or 'cuda:N')
        world_size:     Number of distributed ranks (1 for single-GPU)
    """
    muon_opt, adam_opt, mamba_core_opt = optimizers
    data_iter = iter(train_loader)
    total_tokens = 0
    t0 = time.time()
    previous_ctx = scheduler.get_lr_and_ctx(0)[1]
    previous_phase = "Phase_1"

    for step in range(total_steps):
        current_lr, current_ctx, phase_name = scheduler.step(step)

        need_reload = False
        if phase_name != previous_phase and phase_name != "Complete" and previous_phase is not None:
            need_reload = True
            if is_main_process():
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
        accumulated_loss_sum = 0.0
        accumulated_valid_tokens = 0

        for _ in range(GRAD_ACCUM_STEPS):
            # Tell CUDAGraphs a new forward/backward cycle is starting.
            # Prevents RuntimeError about overwriting static graph memory during grad accumulation.
            torch.compiler.cudagraph_mark_step_begin()

            x, y, cu_seqlens, n_segs = next(data_iter)

            x, y = x.to(device)[:, :current_ctx], y.to(device)[:, :current_ctx]
            seq_idx = create_seq_idx_batch(cu_seqlens, n_segs, current_ctx)

            with maybe_autocast(device, amp_dtype=AMP_DTYPE):
                loss_sum, valid_tokens = model(x, seq_idx=seq_idx, targets=y)
            safe_loss = loss_sum / SAFE_DIVISOR
            scaler.scale(safe_loss).backward()
            accumulated_loss_sum += loss_sum.detach().item()
            accumulated_valid_tokens += valid_tokens.detach().item()

        for opt in [muon_opt, adam_opt, mamba_core_opt]: scaler.unscale_(opt)
        global_valid_tokens = accumulated_valid_tokens
        if world_size > 1:
            valid_token_tensor = torch.tensor(accumulated_valid_tokens, device=device, dtype=torch.float32)
            torch.distributed.all_reduce(valid_token_tensor, op=torch.distributed.ReduceOp.SUM)
            global_valid_tokens = valid_token_tensor.item()
        grad_scale = (world_size * SAFE_DIVISOR) / max(global_valid_tokens, 1.0)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(grad_scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(muon_opt)
        scaler.step(adam_opt)
        scaler.step(mamba_core_opt)
        scaler.update()

        for opt in [muon_opt, adam_opt, mamba_core_opt]: opt.zero_grad()

        accumulated_loss = accumulated_loss_sum * grad_scale

        tokens_this_step = BATCH_SIZE * GRAD_ACCUM_STEPS * current_ctx
        total_tokens += tokens_this_step

        if step % 10 == 0 and is_main_process():
            t1 = time.time()
            dt, t0 = t1 - t0, t1
            toks = tokens_this_step * world_size
            print(f"Step {step:06d} | {phase_name:<15} | Loss: {accumulated_loss:.4f} | Tok/s: {toks/dt:.0f}")
            wandb.log({"Train/Loss": accumulated_loss, "System/Total_Tokens": total_tokens * world_size}, step=step)

        if step > 0 and step % SAVE_EVERY == 0 and is_main_process():
            torch.save({'step': step, 'model_state_dict': raw_model.state_dict()}, os.path.join(checkpoint_dir, f"step_{step:06d}.pt"))
        barrier()


def main():
    global DEVICE
    rank, local_rank, world_size, device = setup_distributed()
    DEVICE = device  # update module-level DEVICE for create_seq_idx_batch

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if is_main_process():
        wandb.init(project="Agentic-1.58b-Model", name=f"run-bitmamba-{MODE}", config=CURRICULUM_CONFIG)
    
    if is_main_process():
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
    
    # IMPORTANT: torch.compile is compatible with gradient checkpointing ONLY when
    # use_reentrant=False (set in BitMambaLLM._backbone). Do NOT change checkpointing
    # to use_reentrant=True — it will silently corrupt gradients under compile.
    # Note: Using mode="default" instead of "reduce-overhead" to disable CUDA Graphs,
    # which conflict with gradient checkpointing + accumulation.
    model._backbone = torch.compile(model._backbone, mode="default")

    # Wrap in DDP after compile but before optimizer creation
    model = wrap_model_ddp(model, local_rank)
    raw_model = unwrap_model(model)  # for state_dict saves and output head access
    
    muon_opt, adam_opt, mamba_core_opt = setup_mamba_optimizers(raw_model, CURRICULUM_CONFIG)
    scheduler = FGWSD_Scheduler(muon_opt, adam_opt, mamba_core_opt, TOTAL_STEPS, CURRICULUM_CONFIG)
    
    # Start with Phase_1 (warmup) data config for FG-WSD
    initial_ctx = CURRICULUM_CONFIG['phases'][0]['ctx']
    train_loader, tokenizer = create_dataloaders(
        TRAIN_CONFIGS["Phase_1"], tokenizer_path="custom_agentic_tokenizer", 
        max_seq_len=initial_ctx, batch_size=BATCH_SIZE
    )
    model.train()
    # G10: GradScaler for FP16 (no-op when using BF16)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.startswith("cuda") and AMP_DTYPE == torch.float16))
    
    run_training_steps(
        model=model,
        raw_model=raw_model,
        optimizers=(muon_opt, adam_opt, mamba_core_opt),
        scheduler=scheduler,
        train_loader=train_loader,
        scaler=scaler,
        total_steps=TOTAL_STEPS,
        checkpoint_dir=CHECKPOINT_DIR,
        device=DEVICE,
        world_size=world_size,
    )

    if is_main_process():
        wandb.finish()
    cleanup_distributed()

if __name__ == "__main__":
    main()

