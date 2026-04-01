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
import wandb
from model import BitMambaLLM, chunked_cross_entropy, maybe_autocast
from optim import setup_mamba_optimizers
from sft_data import SFT_STAGES, create_sft_dataloader
from transformers import AutoTokenizer
from dist_utils import (
    setup_distributed, cleanup_distributed, is_main_process,
    wrap_model_ddp, unwrap_model, barrier, get_world_size,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # overridden by setup_distributed()
PRETRAINED_CKPT = "checkpoints/bitmamba_parent/step_1000000.pt" 
CHECKPOINT_DIR = "checkpoints/sft"
MODEL_CONFIG = dict(vocab_size=64000, dim=1024, n_layers=40, d_state=128, expand=2, use_checkpoint=True)

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8


def run_sft_stage(model, tokenizer, stage_cfg, stage_num, global_step):
    """Run one SFT stage: create dataloader, optimizer, and train for N epochs."""
    raw_model = unwrap_model(model)
    name = stage_cfg["name"]
    lr = stage_cfg["lr"]
    epochs = stage_cfg["epochs"]

    train_loader = create_sft_dataloader(
        stage_cfg["paths"], tokenizer,
        max_seq_len=stage_cfg["max_seq_len"],
        batch_size=BATCH_SIZE,
        reasoning_off_prob=stage_cfg["reasoning_off_prob"],
    )

    muon_opt, adam_opt, mamba_opt = setup_mamba_optimizers(raw_model, {"peak_lr": lr, "end_lr": lr * 0.1})
    total_optim_steps = len(train_loader) * epochs // GRAD_ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_opt, T_max=max(total_optim_steps, 1), eta_min=lr * 0.1)

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"SFT Stage {stage_num}: {name} | epochs={epochs} lr={lr} samples={len(train_loader.dataset)}")
        print(f"{'='*60}")

    model.train()
    for epoch in range(epochs):
        # DistributedSampler must be told the epoch so it shuffles differently each time
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        for opt in [muon_opt, adam_opt, mamba_opt]: opt.zero_grad()
        accumulated_loss = 0.0
        accumulated_loss_sum = 0.0
        accumulated_valid_tokens = 0
        n_batches = len(train_loader)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            with maybe_autocast(DEVICE):
                hidden = model.forward_hidden(x, seq_idx=None)
                loss_sum, valid_tokens = chunked_cross_entropy(
                    hidden[:, :-1, :],
                    raw_model.output,
                    y[..., 1:],
                    ignore_index=-100,
                    return_stats=True,
                )
            loss_sum.backward()
            accumulated_loss_sum += loss_sum.detach().item()
            accumulated_valid_tokens += valid_tokens.detach().item()
            
            should_step = ((batch_idx + 1) % GRAD_ACCUM_STEPS == 0) or (batch_idx + 1 == n_batches)
            if should_step:
                world_size = get_world_size()
                global_valid_tokens = accumulated_valid_tokens
                if world_size > 1:
                    valid_token_tensor = torch.tensor(accumulated_valid_tokens, device=DEVICE, dtype=torch.float32)
                    torch.distributed.all_reduce(valid_token_tensor, op=torch.distributed.ReduceOp.SUM)
                    global_valid_tokens = valid_token_tensor.item()
                grad_scale = world_size / max(global_valid_tokens, 1.0)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(grad_scale)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Fix #14: Call scheduler.step() before get_last_lr() to avoid warning
                scheduler.step()
                # Sync LRs across optimizer groups
                current_lr = scheduler.get_last_lr()[0]
                for opt in [muon_opt, adam_opt]:
                    for group in opt.param_groups: group['lr'] = current_lr
                for group in mamba_opt.param_groups: group['lr'] = current_lr * 0.1
                
                for opt in [muon_opt, adam_opt, mamba_opt]: opt.step()
                for opt in [muon_opt, adam_opt, mamba_opt]: opt.zero_grad()
                accumulated_loss = accumulated_loss_sum * grad_scale
                
                if global_step % 10 == 0 and is_main_process():
                    wandb.log({
                        f"SFT/{name}_Loss": accumulated_loss,
                        "SFT/Stage": stage_num,
                        "SFT/LR": current_lr,
                    }, step=global_step)
                accumulated_loss = 0.0
                accumulated_loss_sum = 0.0
                accumulated_valid_tokens = 0
                global_step += 1
                
        if is_main_process():
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"sft_stage{stage_num}_{name}_epoch{epoch+1}.pt")
            torch.save({'stage': stage_num, 'epoch': epoch, 'model_state_dict': raw_model.state_dict()}, ckpt_path)
            print(f"  Epoch {epoch+1}/{epochs} done → {ckpt_path}")
        barrier()

    return global_step


def main():
    global DEVICE
    rank, local_rank, world_size, device = setup_distributed()
    DEVICE = device

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if is_main_process():
        wandb.init(project="Agentic-1.58b-Model", name="run-sft-bitmamba-3stage")
    
    tokenizer = AutoTokenizer.from_pretrained("custom_agentic_tokenizer")
    model = BitMambaLLM(**MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(PRETRAINED_CKPT, map_location=DEVICE)['model_state_dict'])

    model = wrap_model_ddp(model, local_rank)
    raw_model = unwrap_model(model)
    
    global_step = 0
    for stage_num, stage_cfg in enumerate(SFT_STAGES, start=1):
        global_step = run_sft_stage(model, tokenizer, stage_cfg, stage_num, global_step)
    
    # Save final checkpoint
    if is_main_process():
        torch.save({'model_state_dict': raw_model.state_dict()}, os.path.join(CHECKPOINT_DIR, "sft_final.pt"))
        print(f"\nSFT complete. Final checkpoint → {CHECKPOINT_DIR}/sft_final.pt")
        wandb.finish()
    cleanup_distributed()

if __name__ == "__main__":
    main()

