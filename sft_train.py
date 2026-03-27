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
from model import BitMambaLLM, chunked_cross_entropy
from optim import setup_mamba_optimizers
from sft_data import SFT_STAGES, create_sft_dataloader
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED_CKPT = "checkpoints/bitmamba_parent/step_1000000.pt" 
CHECKPOINT_DIR = "checkpoints/sft"
MODEL_CONFIG = dict(vocab_size=64000, dim=1024, n_layers=40, d_state=128, expand=2)

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8


def run_sft_stage(model, tokenizer, stage_cfg, stage_num, global_step):
    """Run one SFT stage: create dataloader, optimizer, and train for N epochs."""
    name = stage_cfg["name"]
    lr = stage_cfg["lr"]
    epochs = stage_cfg["epochs"]

    train_loader = create_sft_dataloader(
        stage_cfg["paths"], tokenizer,
        max_seq_len=stage_cfg["max_seq_len"],
        batch_size=BATCH_SIZE,
        reasoning_off_prob=stage_cfg["reasoning_off_prob"],
    )

    muon_opt, adam_opt, mamba_opt = setup_mamba_optimizers(model, {"peak_lr": lr, "end_lr": lr * 0.1})
    total_optim_steps = len(train_loader) * epochs // GRAD_ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_opt, T_max=max(total_optim_steps, 1), eta_min=lr * 0.1)

    print(f"\n{'='*60}")
    print(f"SFT Stage {stage_num}: {name} | epochs={epochs} lr={lr} samples={len(train_loader.dataset)}")
    print(f"{'='*60}")

    model.train()
    for epoch in range(epochs):
        for opt in [muon_opt, adam_opt, mamba_opt]: opt.zero_grad()
        accumulated_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                hidden = model.forward_hidden(x, seq_idx=None)
                loss = chunked_cross_entropy(hidden, model.output, y[..., 1:], ignore_index=-100) / GRAD_ACCUM_STEPS
            loss.backward()
            accumulated_loss += loss.item()
            
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
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
                
                if global_step % 10 == 0:
                    wandb.log({
                        f"SFT/{name}_Loss": accumulated_loss,
                        "SFT/Stage": stage_num,
                        "SFT/LR": current_lr,
                    }, step=global_step)
                accumulated_loss = 0.0
                global_step += 1
                
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"sft_stage{stage_num}_{name}_epoch{epoch+1}.pt")
        torch.save({'stage': stage_num, 'epoch': epoch, 'model_state_dict': model.state_dict()}, ckpt_path)
        print(f"  Epoch {epoch+1}/{epochs} done → {ckpt_path}")

    return global_step


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    wandb.init(project="Agentic-1.58b-Model", name="run-sft-bitmamba-3stage")
    
    tokenizer = AutoTokenizer.from_pretrained("custom_agentic_tokenizer")
    model = BitMambaLLM(**MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(PRETRAINED_CKPT, map_location=DEVICE)['model_state_dict'])
    
    global_step = 0
    for stage_num, stage_cfg in enumerate(SFT_STAGES, start=1):
        global_step = run_sft_stage(model, tokenizer, stage_cfg, stage_num, global_step)
    
    # Save final checkpoint
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(CHECKPOINT_DIR, "sft_final.pt"))
    print(f"\nSFT complete. Final checkpoint → {CHECKPOINT_DIR}/sft_final.pt")
    wandb.finish()

if __name__ == "__main__":
    main()

