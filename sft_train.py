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
from model import BitMambaLLM
from optim import setup_mamba_optimizers
from sft_data import create_sft_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED_CKPT = "checkpoints/bitmamba_parent/step_1000000.pt" 
LOCAL_SFT_DATA = "local_data/sft" 
CHECKPOINT_DIR = "checkpoints/sft"
MODEL_CONFIG = dict(vocab_size=64000, dim=1024, n_layers=40, d_state=128, expand=2)

BATCH_SIZE, GRAD_ACCUM_STEPS, EPOCHS, PEAK_LR = 2, 8, 2, 2e-5             

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    wandb.init(project="Agentic-1.58b-Model", name="run-sft-bitmamba")
    
    model = BitMambaLLM(**MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(PRETRAINED_CKPT, map_location=DEVICE)['model_state_dict'])
    
    train_loader, _ = create_sft_dataloader(LOCAL_SFT_DATA, tokenizer_path="custom_agentic_tokenizer", max_seq_len=4096, batch_size=BATCH_SIZE)
    
    muon_opt, adam_opt, mamba_opt = setup_mamba_optimizers(model, {"peak_lr": PEAK_LR, "end_lr": 1e-6})
    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam_opt, T_max=total_steps, eta_min=1e-6)
    
    model.train()
    step = 0
    for epoch in range(EPOCHS):
        for opt in [muon_opt, adam_opt, mamba_opt]: opt.zero_grad()
        accumulated_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # seq_idx=None during SFT assumes 1 document per sequence batch
                logits = model(x, seq_idx=None) 
                loss = F.cross_entropy(logits[..., :-1, :].contiguous().view(-1, logits.size(-1)), y[..., 1:].contiguous().view(-1)) / GRAD_ACCUM_STEPS
            loss.backward()
            accumulated_loss += loss.item()
            
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for opt in [muon_opt, adam_opt]:
                    for group in opt.param_groups: group['lr'] = scheduler.get_last_lr()[0]
                for group in mamba_opt.param_groups: group['lr'] = scheduler.get_last_lr()[0] * 0.1
                
                for opt in [muon_opt, adam_opt, mamba_opt]: opt.step()
                scheduler.step()
                for opt in [muon_opt, adam_opt, mamba_opt]: opt.zero_grad()
                
                if step % 10 == 0: wandb.log({"SFT/Loss": accumulated_loss}, step=step)
                accumulated_loss = 0.0
                step += 1
                
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(CHECKPOINT_DIR, f"sft_epoch_{epoch+1}.pt"))
    wandb.finish()

if __name__ == "__main__":
    main()

