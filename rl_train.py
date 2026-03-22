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
from datasets import load_dataset
from transformers import AutoTokenizer
from model import BitMambaLLM
from optim import setup_mamba_optimizers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SFT_CKPT = "checkpoints/sft/sft_epoch_2.pt" 
LOCAL_RL_DATA = "local_data/rl" 
CHECKPOINT_DIR = "checkpoints/rl"
MODEL_CONFIG = dict(vocab_size=64000, dim=1024, n_layers=40, d_state=128, expand=2)

BATCH_SIZE, GROUP_SIZE, TOTAL_STEPS, PEAK_LR = 1, 4, 10_000, 1e-6             

def compute_rewards(completions, ground_truth):
    rewards = []
    for comp in completions:
        reward, pass_rate = 0.0, 0
        if "<think>" in comp and "</think>" in comp:
            reward += 0.5
            final_answer = comp.split("</think>")[-1].strip()
            if ground_truth.lower() in final_answer.lower():
                reward += 2.0
                pass_rate = 1
                
        if pass_rate == 1: 
            parts = comp.split("</think>")
            thought_text = parts[0].replace("<think>", "").strip()
            final_answer = parts[-1].strip()
            thought_ratio = len(thought_text) / max(1, len(final_answer))
            
            if thought_ratio > 10.0: reward -= 0.5 
            else: reward += (1.0 / max(1.0, thought_ratio)) 
            reward -= len(comp) * 0.0001
            
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    wandb.init(project="Agentic-1.58b-Model", name="run-rl-grpo")
    
    tokenizer = AutoTokenizer.from_pretrained("custom_agentic_tokenizer")
    files = [os.path.join(LOCAL_RL_DATA, f) for f in os.listdir(LOCAL_RL_DATA) if f.endswith(('.jsonl', '.json', '.parquet'))]
    fmt = "parquet" if files[0].endswith(".parquet") else "json"
    dataset = load_dataset(fmt, data_files=files, split="train", streaming=True)
    data_iter = iter(dataset)
    
    model = BitMambaLLM(**MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(SFT_CKPT, map_location=DEVICE)['model_state_dict'])
    muon_opt, adam_opt, mamba_opt = setup_mamba_optimizers(model, {"peak_lr": PEAK_LR, "end_lr": 1e-6})
    
    for step in range(TOTAL_STEPS):
        try: sample = next(data_iter)
        except StopIteration:
            data_iter = iter(dataset)
            sample = next(data_iter)
            
        question = sample.get('question', sample.get('problem', ''))
        ground_truth = str(sample.get('answer', sample.get('solution', '')))
        
        sys_prompt = "You are a deductive reasoning agent. You must analyze the user's request step-by-step within <think> tags before acting."
        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        
        model.eval()
        completions_ids, completions_text = [], []
        with torch.no_grad():
            for _ in range(GROUP_SIZE):
                generated = model.generate(input_ids, max_new_tokens=512, temperature=0.8, do_sample=True, eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0])
                gen_only = generated[0][input_ids.shape[1]:]
                completions_ids.append(gen_only)
                completions_text.append(tokenizer.decode(gen_only, skip_special_tokens=True))
                
        rewards = compute_rewards(completions_text, ground_truth)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        model.train()
        for opt in [muon_opt, adam_opt, mamba_opt]: opt.zero_grad()
        policy_loss = 0.0
        
        for i in range(GROUP_SIZE):
            full_seq = torch.cat([input_ids[0], completions_ids[i]]).unsqueeze(0)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(full_seq, seq_idx=None)
                log_probs = -F.cross_entropy(logits[0, input_ids.shape[1]-1 : -1, :].contiguous(), completions_ids[i].contiguous(), reduction='none')
                loss = - (log_probs * advantages[i]).mean() / GROUP_SIZE
            loss.backward()
            policy_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in [muon_opt, adam_opt, mamba_opt]: opt.step()
        
        if step % 10 == 0: wandb.log({"RL/Mean_Reward": rewards.mean().item(), "RL/Policy_Loss": policy_loss}, step=step)
        if step > 0 and step % 1000 == 0: torch.save({'step': step, 'model_state_dict': model.state_dict()}, os.path.join(CHECKPOINT_DIR, f"rl_step_{step:06d}.pt"))
    wandb.finish()

@torch.no_grad()
def generate_wrapper(model, input_ids, max_new_tokens, temperature, do_sample, eos_token_id):
    curr_ids = input_ids
    for _ in range(max_new_tokens):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16): logits = model(curr_ids, seq_idx=None)
        next_token = torch.multinomial(F.softmax(logits[0, -1, :] / temperature, dim=-1), num_samples=1)
        curr_ids = torch.cat([curr_ids, next_token.unsqueeze(0)], dim=-1)
        if next_token.item() == eos_token_id: break
    return curr_ids

BitMambaLLM.generate = generate_wrapper

if __name__ == "__main__":
    main()

