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
from model import BitMambaLLM, maybe_autocast
from optim import setup_mamba_optimizers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SFT_CKPT = "checkpoints/sft/sft_final.pt"
LOCAL_RL_DATA = "local_data/rl/reasoning" 
CHECKPOINT_DIR = "checkpoints/rl"
MODEL_CONFIG = dict(vocab_size=64000, dim=1024, n_layers=40, d_state=128, expand=2, use_checkpoint=True)

BATCH_SIZE = 1
GROUP_SIZE = 8             # Increased from 4 (Nanbeige4-3B / Llama-Nemotron recommend 8-16)
TOTAL_STEPS = 10_000
PEAK_LR = 1e-6
FILTER_LOW = 0.10          # On-policy filtering: discard if pass_rate < 10%
FILTER_HIGH = 0.90          # On-policy filtering: discard if pass_rate > 90%
FILTER_BATCH = 64           # Number of problems to evaluate per filtering round
MAX_GEN_TOKENS = 512


def collect_data_files(root_dir):
    files = []
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for name in filenames:
            if name.endswith((".jsonl", ".json", ".parquet")):
                files.append(os.path.join(root, name))
    return files

# ---------------------------------------------------------------------------
# Reward functions — separated into format and accuracy (Llama-Nemotron §5.1)
# ---------------------------------------------------------------------------

def compute_format_reward(completion):
    """Check structural format: <think>...</think> followed by answer."""
    thought_text, final_answer = _extract_thought_and_answer(completion)
    if thought_text is None:
        return 0.0
    if final_answer:
        return 1.0
    return 0.5   # tags present but no answer after


def compute_accuracy_reward(completion, ground_truth):
    """Check if the extracted final answer contains the ground truth."""
    _, final_answer = _extract_thought_and_answer(completion)
    if not final_answer:
        return 0.0
    if ground_truth.lower() in final_answer.lower():
        return 2.0
    return 0.0


def compute_conciseness_penalty(completion):
    """Penalize verbose thinking relative to answer length."""
    thought_text, final_answer = _extract_thought_and_answer(completion)
    if not thought_text or not final_answer:
        return 0.0
    thought_ratio = len(thought_text) / max(1, len(final_answer))
    if thought_ratio > 10.0:
        return -0.5
    return 0.0


def _extract_thought_and_answer(completion):
    """Extract thought and final answer from strict <think>...</think> format."""
    if "<think>" not in completion:
        return None, None

    after_open = completion.split("<think>", 1)[1]

    if "</think>" not in after_open:
        return None, None

    thought_text, answer_part = after_open.split("</think>", 1)
    thought_text = thought_text.strip()
    final_answer = answer_part.strip()
    return thought_text, final_answer


def compute_rewards(completions, ground_truth):
    """Combined reward: format + accuracy + conciseness + length penalty."""
    rewards = []
    for comp in completions:
        r_format = compute_format_reward(comp)
        r_accuracy = compute_accuracy_reward(comp, ground_truth)
        r_concise = compute_conciseness_penalty(comp)
        r_length = -len(comp) * 0.0001
        rewards.append(r_format + r_accuracy + r_concise + r_length)
    return torch.tensor(rewards, dtype=torch.float32).to(DEVICE)

# ---------------------------------------------------------------------------
# On-policy difficulty filtering (Nanbeige4-3B §3.4)
# ---------------------------------------------------------------------------

@torch.no_grad()
def filter_problems_on_policy(model, tokenizer, problems, eos_id):
    """Pre-pass: compute per-problem pass rate, keep only those in [FILTER_LOW, FILTER_HIGH]."""
    filtered = []
    model.eval()
    
    for sample in problems:
        question = sample.get('problem', sample.get('question', ''))
        ground_truth = str(sample.get('expected_answer', sample.get('answer', sample.get('solution', ''))))
        
        # Use pre-computed pass_rate from dataset if available (OpenMathReasoning)
        precomputed = sample.get('pass_rate_72b_tir', 'n/a')
        if precomputed not in ('n/a', None, ''):
            try:
                pass_rate = float(precomputed)
                if FILTER_LOW <= pass_rate <= FILTER_HIGH:
                    sample['_pass_rate'] = pass_rate
                    filtered.append(sample)
                continue
            except (ValueError, TypeError):
                pass
        
        sys_prompt = "You are a deductive reasoning agent. You must analyze the user's request step-by-step within <think> tags before acting."
        prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        
        # Quick pass-rate estimate with GROUP_SIZE generations
        n_correct = 0
        for _ in range(GROUP_SIZE):
            generated = model.generate(input_ids, max_new_tokens=MAX_GEN_TOKENS, temperature=0.8,
                                       do_sample=True, eos_token_id=eos_id)
            gen_text = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
            if compute_accuracy_reward(gen_text, ground_truth) > 0:
                n_correct += 1
        
        pass_rate = n_correct / GROUP_SIZE
        if FILTER_LOW <= pass_rate <= FILTER_HIGH:
            sample['_pass_rate'] = pass_rate
            filtered.append(sample)
    
    return filtered


def run_rl_steps(model, tokenizer, dataset, optimizers, eos_id,
                 total_steps, checkpoint_dir, device):
    """Inner GRPO training loop, decoupled from model loading and wandb init."""
    muon_opt, adam_opt, mamba_opt = optimizers
    data_iter = iter(dataset)
    step = 0

    while step < total_steps:
        raw_batch = []
        for _ in range(FILTER_BATCH):
            try: raw_batch.append(next(data_iter))
            except StopIteration:
                data_iter = iter(dataset)
                raw_batch.append(next(data_iter))
        
        filtered = filter_problems_on_policy(model, tokenizer, raw_batch, eos_id)
        if not filtered:
            print(f"  Step {step}: filter returned 0 problems, retrying...")
            continue
        
        filtered.sort(key=lambda s: -s['_pass_rate'])
        print(f"  Step {step}: filtered {len(filtered)}/{FILTER_BATCH} problems (pass_rate range: "
              f"{filtered[-1]['_pass_rate']:.0%}–{filtered[0]['_pass_rate']:.0%})")
        
        for sample in filtered:
            if step >= total_steps:
                break
                
            question = sample.get('problem', sample.get('question', ''))
            ground_truth = str(sample.get('expected_answer', sample.get('answer', sample.get('solution', ''))))
            
            sys_prompt = "You are a deductive reasoning agent. You must analyze the user's request step-by-step within <think> tags before acting."
            prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            if device.startswith("cuda"):
                for opt in [muon_opt, adam_opt, mamba_opt]:
                    for state in opt.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor): state[k] = v.cpu()
                torch.cuda.empty_cache()
            
            model.eval()
            completions_ids, completions_text, old_log_probs_list = [], [], []
            with torch.no_grad():
                for _ in range(GROUP_SIZE):
                    generated = model.generate(input_ids, max_new_tokens=MAX_GEN_TOKENS, temperature=0.8,
                                               do_sample=True, eos_token_id=eos_id)
                    gen_only = generated[0][input_ids.shape[1]:]
                    completions_ids.append(gen_only.cpu())
                    completions_text.append(tokenizer.decode(gen_only, skip_special_tokens=True))
                    
                    full_seq = torch.cat([input_ids[0], gen_only]).unsqueeze(0)
                    with maybe_autocast(device):
                        hidden = model.forward_hidden(full_seq, seq_idx=None)
                        hidden_slice = hidden[0:1, input_ids.shape[1]-1:-1, :]
                        logits = model.output(hidden_slice)
                        log_probs = -F.cross_entropy(
                            logits[0, :, :].contiguous(),
                            gen_only.contiguous(), reduction='none'
                        )
                    old_log_probs_list.append(log_probs.detach())
                    
            rewards = compute_rewards(completions_text, ground_truth)
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            if device.startswith("cuda"):
                for opt in [muon_opt, adam_opt, mamba_opt]:
                    for state in opt.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor): state[k] = v.to(device)
            
            model.train()
            for opt in [muon_opt, adam_opt, mamba_opt]: opt.zero_grad()
            policy_loss = 0.0
            EPS = 0.2
            
            for i in range(GROUP_SIZE):
                comp_ids = completions_ids[i].to(device)
                old_log_probs = old_log_probs_list[i].to(device)
                
                full_seq = torch.cat([input_ids[0], comp_ids]).unsqueeze(0)
                with maybe_autocast(device):
                    hidden = model.forward_hidden(full_seq, seq_idx=None)
                    hidden_slice = hidden[0:1, input_ids.shape[1]-1:-1, :]
                    logits = model.output(hidden_slice)
                    log_probs = -F.cross_entropy(
                        logits[0, :, :].contiguous(),
                        comp_ids.contiguous(), reduction='none'
                    )
                    
                    ratio = torch.exp(log_probs - old_log_probs)
                    surr1 = ratio * advantages[i]
                    surr2 = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS) * advantages[i]
                    loss = -torch.min(surr1, surr2).mean() / GROUP_SIZE
                loss.backward()
                policy_loss += loss.item()
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in [muon_opt, adam_opt, mamba_opt]: opt.step()
            
            if step % 10 == 0:
                wandb.log({
                    "RL/Mean_Reward": rewards.mean().item(),
                    "RL/Policy_Loss": policy_loss,
                    "RL/Format_Reward": sum(compute_format_reward(c) for c in completions_text) / len(completions_text),
                    "RL/Pass_Rate": sample['_pass_rate'],
                }, step=step)
            if step > 0 and step % 1000 == 0:
                torch.save({'step': step, 'model_state_dict': model.state_dict()},
                           os.path.join(checkpoint_dir, f"rl_step_{step:06d}.pt"))
            step += 1
    
    torch.save({'step': step, 'model_state_dict': model.state_dict()},
               os.path.join(checkpoint_dir, "rl_final.pt"))


def main():
    global DEVICE

    # RL is single-GPU only (GRPO generation is inherently sequential).
    # Skip DDP init entirely to avoid collective-operation deadlocks.
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        print(f"[rl_train] Rank {rank}: RL is single-GPU only, exiting.")
        return

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    wandb.init(project="Agentic-1.58b-Model", name="run-rl-grpo")
    
    tokenizer = AutoTokenizer.from_pretrained("custom_agentic_tokenizer")
    eos_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    
    files = collect_data_files(LOCAL_RL_DATA)
    if not files:
        raise FileNotFoundError(f"No .json/.jsonl/.parquet files found under {LOCAL_RL_DATA}")

    parquet_files = [f for f in files if f.endswith(".parquet")]
    json_files = [f for f in files if f.endswith((".json", ".jsonl"))]
    if parquet_files:
        dataset = load_dataset("parquet", data_files=parquet_files, split="train", streaming=True)
    else:
        dataset = load_dataset("json", data_files=json_files, split="train", streaming=True)
    data_iter = iter(dataset)
    
    model = BitMambaLLM(**MODEL_CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(SFT_CKPT, map_location=DEVICE)['model_state_dict'])
    muon_opt, adam_opt, mamba_opt = setup_mamba_optimizers(model, {"peak_lr": PEAK_LR, "end_lr": 1e-6})

    run_rl_steps(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        optimizers=(muon_opt, adam_opt, mamba_opt),
        eos_id=eos_id,
        total_steps=TOTAL_STEPS,
        checkpoint_dir=CHECKPOINT_DIR,
        device=DEVICE,
    )

    wandb.finish()


if __name__ == "__main__":
    main()

