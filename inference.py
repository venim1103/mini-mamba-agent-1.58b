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
from transformers import AutoTokenizer
from model import BitMambaLLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODE = "scout" 

if MODE == "scout":
    MODEL_CONFIG = dict(vocab_size=64000, dim=512, n_layers=24, d_state=64, expand=2)
    CKPT_PATH = "checkpoints/bitmamba_scout/step_100000.pt" 
else:
    MODEL_CONFIG = dict(vocab_size=64000, dim=1024, n_layers=40, d_state=128, expand=2)
    CKPT_PATH = "checkpoints/bitmamba_parent/step_1000000.pt" 

def generate(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    eos_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    
    print(f"\nPrompt: {prompt}")
    print("Generating...", flush=True)
    
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=True, eos_token_id=eos_id
        )
    
    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
    print(generated_text)
    print()

if __name__ == "__main__":
    print("Loading custom tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("custom_agentic_tokenizer")
    
    print(f"Loading {MODE.upper()} BitMamba model from {CKPT_PATH}...")
    model = BitMambaLLM(**MODEL_CONFIG).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE)['model_state_dict'])
    except FileNotFoundError:
        print(f"Error: Could not find {CKPT_PATH}.")
        exit(1)
    
    print("\n--- Testing Model Logic ---")
    chat_prompt = "<|im_start|>system\nYou are a deductive reasoning agent. You must analyze the user's request step-by-step within <think> tags before acting.<|im_end|>\n<|im_start|>user\nIf I have 3 apples and eat 1, how many are left?<|im_end|>\n<|im_start|>assistant\n<think>\n"
    generate(model, tokenizer, prompt=chat_prompt, max_new_tokens=150)

