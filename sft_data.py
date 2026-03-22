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
import os
import random
import re
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class SFTChatDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=4096, reasoning_off_prob=0.3):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.reasoning_off_prob = reasoning_off_prob
        
        self.sys_reasoning_on = "You are a deductive reasoning agent. You must analyze the user's request step-by-step within <think> tags before acting."
        self.sys_reasoning_off = "You are a direct, concise agent. Provide the final answer immediately without internal monologue."
        
        if os.path.isdir(data_path):
            files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('.jsonl', '.json', '.parquet'))]
            fmt = "parquet" if files[0].endswith(".parquet") else "json"
            self.raw_data = load_dataset(fmt, data_files=files, split="train")
        else:
            fmt = "parquet" if data_path.endswith(".parquet") else "json"
            self.raw_data = load_dataset(fmt, data_files=data_path, split="train")
            
        self.im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        self.im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        self.nl = tokenizer.encode("\n", add_special_tokens=False)[0]

    def __len__(self): return len(self.raw_data)

    def __getitem__(self, idx):
        messages = self.raw_data[idx].get("messages", self.raw_data[idx].get("conversations", []))
        input_ids, labels = [], []
        
        strip_reasoning = random.random() < self.reasoning_off_prob
        system_prompt = self.sys_reasoning_off if strip_reasoning else self.sys_reasoning_on
        messages = [{"role": "system", "content": system_prompt}] + messages
        
        for msg in messages:
            role = msg.get("role", msg.get("from", ""))
            content = msg.get("content", msg.get("value", ""))
            if role in ["human", "user"]: role = "user"
            if role in ["gpt", "assistant"]: role = "assistant"
            
            if role == "assistant" and strip_reasoning:
                content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
            
            header_tokens = [self.im_start] + self.tokenizer.encode(role, add_special_tokens=False) + [self.nl]
            content_tokens = self.tokenizer.encode(content, add_special_tokens=False) + [self.im_end, self.nl]
            msg_tokens = header_tokens + content_tokens
            input_ids.extend(msg_tokens)
            
            if role == "user" or role == "system":
                labels.extend([-100] * len(msg_tokens))
            elif role == "assistant":
                labels.extend([-100] * len(header_tokens) + content_tokens)

        input_ids = input_ids[:self.max_seq_len]
        labels = labels[:self.max_seq_len]
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)
            
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def create_sft_dataloader(data_path, tokenizer_path="custom_agentic_tokenizer", max_seq_len=4096, batch_size=2):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = SFTChatDataset(data_path, tokenizer, max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4), tokenizer

