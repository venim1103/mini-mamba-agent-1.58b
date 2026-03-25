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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset


# ---------------------------------------------------------------------------
# 3-Stage SFT Configuration (Llama-Nemotron Nano §4.2, Nanbeige4-3B §3.1-3.2)
# ---------------------------------------------------------------------------
# Stage 1 (Cold-start): reasoning-only data, multiple epochs, no toggle
# Stage 2 (Mixed): reasoning + general chat, reasoning toggle training
# Stage 3 (Polish): tool-use focus, low epoch count
# ---------------------------------------------------------------------------
SFT_STAGES = [
    {
        "name": "cold_start",
        "paths": [
            {"path": "local_data/sft/reasoning/open-math-reasoning", "format": "parquet"},
            {"path": "local_data/sft/reasoning/nemotron-post-training", "format": "jsonl"},
            {"path": "local_data/sft/reasoning/openr1-math", "format": "parquet"},
        ],
        "epochs": 4,
        "lr": 1e-4,
        "reasoning_off_prob": 0.0,   # reasoning always ON
        "max_seq_len": 4096,
    },
    {
        "name": "mixed",
        "paths": [
            {"path": "local_data/sft/reasoning/open-math-reasoning", "format": "parquet"},
            {"path": "local_data/sft/reasoning/nemotron-post-training", "format": "jsonl"},
            {"path": "local_data/sft/reasoning/openr1-math", "format": "parquet"},
            {"path": "local_data/sft/mixed/smol-smoltalk", "format": "parquet"},
        ],
        "epochs": 2,
        "lr": 5e-5,
        "reasoning_off_prob": 0.3,   # 30% reasoning toggle
        "max_seq_len": 4096,
    },
    {
        "name": "polish",
        "paths": [
            {"path": "local_data/sft/tool_calling/apigen-fc", "format": "parquet"},
            {"path": "local_data/sft/tool_calling/xlam-irrelevance", "format": "json"},
        ],
        "epochs": 2,
        "lr": 2e-5,
        "reasoning_off_prob": 0.1,   # mostly reasoning ON for tool planning
        "max_seq_len": 4096,
    },
]


class SFTChatDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=4096, reasoning_off_prob=0.3, format_hint=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.reasoning_off_prob = reasoning_off_prob
        
        self.sys_reasoning_on = "You are a deductive reasoning agent. You must analyze the user's request step-by-step within <think> tags before acting."
        self.sys_reasoning_off = "You are a direct, concise agent. Provide the final answer immediately without internal monologue."
        
        if os.path.isdir(data_path):
            files = []
            for root, _, filenames in os.walk(data_path):
                for f in filenames:
                    if f.endswith((".jsonl", ".json", ".parquet")):
                        files.append(os.path.join(root, f))
            if not files:
                raise FileNotFoundError(f"No .json/.jsonl/.parquet files found under {data_path}")

            parquet_files = [f for f in files if f.endswith(".parquet")]
            jsonl_files = [f for f in files if f.endswith(".jsonl")]
            json_files = [f for f in files if f.endswith((".json", ".jsonl"))]

            # Explicit per-source routing for mixed-format directories.
            if format_hint == "parquet":
                if not parquet_files:
                    raise FileNotFoundError(f"No .parquet files found under {data_path}")
                self.raw_data = load_dataset("parquet", data_files=parquet_files, split="train")
            elif format_hint == "jsonl":
                if not jsonl_files:
                    raise FileNotFoundError(f"No .jsonl files found under {data_path}")
                self.raw_data = load_dataset("json", data_files=jsonl_files, split="train")
            elif format_hint == "json":
                json_only_files = [f for f in files if f.endswith(".json")]
                if not json_only_files:
                    raise FileNotFoundError(f"No .json files found under {data_path}")
                self.raw_data = load_dataset("json", data_files=json_only_files, split="train")
            elif parquet_files:
                # Default behavior: prefer parquet when no format hint is provided.
                self.raw_data = load_dataset("parquet", data_files=parquet_files, split="train")
            else:
                self.raw_data = load_dataset("json", data_files=json_files, split="train")
        else:
            if format_hint == "parquet":
                fmt = "parquet"
            elif format_hint in ("json", "jsonl"):
                fmt = "json"
            else:
                fmt = "parquet" if data_path.endswith(".parquet") else "json"
            self.raw_data = load_dataset(fmt, data_files=data_path, split="train")
            
        self.im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        self.im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        self.nl = tokenizer.encode("\n", add_special_tokens=False)[0]

    def __len__(self): return len(self.raw_data)

    def _row_to_messages(self, row):
        """Auto-detect data format and convert to a list of chat messages."""
        # Format 1: Chat — already has messages/conversations
        if "messages" in row and row["messages"]:
            return row["messages"]
        if "conversations" in row and row["conversations"]:
            return row["conversations"]

        # Format 2: Math reasoning — problem + solution (OpenMathReasoning, OpenR1-Math)
        problem = row.get("problem") or row.get("question") or row.get("prompt", "")
        solution = row.get("generated_solution") or row.get("solution") or row.get("response", "")
        if problem and solution:
            return [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": f"<think>\n{solution}\n</think>"},
            ]

        # Format 3: Tool-calling — query + tools + answers (xLAM, APIGen)
        query = row.get("query", "")
        tools = row.get("tools", "")
        answers = row.get("answers", "")
        if query and tools:
            tool_desc = tools if isinstance(tools, str) else str(tools)
            answer_str = answers if isinstance(answers, str) else str(answers)
            return [
                {"role": "system", "content": f"You have access to the following tools:\n{tool_desc}"},
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer_str},
            ]

        # Fallback: try to find any text pair
        for q_key in ["instruction", "input"]:
            for a_key in ["output", "response"]:
                if q_key in row and a_key in row:
                    return [
                        {"role": "user", "content": str(row[q_key])},
                        {"role": "assistant", "content": str(row[a_key])},
                    ]
        return []

    def __getitem__(self, idx):
        row = self.raw_data[idx]
        messages = self._row_to_messages(row)
        if not messages:
            # Return an empty-ish sample that the collate fn can pad
            return torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long), torch.tensor([-100], dtype=torch.long)

        input_ids, labels = [], []
        
        strip_reasoning = random.random() < self.reasoning_off_prob

        # Only inject system prompt if the messages don't already start with one
        # (tool-calling format injects its own system prompt with tool defs)
        if not messages or messages[0].get("role") != "system":
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
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def sft_collate_fn(batch):
    """Dynamic padding to the longest sequence in the batch (not global max)."""
    input_ids_list, labels_list = zip(*batch)
    max_len = max(ids.size(0) for ids in input_ids_list)
    
    padded_ids, padded_labels = [], []
    for ids, lbl in zip(input_ids_list, labels_list):
        pad_len = max_len - ids.size(0)
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
            lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        padded_ids.append(ids)
        padded_labels.append(lbl)
    
    return torch.stack(padded_ids), torch.stack(padded_labels)


def create_sft_dataloader(data_paths, tokenizer, max_seq_len=4096, batch_size=2, reasoning_off_prob=0.3):
    """Create a DataLoader from one or more data directories/files."""
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    datasets = []
    for source in data_paths:
        if isinstance(source, dict):
            path = source["path"]
            format_hint = source.get("format")
        else:
            path = source
            format_hint = None
        datasets.append(SFTChatDataset(path, tokenizer, max_seq_len, reasoning_off_prob, format_hint=format_hint))
    
    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    return DataLoader(
        combined, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=4, collate_fn=sft_collate_fn
    )

