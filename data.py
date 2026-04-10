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
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from context_config import CONTEXT_LENGTH

def get_text_column(dataset_stream):
    sample_row = next(iter(dataset_stream))
    for name in ['text', 'content', 'trajectory', 'prompt', 'response', 'question']:
        if name in sample_row and isinstance(sample_row[name], str): return name
    for key, value in sample_row.items():
        if isinstance(value, str): return key
    raise ValueError("Could not detect text column.")

def create_infinite_stream(hf_dataset):
    while True:
        for item in hf_dataset: yield item

def extract_text_from_row(row):
    """Dynamically parses complex schemas into a clean training string."""
    parts = []
    if 'messages' in row and isinstance(row['messages'], list):
        for msg in row['messages']:
            if isinstance(msg, dict) and 'content' in msg:
                parts.append(f"{msg.get('role', 'user')}: {msg['content']}")
    
    for key in ['problem', 'question', 'prompt', 'facts', 'hypothesis']:
        if key in row and isinstance(row[key], str):
            parts.append(f"{key.capitalize()}: {row[key]}")
            
    if 'proofs' in row and isinstance(row['proofs'], list):
        parts.append(f"Proof: {' '.join(row['proofs'])}")
        
    for key in ['solution', 'answer', 'response']:
        if key in row and isinstance(row[key], str):
            parts.append(f"{key.capitalize()}: {row[key]}")
            
    for key in ['text', 'content', 'trajectory']:
        if key in row and isinstance(row[key], str) and not parts:
            parts.append(row[key])
            
    if parts:
        return "\n".join(parts)
        
    # Absolute fallback
    best_val = ""
    for val in row.values():
        if isinstance(val, str) and len(val) > len(best_val):
            best_val = val
    return best_val

def packed_token_stream(dataset_stream, tokenizer, text_column, max_seq_len):
    buffer = []
    doc_lengths = []
    
    for row in dataset_stream:
        # Dynamically extract all available text to handle complex schemas
        text = extract_text_from_row(row)
        if not text: continue
        
        # We intentionally do not truncate individual documents here because this
        # stream packs and slices to `max_seq_len` downstream. Disable the HF
        # "sequence longer than model_max_length" warning to avoid noisy logs.
        tokens = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"] + [tokenizer.eos_token_id]
        buffer.extend(tokens)
        doc_lengths.append(len(tokens))
        
        while len(buffer) >= max_seq_len + 1:
            chunk = buffer[:max_seq_len + 1]
            cu_seqlens = [0]
            current_len = 0
            
            while len(doc_lengths) > 0 and current_len + doc_lengths[0] <= max_seq_len:
                current_len += doc_lengths.pop(0)
                cu_seqlens.append(current_len)
                
            if current_len < max_seq_len and len(doc_lengths) > 0:
                # Partial document fills the remainder of the max_seq_len window
                cu_seqlens.append(max_seq_len)
                remainder = max_seq_len - current_len
                doc_lengths[0] -= remainder
            
            # Fix #6: The chunk consumes max_seq_len + 1 tokens from the buffer
            # (the extra +1 is the overlap token for x/y pair construction).
            # cu_seqlens only tracks max_seq_len tokens, so we must account for
            # the 1 extra token consumed from the trailing document.
            if len(doc_lengths) > 0:
                doc_lengths[0] -= 1
                if doc_lengths[0] <= 0:
                    doc_lengths.pop(0)
            
            buffer = buffer[max_seq_len + 1:]
            yield (
                torch.tensor(chunk[:-1], dtype=torch.long), 
                torch.tensor(chunk[1:], dtype=torch.long),
                torch.tensor(cu_seqlens, dtype=torch.int32)
            )

class AgenticDataMixture(IterableDataset):
    def __init__(self, streams_dict, target_proportions):
        super().__init__()
        self.stream_names = list(streams_dict.keys())
        self.streams = streams_dict
        self.weights = target_proportions
    def __iter__(self):
        while True:
            name = random.choices(self.stream_names, weights=self.weights, k=1)[0]
            yield next(self.streams[name])

def packed_collate_fn(batch):
    """Custom collate that handles variable-length cu_seqlens across batch elements.

    Each sample is (x, y, cu_seqlens) where x and y have fixed length but
    cu_seqlens varies.  We pad cu_seqlens to the longest in the batch with -1
    sentinel values and return a lengths tensor so consumers know where the
    real values end.

    Returns:
        x:            (batch_size, seq_len)      — LongTensor
        y:            (batch_size, seq_len)      — LongTensor
        cu_seqlens:   (batch_size, max_n_segs)   — Int32Tensor, padded with -1
        n_segs:       (batch_size,)              — Int32Tensor, real lengths
    """
    xs, ys, cu_list = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)

    lengths = [cs.shape[0] for cs in cu_list]
    max_len = max(lengths)

    padded = torch.full((len(cu_list), max_len), -1, dtype=torch.int32)
    for i, cs in enumerate(cu_list):
        padded[i, :cs.shape[0]] = cs

    n_segs = torch.tensor(lengths, dtype=torch.int32)
    return x, y, padded, n_segs


def create_dataloaders(datasets_config, tokenizer_path="custom_agentic_tokenizer", max_seq_len=CONTEXT_LENGTH, batch_size=2):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    packed_streams, weights = {}, []
    
    # Flatten config to handle subdirectories individually and prevent mixed schema CastErrors
    flat_config = []
    for config in datasets_config:
        name, target_path, fmt, weight = config["name"], config["path"], config["format"], config["weight"]
        if os.path.isdir(target_path):
            # Only include subdirectories that actually contain files of the target format
            subdirs = [
                d for d in os.listdir(target_path) 
                if os.path.isdir(os.path.join(target_path, d)) 
                and any(f.endswith(f'.{fmt}') for _, _, files in os.walk(os.path.join(target_path, d)) for f in files)
            ]
            if subdirs:
                sub_weight = weight / len(subdirs)
                for d in subdirs:
                    flat_config.append({
                        "name": f"{name}_{d}",
                        "path": os.path.join(target_path, d),
                        "format": fmt,
                        "weight": sub_weight
                    })
                continue
        flat_config.append(config)
            
    for config in flat_config:
        name, target_path, fmt, weight = config["name"], config["path"], config["format"], config["weight"]
        data_files = os.path.join(target_path, f"**/*.{fmt}") if os.path.isdir(target_path) else target_path
        
        raw_dataset = load_dataset(fmt, data_files=data_files, split='train', streaming=True)
        infinite_raw_stream = create_infinite_stream(raw_dataset)
        # Pass None for text_column since packed_token_stream now uses extract_text_from_row
        packed_streams[name] = iter(packed_token_stream(infinite_raw_stream, tokenizer, None, max_seq_len))
        weights.append(weight)

    mixture_dataset = AgenticDataMixture(packed_streams, weights)
    return DataLoader(
        mixture_dataset, batch_size=batch_size, num_workers=0,
        pin_memory=True, collate_fn=packed_collate_fn,
    ), tokenizer
