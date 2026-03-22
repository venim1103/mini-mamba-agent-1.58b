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
from datasets import load_dataset
from transformers import AutoTokenizer

# ==========================================
# 1. Tokenizer Configuration
# ==========================================
# We use DeepSeek-Coder as the "template" because it contains highly optimized 
# Regular Expressions (Regex) for splitting code, indentations, and whitespace properly.
# We are NOT keeping its vocabulary; we are just borrowing its splitting rules.
TEMPLATE_TOKENIZER = "deepseek-ai/deepseek-coder-1.3b-base"

# The "Goldilocks" size for a 500M parameter model
VOCAB_SIZE = 64_000 

DATA_DIR = "local_data/train"
OUTPUT_DIR = "custom_agentic_tokenizer"

# These must be preserved as single tokens so the model can reason and format perfectly!
SPECIAL_TOKENS = [
    "<|im_start|>", 
    "<|im_end|>", 
    "<think>", 
    "</think>",
    "<|eos|>",
    "<|pad|>"
]

# ==========================================
# 2. RAM-Friendly Data Iterator
# ==========================================
def get_text_column(sample_row):
    """Automatically detects which column holds the text data."""
    for name in ['text', 'content', 'trajectory', 'prompt', 'response']:
        if name in sample_row and isinstance(sample_row[name], str):
            return name
    for key, value in sample_row.items():
        if isinstance(value, str): return key
    return None

def batch_iterator(batch_size=10000):
    """
    Streams data from your local pre-training folders in batches.
    This prevents the script from crashing your RAM!
    """
    print(f"Scanning {DATA_DIR} for datasets...")
    
    # Iterate through logic, code, web, and tools folders
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(('.jsonl', '.json', '.parquet')):
                file_path = os.path.join(root, file)
                fmt = "parquet" if file.endswith(".parquet") else "json"
                
                print(f"Streaming data from {file}...")
                dataset = load_dataset(fmt, data_files=file_path, split="train", streaming=True)
                
                # Get the text column name from the first row
                sample_row = next(iter(dataset))
                text_col = get_text_column(sample_row)
                
                if not text_col:
                    print(f"  -> Skipping {file} (No text column found)")
                    continue
                
                # Yield batches of text to the tokenizer trainer
                batch = []
                for row in dataset:
                    if row[text_col]:
                        batch.append(row[text_col])
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch

# ==========================================
# 3. Training Execution
# ==========================================
def main():
    print(f"Loading template tokenizer ({TEMPLATE_TOKENIZER})...")
    # Load the base tokenizer to inherit its underlying BPE structure and code-regex rules
    old_tokenizer = AutoTokenizer.from_pretrained(TEMPLATE_TOKENIZER)
    
    print(f"Training new vocabulary of size {VOCAB_SIZE}. This will use your CPU and may take 15-45 minutes...")
    
    # Train the new tokenizer purely on your local data stream!
    new_tokenizer = old_tokenizer.train_new_from_iterator(
        text_iterator=batch_iterator(),
        vocab_size=VOCAB_SIZE,
        new_special_tokens=SPECIAL_TOKENS
    )
    
    # Set the pad and eos tokens
    new_tokenizer.pad_token = "<|pad|>"
    new_tokenizer.eos_token = "<|eos|>"
    
    print(f"Saving custom tokenizer to ./{OUTPUT_DIR} ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    new_tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Done! Your model's vocabulary is now perfectly mathematically tuned to your data.")

if __name__ == "__main__":
    main()

