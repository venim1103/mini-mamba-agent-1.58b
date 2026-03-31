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

import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
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
MAX_BATCH_EXAMPLES = int(os.getenv("TOKENIZER_MAX_BATCH_EXAMPLES", "128"))
MAX_BATCH_CHARACTERS = int(os.getenv("TOKENIZER_MAX_BATCH_CHARACTERS", "2000000"))
PARQUET_BATCH_SIZE = int(os.getenv("TOKENIZER_PARQUET_BATCH_SIZE", "256"))
MAX_TEXT_CHARACTERS = int(os.getenv("TOKENIZER_MAX_TEXT_CHARACTERS", "0"))

# These must be preserved as single tokens so the model can reason and format perfectly!
SPECIAL_TOKENS = [
    "<|im_start|>", 
    "<|im_end|>", 
    "<think>", 
    "</think>",
    "<|eos|>",
    "<|pad|>"
]

SUPPORTED_SUFFIXES = (".jsonl", ".json", ".parquet")

# ==========================================
# 2. RAM-Friendly Data Iterator
# ==========================================
def get_text_column(sample_row):
    """Automatically detects which column holds the text data."""
    for name in ['text', 'content', 'trajectory', 'prompt', 'response']:
        if name in sample_row and isinstance(sample_row[name], str):
            return name
    for key, value in sample_row.items():
        if isinstance(value, str):
            return key
    return None

def maybe_trim_text(text):
    if not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    if MAX_TEXT_CHARACTERS > 0:
        return text[:MAX_TEXT_CHARACTERS]

    return text

def iter_data_files():
    for root, _, files in os.walk(DATA_DIR):
        for file_name in sorted(files):
            if file_name.endswith(SUPPORTED_SUFFIXES):
                yield os.path.join(root, file_name)

def iter_text_from_rows(rows, source_name):
    text_col = None

    for row in rows:
        if text_col is None:
            text_col = get_text_column(row)
            if not text_col:
                print(f"  -> Skipping {source_name} (No text column found)")
                return

        text = maybe_trim_text(row.get(text_col))
        if text:
            yield text

def iter_jsonl_rows(file_path):
    with open(file_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  -> Skipping malformed JSON line {line_number} in {file_path}: {exc}")
                continue

            if isinstance(row, dict):
                yield row

def iter_json_texts(file_path):
    dataset = load_dataset("json", data_files=file_path, split="train", streaming=True)
    yield from iter_text_from_rows(dataset, os.path.basename(file_path))

def iter_parquet_texts(file_path):
    parquet_file = pq.ParquetFile(file_path)

    # Detect the text column from the schema so we only read that single column,
    # avoiding materialising every column into Python memory.
    # Only consider columns whose Arrow type is string/large_string so the
    # fallback branch of get_text_column cannot accidentally pick an int column.
    string_col_names = [
        field.name for field in parquet_file.schema_arrow
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    ]
    sample_row = {name: "" for name in string_col_names}
    text_col = get_text_column(sample_row)
    if not text_col:
        print(f"  -> Skipping {os.path.basename(file_path)} (No text column found)")
        return

    for batch in parquet_file.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=[text_col]):
        for text in batch.column(text_col).to_pylist():
            text = maybe_trim_text(text)
            if text:
                yield text

def iter_file_texts(file_path):
    print(f"Streaming data from {file_path}...")

    if file_path.endswith(".jsonl"):
        yield from iter_text_from_rows(iter_jsonl_rows(file_path), os.path.basename(file_path))
        return

    if file_path.endswith(".parquet"):
        yield from iter_parquet_texts(file_path)
        return

    yield from iter_json_texts(file_path)

def batch_iterator(max_batch_examples=MAX_BATCH_EXAMPLES, max_batch_characters=MAX_BATCH_CHARACTERS):
    """
    Streams local data files with bounded buffering so RAM usage stays stable.
    """
    print(f"Scanning {DATA_DIR} for datasets...")

    batch = []
    batch_characters = 0

    for file_path in iter_data_files():
        for text in iter_file_texts(file_path):
            batch.append(text)
            batch_characters += len(text)

            if len(batch) >= max_batch_examples or batch_characters >= max_batch_characters:
                yield batch
                batch = []
                batch_characters = 0

    if batch:
        yield batch

# ==========================================
# 3. Training Execution
# ==========================================
def main():
    print(f"Loading template tokenizer ({TEMPLATE_TOKENIZER})...")
    # Load the base tokenizer to inherit its underlying BPE structure and code-regex rules
    old_tokenizer = AutoTokenizer.from_pretrained(TEMPLATE_TOKENIZER)
    
    print(
        f"Training new vocabulary of size {VOCAB_SIZE} with up to "
        f"{MAX_BATCH_EXAMPLES} examples / {MAX_BATCH_CHARACTERS} characters buffered at once."
    )
    
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

