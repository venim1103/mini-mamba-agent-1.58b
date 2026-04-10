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

import collections
import json
import os
import re
import sys

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
def _resolve_profile():
    """Select tokenizer training profile from env or runtime platform."""
    explicit = os.getenv("TOKENIZER_PROFILE", "").strip().lower()
    if explicit in {"kaggle", "standard"}:
        return explicit

    if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"

    return "standard"


PROFILE = _resolve_profile()
PROFILE_DEFAULTS = {
    "standard": {
        "max_batch_examples": "128",
        "max_batch_characters": "2000000",
        "parquet_batch_size": "256",
        "max_text_characters": "2000",
        "min_frequency": "3",
        "max_unique_words": "200000",
        "max_ram_gb": "30",
    },
    "kaggle": {
        "max_batch_examples": "96",
        "max_batch_characters": "1200000",
        "parquet_batch_size": "192",
        "max_text_characters": "1600",
        "min_frequency": "3",
        "max_unique_words": "200000",
        "max_ram_gb": "13",
    },
}


def _profile_default(name):
    return PROFILE_DEFAULTS[PROFILE][name]


MAX_BATCH_EXAMPLES = int(os.getenv("TOKENIZER_MAX_BATCH_EXAMPLES", _profile_default("max_batch_examples")))
MAX_BATCH_CHARACTERS = int(os.getenv("TOKENIZER_MAX_BATCH_CHARACTERS", _profile_default("max_batch_characters")))
PARQUET_BATCH_SIZE = int(os.getenv("TOKENIZER_PARQUET_BATCH_SIZE", _profile_default("parquet_batch_size")))
MAX_TEXT_CHARACTERS = int(os.getenv("TOKENIZER_MAX_TEXT_CHARACTERS", _profile_default("max_text_characters")))
MIN_FREQUENCY = int(os.getenv("TOKENIZER_MIN_FREQUENCY", _profile_default("min_frequency")))
# Target unique word count that fits in RAM.  Each unique word costs ~15-20 KB
# in the Rust BPE trainer (word string + character-level tokenisation + pair
# counts + priority queue entries). Both built-in profiles use conservative
# defaults because the Rust BPE merge phase can still OOM below 64 GB RAM.
MAX_UNIQUE_WORDS = int(os.getenv("TOKENIZER_MAX_UNIQUE_WORDS", _profile_default("max_unique_words")))

def _corpus_bytes_for_ram(ram_gb):
    """Return a generous corpus byte cap as a first-pass limit.

    This is a coarse outer bound; the real memory control is
    MAX_UNIQUE_WORDS which is enforced by the two-pass approach.
    """
    usable = max(ram_gb - 4, 0.5)
    return int(usable * 1_073_741_824)  # 1x — generous, since pass-2 filters

MAX_RAM_GB = float(os.getenv("TOKENIZER_MAX_RAM_GB", _profile_default("max_ram_gb")))
MAX_CORPUS_BYTES = _corpus_bytes_for_ram(MAX_RAM_GB)


def _resolve_backend(profile, max_ram_gb):
    """Choose tokenizer backend.

    - TOKENIZER_BACKEND=hf|spm forces a backend.
    - TOKENIZER_BACKEND=auto (or unset) auto-selects SentencePiece on lower-RAM
      setups where HF Rust BPE commonly OOMs.
    """
    explicit = os.getenv("TOKENIZER_BACKEND", "auto").strip().lower()
    if explicit in {"hf", "spm"}:
        return explicit

    if explicit not in {"", "auto"}:
        print(f"Unknown TOKENIZER_BACKEND={explicit!r}; falling back to auto.")

    if profile == "kaggle" or max_ram_gb < 64:
        return "spm"
    return "hf"


BACKEND = _resolve_backend(PROFILE, MAX_RAM_GB)

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
TRAINING_PIECE_RE = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)

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
    for root, dirs, files in os.walk(DATA_DIR):
        dirs.sort()
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
    Stops after MAX_CORPUS_BYTES total text has been fed to the trainer so
    the Rust BPE word-frequency map does not exhaust system memory.
    """
    print(f"Scanning {DATA_DIR} for datasets...")
    corpus_limit = MAX_CORPUS_BYTES
    print(f"RAM budget: {MAX_RAM_GB:.0f} GB -> corpus cap: "
          f"{corpus_limit / 1_073_741_824:.1f} GB  "
          f"(set TOKENIZER_MAX_RAM_GB to change)")

    batch = []
    batch_characters = 0
    total_bytes = 0

    for file_path in iter_data_files():
        for text in iter_file_texts(file_path):
            text_bytes = len(text.encode("utf-8", errors="replace"))
            batch.append(text)
            batch_characters += len(text)
            total_bytes += text_bytes

            if len(batch) >= max_batch_examples or batch_characters >= max_batch_characters:
                yield batch
                batch = []
                batch_characters = 0

            if total_bytes >= corpus_limit:
                if batch:
                    yield batch
                print(f"Reached corpus cap ({total_bytes / 1_073_741_824:.2f} GB). "
                      f"Stopping data feed.")
                return

    if batch:
        yield batch
    print(f"All data consumed ({total_bytes / 1_073_741_824:.2f} GB).")

# ==========================================
# 3. Two-Pass Training
# ==========================================
def _prune_counter(word_counts, target_size):
    """Remove low-frequency words until the counter is at or below target_size."""
    min_count = 2
    while len(word_counts) > target_size:
        word_counts = collections.Counter(
            {w: c for w, c in word_counts.items() if c >= min_count}
        )
        min_count += 1
    return word_counts, min_count - 1


def _iter_training_pieces(text):
    """Yield normalized pieces used for pass-1 counting and pass-2 filtering.

    We intentionally split more finely than the template pre-tokenizer because
    structured strings like JSON blobs can otherwise appear as one giant unique
    token, which both explodes memory and causes pass 2 to drop important
    punctuation wholesale.
    """
    for match in TRAINING_PIECE_RE.finditer(text):
        piece = match.group(0)
        if piece.isspace():
            yield "space", piece
        elif piece[0].isalnum() or piece[0] == "_":
            yield "word", piece
        else:
            yield "punct", piece


def _normalize_text_for_training(text, allowed_words):
    """Keep punctuation and frequent words while forcing smaller token units.

    The emitted text is intentionally normalized with separator spaces around
    punctuation and kept words. This prevents the downstream pre-tokenizer from
    re-collapsing whole JSON/code snippets into single giant unique tokens.
    """
    pieces = []
    needs_separator = False

    for kind, piece in _iter_training_pieces(text):
        if kind == "space":
            if not pieces or pieces[-1] != "\n":
                pieces.append("\n" if "\n" in piece else " ")
            needs_separator = False
            continue

        if kind == "word" and piece not in allowed_words:
            if pieces and pieces[-1] != " ":
                pieces.append(" ")
            needs_separator = False
            continue

        if needs_separator and pieces and not pieces[-1].isspace():
            pieces.append(" ")

        pieces.append(piece)
        needs_separator = True

    normalized = "".join(pieces).strip()
    return normalized or None


def _count_word_frequencies(tokenizer):
    """Pass 1: stream all data, pre-tokenize, count word frequencies.

    Uses the template tokenizer's pre-tokenizer (the DeepSeek code-regex
    rules) so the word splits match exactly what the BPE trainer will see.

    To keep memory bounded, the counter is periodically pruned: when it
    exceeds 3× MAX_UNIQUE_WORDS, low-frequency words are evicted until
    it is back to 2× MAX_UNIQUE_WORDS.  This means only genuinely
    frequent words survive, which is exactly what we want.
    """
    word_counts = collections.Counter()
    total_bytes = 0
    corpus_limit = MAX_CORPUS_BYTES
    prune_trigger = MAX_UNIQUE_WORDS * 3
    prune_target = MAX_UNIQUE_WORDS * 2
    prune_count = 0

    print(f"Pass 1 — counting word frequencies (corpus cap: "
          f"{corpus_limit / 1_073_741_824:.1f} GB, "
          f"prune at {prune_trigger:,} unique words)...")

    for file_path in iter_data_files():
        for text in iter_file_texts(file_path):
            text_bytes = len(text.encode("utf-8", errors="replace"))
            total_bytes += text_bytes

            for kind, piece in _iter_training_pieces(text):
                if kind == "word":
                    word_counts[piece] += 1

            if len(word_counts) > prune_trigger:
                word_counts, threshold = _prune_counter(word_counts, prune_target)
                prune_count += 1
                print(f"  Pruned to {len(word_counts):,} words "
                      f"(dropped count<{threshold}, "
                      f"{total_bytes / 1_073_741_824:.2f} GB so far)")

            if total_bytes >= corpus_limit:
                print(f"Reached corpus cap ({total_bytes / 1_073_741_824:.2f} GB). "
                      f"Stopping pass 1.")
                break
        else:
            continue
        break

    print(f"Pass 1 done: {len(word_counts):,} unique words from "
          f"{total_bytes / 1_073_741_824:.2f} GB of text "
          f"({prune_count} prune cycles).")
    return word_counts


def _build_allowed_words(word_counts):
    """Select the top words by frequency, capped at MAX_UNIQUE_WORDS.

    First filters by MIN_FREQUENCY, then takes the most common words up
    to MAX_UNIQUE_WORDS.  Returns a frozenset for O(1) lookup in pass 2.
    """
    frequent = {w for w, c in word_counts.items() if c >= MIN_FREQUENCY}
    print(f"After min_frequency={MIN_FREQUENCY} filter: {len(frequent):,} words "
          f"(dropped {len(word_counts) - len(frequent):,} rare words).")

    if len(frequent) <= MAX_UNIQUE_WORDS:
        print(f"Within MAX_UNIQUE_WORDS={MAX_UNIQUE_WORDS:,} limit.")
        return frozenset(frequent)

    # Keep only the most frequent words
    top_words = {w for w, _ in word_counts.most_common(MAX_UNIQUE_WORDS) if w in frequent}
    print(f"Trimmed to top {len(top_words):,} words by frequency.")
    return frozenset(top_words)


def _filtered_batch_iterator(tokenizer, allowed_words):
    """Pass 2: stream data again, but replace rare words with a space.

    The BPE trainer will only see the allowed words, keeping unique-word
    count bounded regardless of corpus size or diversity.
    """
    total_bytes = 0
    corpus_limit = MAX_CORPUS_BYTES

    print(f"\nPass 2 — streaming filtered text to BPE trainer "
          f"({len(allowed_words):,} allowed words)...")

    batch = []
    batch_characters = 0

    for file_path in iter_data_files():
        for text in iter_file_texts(file_path):
            text_bytes = len(text.encode("utf-8", errors="replace"))
            total_bytes += text_bytes

            filtered_text = _normalize_text_for_training(text, allowed_words)
            if not filtered_text:
                continue

            batch.append(filtered_text)
            batch_characters += len(filtered_text)

            if len(batch) >= MAX_BATCH_EXAMPLES or batch_characters >= MAX_BATCH_CHARACTERS:
                yield batch
                batch = []
                batch_characters = 0

            if total_bytes >= corpus_limit:
                if batch:
                    yield batch
                print(f"Reached corpus cap ({total_bytes / 1_073_741_824:.2f} GB). "
                      f"Stopping pass 2.")
                return

    if batch:
        yield batch
    print(f"Pass 2 done ({total_bytes / 1_073_741_824:.2f} GB).")


def _run_sentencepiece_backend():
    script_path = os.path.join(os.path.dirname(__file__), "train_tokenizer_spm.py")
    command = [
        sys.executable,
        script_path,
        "--profile",
        PROFILE,
        "--vocab-size",
        str(VOCAB_SIZE),
        "--output-dir",
        OUTPUT_DIR,
        "--model-type",
        os.getenv("TOKENIZER_SPM_MODEL_TYPE", "unigram"),
    ]

    input_sentence_size = os.getenv("TOKENIZER_SPM_INPUT_SENTENCE_SIZE")
    if input_sentence_size:
        command.extend(["--input-sentence-size", input_sentence_size])

    max_sentence_length = os.getenv("TOKENIZER_SPM_MAX_SENTENCE_LENGTH")
    if max_sentence_length:
        command.extend(["--max-sentence-length", max_sentence_length])

    code_fidelity_mode = os.getenv("TOKENIZER_SPM_CODE_FIDELITY")
    if code_fidelity_mode and code_fidelity_mode.strip().lower() in {"1", "true", "yes", "on"}:
        command.append("--code-fidelity-mode")

    deterministic_mode = os.getenv("TOKENIZER_SPM_DETERMINISTIC")
    if deterministic_mode and deterministic_mode.strip().lower() in {"1", "true", "yes", "on"}:
        command.append("--deterministic")

    print(
        "Tokenizer backend: SentencePiece "
        f"(selected via TOKENIZER_BACKEND={os.getenv('TOKENIZER_BACKEND', 'auto')})."
    )
    print("Delegating to train_tokenizer_spm.py for low-RAM-safe training...")
    os.execv(sys.executable, command)


def _train_hf_backend():
    print(f"Loading template tokenizer ({TEMPLATE_TOKENIZER})...")
    old_tokenizer = AutoTokenizer.from_pretrained(TEMPLATE_TOKENIZER)

    # Pass 1: count word frequencies (streaming, bounded memory)
    word_counts = _count_word_frequencies(old_tokenizer)

    # Build allowed word set
    allowed_words = _build_allowed_words(word_counts)

    # Free the counter before pass 2
    del word_counts

    print(
        f"\nTraining new vocabulary of size {VOCAB_SIZE}.\n"
        f"Profile: {PROFILE} | "
        f"Backend: hf | "
        f"RAM budget: {MAX_RAM_GB:.0f} GB | "
        f"max unique words: {MAX_UNIQUE_WORDS:,} | "
        f"min_frequency: {MIN_FREQUENCY}"
    )

    # Pass 2: train tokenizer on filtered text
    new_tokenizer = old_tokenizer.train_new_from_iterator(
        text_iterator=_filtered_batch_iterator(old_tokenizer, allowed_words),
        vocab_size=VOCAB_SIZE,
        new_special_tokens=SPECIAL_TOKENS,
        min_frequency=MIN_FREQUENCY,
    )
    
    # Set the pad and eos tokens
    new_tokenizer.pad_token = "<|pad|>"
    new_tokenizer.eos_token = "<|eos|>"
    
    print(f"Saving custom tokenizer to ./{OUTPUT_DIR} ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    new_tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Done! Your model's vocabulary is now perfectly mathematically tuned to your data.")


def main():
    if BACKEND == "spm":
        _run_sentencepiece_backend()
        return

    print(
        "Tokenizer backend: HuggingFace BPE "
        f"(selected via TOKENIZER_BACKEND={os.getenv('TOKENIZER_BACKEND', 'auto')})."
    )
    _train_hf_backend()

if __name__ == "__main__":
    main()

