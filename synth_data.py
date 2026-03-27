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

"""Synthetic data generation pipeline (Nemotron-H §2.3, Nanbeige4-3B §2.2).

Generates augmented training samples from raw source documents using
the trained model itself as a rewriter. Five prompt strategies are
supported (following Nemotron-H):

1. diverse_qa   — Generate diverse QA pairs from a source passage.
2. distill      — Rewrite into concise, information-dense form.
3. extract      — Pull structured knowledge from unstructured text.
4. knowledge    — Produce bulleted knowledge lists.
5. rephrase     — Rewrite low-quality text in clean, encyclopedic style.

Usage:
    python synth_data.py --strategy diverse_qa --input local_data/train/web --output local_data/synth/web_qa
    python synth_data.py --strategy distill --input local_data/train/code --output local_data/synth/code_distill
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from model import BitMambaLLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CKPT = "checkpoints/bitmamba_parent/step_1000000.pt"
MODEL_CONFIG = dict(vocab_size=64000, dim=1024, n_layers=40, d_state=128, expand=2)

# ---------------------------------------------------------------------------
# Prompt templates for each strategy (Nemotron-H §2.3)
# ---------------------------------------------------------------------------

STRATEGY_PROMPTS = {
    "diverse_qa": (
        "<|im_start|>system\nYou are a meticulous educator. "
        "Given a passage, create 3 diverse question-answer pairs that test different "
        "levels of understanding (factual recall, inference, and application).<|im_end|>\n"
        "<|im_start|>user\nPassage:\n{text}\n\n"
        "Generate 3 QA pairs in the format:\nQ1: ...\nA1: ...\nQ2: ...\nA2: ...\nQ3: ...\nA3: ...<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "distill": (
        "<|im_start|>system\nYou are a technical writer who distills long documents into "
        "concise, information-dense summaries. Preserve all key facts and relationships.<|im_end|>\n"
        "<|im_start|>user\nRewrite the following text in a concise, clear form. "
        "Keep all essential information but remove redundancy:\n\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "extract": (
        "<|im_start|>system\nYou are a knowledge extraction engine. "
        "Extract structured facts from unstructured text.<|im_end|>\n"
        "<|im_start|>user\nExtract all key entities, relationships, and facts from this text "
        "as a structured list:\n\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "knowledge": (
        "<|im_start|>system\nYou are an encyclopedic knowledge organizer.<|im_end|>\n"
        "<|im_start|>user\nConvert the following text into a bulleted knowledge list. "
        "Each bullet should be a self-contained fact:\n\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "rephrase": (
        "<|im_start|>system\nYou are a Wikipedia editor. Rewrite low-quality or informal text "
        "into clean, encyclopedic prose. Fix grammar, improve clarity, and add structure.<|im_end|>\n"
        "<|im_start|>user\nRewrite the following text in a clear, encyclopedic style:\n\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}


# ---------------------------------------------------------------------------
# Generation — uses model.generate() for O(n) cached inference
# ---------------------------------------------------------------------------


def truncate_source(text, tokenizer, max_source_tokens=1024):
    """Truncate source text to fit within the prompt budget."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_source_tokens:
        tokens = tokens[:max_source_tokens]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def iter_source_texts(input_path):
    """Yield raw text strings from a directory of parquet/json/jsonl files."""
    files = []
    for root, _, filenames in os.walk(input_path):
        for f in filenames:
            if f.endswith((".parquet", ".json", ".jsonl")):
                files.append(os.path.join(root, f))
    if not files:
        raise FileNotFoundError(f"No data files found under {input_path}")

    parquet_files = [f for f in files if f.endswith(".parquet")]
    json_files = [f for f in files if f.endswith((".json", ".jsonl"))]

    if parquet_files:
        ds = load_dataset("parquet", data_files=parquet_files, split="train", streaming=True)
    else:
        ds = load_dataset("json", data_files=json_files, split="train", streaming=True)

    for row in ds:
        # Try common text column names
        for col in ["text", "content", "passage", "question", "problem", "instruction"]:
            if col in row and isinstance(row[col], str) and len(row[col]) > 50:
                yield row[col]
                break


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    print(f"Strategy: {args.strategy}")
    print(f"Input:    {args.input}")
    print(f"Output:   {args.output}")
    print(f"Samples:  {args.num_samples}")

    tokenizer = AutoTokenizer.from_pretrained("custom_agentic_tokenizer")
    eos_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    print("Loading model...")
    model = BitMambaLLM(**MODEL_CONFIG).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prompt_template = STRATEGY_PROMPTS[args.strategy]
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"{args.strategy}.jsonl")

    count = 0
    with open(out_path, "w") as fout:
        for text in iter_source_texts(args.input):
            if count >= args.num_samples:
                break

            text = truncate_source(text, tokenizer, max_source_tokens=args.max_source_tokens)
            prompt = prompt_template.format(text=text)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

            if input_ids.shape[1] > 2048:
                continue  # skip overly long prompts

            output_ids = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, do_sample=(args.temperature > 0),
                eos_token_id=eos_id,
            )
            gen_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

            record = {
                "strategy": args.strategy,
                "source": text[:500],  # truncated source for provenance
                "generated": gen_text,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if count % 100 == 0:
                print(f"  Generated {count}/{args.num_samples}")

    print(f"Done. Wrote {count} samples to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic data generation pipeline")
    parser.add_argument("--strategy", required=True, choices=list(STRATEGY_PROMPTS.keys()),
                        help="Generation strategy to use")
    parser.add_argument("--input", required=True, help="Path to source data directory")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Model checkpoint path")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens per generation")
    parser.add_argument("--max_source_tokens", type=int, default=1024, help="Max source tokens in prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0=greedy)")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
