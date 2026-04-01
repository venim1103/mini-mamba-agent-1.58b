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

import argparse
import json
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer


DEFAULT_BASE_TOKENIZER = "deepseek-ai/deepseek-coder-1.3b-base"
DEFAULT_CUSTOM_TOKENIZER = "custom_agentic_tokenizer"

DEFAULT_SAMPLES = [
    {
        "name": "python_code",
        "text": (
            "def binary_search(arr, target):\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        if arr[mid] < target:\n"
            "            left = mid + 1\n"
            "        else:\n"
            "            right = mid - 1\n"
            "    return -1"
        ),
    },
    {
        "name": "math_reasoning",
        "text": (
            "Solve the quadratic equation x^2 + 5x + 6 = 0. "
            "We factor it as (x + 2)(x + 3) = 0, so the solutions are x = -2 and x = -3."
        ),
    },
    {
        "name": "tool_calling",
        "text": (
            "<|im_start|>assistant\n"
            "<think>I should call the weather tool with the provided city and unit.</think>\n"
            '{"name":"get_weather","arguments":{"city":"Berlin","unit":"celsius"}}'
            "<|im_end|>"
        ),
    },
    {
        "name": "web_text",
        "text": (
            "A transformer language model predicts the next token in a sequence. "
            "Training quality depends on data curation, tokenizer coverage, and "
            "optimization stability across domains like code, math, and natural language."
        ),
    },
    {
        "name": "mixed_agentic",
        "text": (
            "User: write a Python function that parses JSON and retries on timeout. "
            "Assistant: <think>I need robust error handling and exponential backoff.</think> "
            "Sure, here is an implementation using try/except and time.sleep."
        ),
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare token counts between two tokenizers on representative samples."
    )
    parser.add_argument(
        "--base-tokenizer",
        default=DEFAULT_BASE_TOKENIZER,
        help="Base tokenizer model ID or local path.",
    )
    parser.add_argument(
        "--custom-tokenizer",
        default=DEFAULT_CUSTOM_TOKENIZER,
        help="Custom tokenizer model ID or local path.",
    )
    parser.add_argument(
        "--samples-file",
        help=(
            "Optional path to a JSON file containing a list of objects with "
            "'name' and 'text' fields."
        ),
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Print the first 20 token IDs for each tokenizer per sample.",
    )
    return parser.parse_args()


def load_samples(samples_file):
    if not samples_file:
        return DEFAULT_SAMPLES

    path = Path(samples_file)
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("Samples file must contain a JSON list.")

    normalized = []
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {index} is not an object.")
        name = sample.get("name")
        text = sample.get("text")
        if not isinstance(name, str) or not isinstance(text, str):
            raise ValueError(f"Sample {index} must have string 'name' and 'text' fields.")
        normalized.append({"name": name, "text": text})
    return normalized


def format_ratio(base_count, custom_count):
    if base_count == 0:
        return "n/a"

    ratio = custom_count / base_count
    if ratio <= 1:
        return f"{ratio:.3f} ({(1 - ratio) * 100:.1f}% fewer tokens)"
    return f"{ratio:.3f} ({(ratio - 1) * 100:.1f}% more tokens)"


def compare_tokenizers(base_tokenizer, custom_tokenizer, samples, show_ids=False):
    ratios = []

    print("Comparison: base tokenizer vs custom tokenizer")
    print(f"Base:   {base_tokenizer.name_or_path}")
    print(f"Custom: {custom_tokenizer.name_or_path}")
    print()

    for sample in samples:
        name = sample["name"]
        text = sample["text"]
        base_ids = base_tokenizer.encode(text, add_special_tokens=False)
        custom_ids = custom_tokenizer.encode(text, add_special_tokens=False)
        roundtrip = custom_tokenizer.decode(custom_ids, skip_special_tokens=False) == text

        if base_ids:
            ratios.append(len(custom_ids) / len(base_ids))

        print(f"{name}:")
        print(f"  chars:   {len(text)}")
        print(f"  base:    {len(base_ids):4d} tokens")
        print(f"  custom:  {len(custom_ids):4d} tokens")
        print(f"  ratio:   {format_ratio(len(base_ids), len(custom_ids))}")
        print(f"  roundtrip exact: {roundtrip}")
        if not roundtrip:
            decoded = custom_tokenizer.decode(custom_ids, skip_special_tokens=False)
            print(f"  decoded preview: {decoded[:160]!r}")
        if show_ids:
            print(f"  base ids:   {base_ids[:20]}")
            print(f"  custom ids: {custom_ids[:20]}")
        print()

    if ratios:
        print(f"Average custom/base ratio: {mean(ratios):.3f}")


def main():
    args = parse_args()
    samples = load_samples(args.samples_file)
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    custom_tokenizer = AutoTokenizer.from_pretrained(args.custom_tokenizer)
    compare_tokenizers(
        base_tokenizer=base_tokenizer,
        custom_tokenizer=custom_tokenizer,
        samples=samples,
        show_ids=args.show_ids,
    )


if __name__ == "__main__":
    main()