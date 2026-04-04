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

"""Tokenizer comparison and manual regression-gate utility.

Quick start examples:

1) Basic comparison against local custom tokenizer:
   /opt/conda/envs/ai/bin/python compare_tokenizers.py \
       --custom-tokenizer custom_agentic_tokenizer

2) Save a JSON report for later inspection:
   /opt/conda/envs/ai/bin/python compare_tokenizers.py \
       --custom-tokenizer custom_agentic_tokenizer \
       --report-json /tmp/tokenizer_report.json

3) Enable regression gates (exit code 1 on failure):
   /opt/conda/envs/ai/bin/python compare_tokenizers.py \
       --custom-tokenizer custom_agentic_tokenizer \
       --max-average-ratio 1.10 \
       --max-sample-ratio 1.40 \
       --require-roundtrip

4) Exclude one or more samples during manual checks:
   /opt/conda/envs/ai/bin/python compare_tokenizers.py \
       --custom-tokenizer custom_agentic_tokenizer \
       --exclude-sample python_code

   /opt/conda/envs/ai/bin/python compare_tokenizers.py \
       --custom-tokenizer custom_agentic_tokenizer \
       --exclude-sample python_code,tool_calling

5) Use a custom sample set:
   /opt/conda/envs/ai/bin/python compare_tokenizers.py \
       --custom-tokenizer custom_agentic_tokenizer \
       --samples-file tests/tokenizer_samples.json

Notes:
- Ratios are custom/base token counts per sample.
- Lower ratio is better (fewer custom tokens than base).
- If all base token counts are zero, average ratio is unavailable.
"""

import argparse
import json
import sys
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
    parser.add_argument(
        "--max-average-ratio",
        type=float,
        help=(
            "Optional regression gate. Fail with exit code 1 if average "
            "custom/base ratio exceeds this value."
        ),
    )
    parser.add_argument(
        "--max-sample-ratio",
        type=float,
        help=(
            "Optional regression gate. Fail with exit code 1 if any sample "
            "custom/base ratio exceeds this value."
        ),
    )
    parser.add_argument(
        "--require-roundtrip",
        action="store_true",
        help="Optional regression gate. Fail if any sample does not roundtrip exactly.",
    )
    parser.add_argument(
        "--report-json",
        help="Optional output path for a JSON report with per-sample metrics.",
    )
    parser.add_argument(
        "--exclude-sample",
        action="append",
        default=[],
        help=(
            "Exclude sample(s) by name. Can be repeated and/or passed as comma-separated "
            "values, e.g. --exclude-sample python_code --exclude-sample web_text,tool_calling."
        ),
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


def filter_samples(samples, excluded_names):
    if not excluded_names:
        return samples

    # Support both repeated --exclude-sample flags and comma-separated values.
    excluded = {
        name.strip()
        for raw in excluded_names
        for name in raw.split(",")
        if name.strip()
    }
    return [sample for sample in samples if sample["name"] not in excluded]


def format_ratio(base_count, custom_count):
    if base_count == 0:
        return "n/a"

    ratio = custom_count / base_count
    if ratio <= 1:
        return f"{ratio:.3f} ({(1 - ratio) * 100:.1f}% fewer tokens)"
    return f"{ratio:.3f} ({(ratio - 1) * 100:.1f}% more tokens)"


def compare_tokenizers(base_tokenizer, custom_tokenizer, samples, show_ids=False):
    ratios = []
    results = []

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

        ratio = None
        if base_ids:
            ratio = len(custom_ids) / len(base_ids)

        decoded = custom_tokenizer.decode(custom_ids, skip_special_tokens=False)
        results.append(
            {
                "name": name,
                "chars": len(text),
                "base_tokens": len(base_ids),
                "custom_tokens": len(custom_ids),
                "ratio": ratio,
                "roundtrip_exact": roundtrip,
                "decoded_preview": decoded[:160],
            }
        )

        print(f"{name}:")
        print(f"  chars:   {len(text)}")
        print(f"  base:    {len(base_ids):4d} tokens")
        print(f"  custom:  {len(custom_ids):4d} tokens")
        print(f"  ratio:   {format_ratio(len(base_ids), len(custom_ids))}")
        print(f"  roundtrip exact: {roundtrip}")
        if not roundtrip:
            print(f"  decoded preview: {decoded[:160]!r}")
        if show_ids:
            print(f"  base ids:   {base_ids[:20]}")
            print(f"  custom ids: {custom_ids[:20]}")
        print()

    average_ratio = None
    if ratios:
        average_ratio = mean(ratios)
        print(f"Average custom/base ratio: {average_ratio:.3f}")

    return {
        "base_tokenizer": base_tokenizer.name_or_path,
        "custom_tokenizer": custom_tokenizer.name_or_path,
        "average_ratio": average_ratio,
        "samples": results,
    }


def evaluate_regressions(report, max_average_ratio=None, max_sample_ratio=None, require_roundtrip=False):
    failures = []

    if max_average_ratio is not None:
        avg = report.get("average_ratio")
        if avg is None:
            failures.append("average ratio unavailable (all base token counts were zero)")
        elif avg > max_average_ratio:
            failures.append(
                f"average ratio {avg:.3f} exceeded max-average-ratio {max_average_ratio:.3f}"
            )

    if max_sample_ratio is not None:
        for sample in report.get("samples", []):
            ratio = sample.get("ratio")
            if ratio is not None and ratio > max_sample_ratio:
                failures.append(
                    f"sample '{sample['name']}' ratio {ratio:.3f} exceeded "
                    f"max-sample-ratio {max_sample_ratio:.3f}"
                )

    if require_roundtrip:
        for sample in report.get("samples", []):
            if not sample.get("roundtrip_exact", False):
                failures.append(f"sample '{sample['name']}' failed exact roundtrip")

    return failures


def main():
    args = parse_args()
    samples = load_samples(args.samples_file)
    samples = filter_samples(samples, args.exclude_sample)
    if not samples:
        raise ValueError("No samples left after applying --exclude-sample filters.")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    custom_tokenizer = AutoTokenizer.from_pretrained(args.custom_tokenizer)
    report = compare_tokenizers(
        base_tokenizer=base_tokenizer,
        custom_tokenizer=custom_tokenizer,
        samples=samples,
        show_ids=args.show_ids,
    )

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {args.report_json}")

    failures = evaluate_regressions(
        report,
        max_average_ratio=args.max_average_ratio,
        max_sample_ratio=args.max_sample_ratio,
        require_roundtrip=args.require_roundtrip,
    )
    if failures:
        print("\nRegression check failed:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)


if __name__ == "__main__":
    main()