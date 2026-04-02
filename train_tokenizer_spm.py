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
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm

import train_tokenizer as base


def _resolve_profile(explicit_profile):
    if explicit_profile in {"kaggle", "standard"}:
        return explicit_profile
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "standard"


SPM_PROFILE_DEFAULTS = {
    "standard": {
        "input_sentence_size": 750_000,
        "max_sentence_length": 2048,
        "domain_quota": {
            "logic": 350_000,
            "code": 650_000,
            "tools": 250_000,
            "web": 750_000,
            "other": 100_000,
        },
    },
    "kaggle": {
        "input_sentence_size": 500_000,
        "max_sentence_length": 1600,
        "domain_quota": {
            "logic": 220_000,
            "code": 300_000,
            "tools": 160_000,
            "web": 280_000,
            "other": 40_000,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a low-RAM SentencePiece tokenizer.")
    parser.add_argument("--profile", choices=["kaggle", "standard"], help="Training profile.")
    parser.add_argument("--vocab-size", type=int, default=64_000)
    parser.add_argument("--model-type", choices=["bpe", "unigram"], default="unigram")
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--output-dir", default="custom_agentic_tokenizer_spm")
    parser.add_argument(
        "--input-sentence-size",
        type=int,
        help="Max sentence count for SPM trainer sampling. Uses profile default if omitted.",
    )
    parser.add_argument(
        "--max-sentence-length",
        type=int,
        help="Max sentence length passed to SPM trainer. Uses profile default if omitted.",
    )
    return parser.parse_args()


def _infer_domain(file_path):
    normalized = file_path.replace("\\", "/").lower()
    if "/logic/" in normalized:
        return "logic"
    if "/code/" in normalized:
        return "code"
    if "/tools/" in normalized:
        return "tools"
    if "/web/" in normalized:
        return "web"
    return "other"


def _build_temp_corpus(profile, max_sentence_length):
    quotas = SPM_PROFILE_DEFAULTS[profile]["domain_quota"]
    domain_counts = defaultdict(int)
    total = 0

    temp = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8")
    temp_path = temp.name

    print(f"Building temporary corpus file: {temp_path}")
    print(f"Profile: {profile} | domain quotas: {quotas}")

    try:
        for file_path in base.iter_data_files():
            domain = _infer_domain(file_path)
            if domain_counts[domain] >= quotas[domain]:
                continue

            for text in base.iter_file_texts(file_path):
                if domain_counts[domain] >= quotas[domain]:
                    break

                text = text.replace("\n", " ").strip()
                if not text:
                    continue

                if max_sentence_length > 0:
                    text = text[:max_sentence_length]

                temp.write(text)
                temp.write("\n")

                domain_counts[domain] += 1
                total += 1

        temp.flush()
        print(f"Temporary corpus built with {total:,} lines.")
        for domain in sorted(quotas):
            print(f"  {domain:>5}: {domain_counts[domain]:,} / {quotas[domain]:,}")
        return temp_path, total
    finally:
        temp.close()


def _export_hf_tokenizer(spm_model_path, output_dir):
    processor = spm.SentencePieceProcessor(model_file=spm_model_path)

    try:
        from tokenizers import Tokenizer
        from tokenizers.decoders import Metaspace as MetaspaceDecoder
        from tokenizers.models import Unigram
        from tokenizers.normalizers import NFKC
        from tokenizers.pre_tokenizers import Metaspace
        from transformers import PreTrainedTokenizerFast
    except Exception as exc:  # pragma: no cover
        print(f"Skipping HuggingFace export (fast tokenizer dependencies unavailable): {exc}")
        return

    try:
        vocab = [(processor.id_to_piece(i), processor.get_score(i)) for i in range(processor.vocab_size())]
        backend = Tokenizer(Unigram(vocab, unk_id=processor.unk_id()))
        backend.normalizer = NFKC()
        backend.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="first")
        backend.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="first")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token="<s>",
            eos_token="<|eos|>",
            unk_token="<|unk|>",
            pad_token="<|pad|>",
            additional_special_tokens=["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
        )
        tokenizer.save_pretrained(output_dir)
        print(f"Saved HuggingFace tokenizer files to ./{output_dir}")
    except Exception as exc:  # pragma: no cover
        print(f"Error during HuggingFace tokenizer export: {exc}")
        raise


def main():
    args = parse_args()
    profile = _resolve_profile(args.profile)
    defaults = SPM_PROFILE_DEFAULTS[profile]

    input_sentence_size = args.input_sentence_size if args.input_sentence_size is not None else defaults["input_sentence_size"]
    max_sentence_length = args.max_sentence_length if args.max_sentence_length is not None else defaults["max_sentence_length"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path, total_lines = _build_temp_corpus(
        profile=profile,
        max_sentence_length=max_sentence_length,
    )

    try:
        if total_lines == 0:
            raise RuntimeError("No training text found for SentencePiece corpus.")

        model_prefix = output_dir / "spm"

        print(
            f"Training SentencePiece ({args.model_type}) with vocab={args.vocab_size:,}, "
            f"input_sentence_size={input_sentence_size:,}, max_sentence_length={max_sentence_length}."
        )

        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=str(model_prefix),
            model_type=args.model_type,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=True,
            max_sentence_length=max_sentence_length,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<|pad|>",
            unk_piece="<|unk|>",
            bos_piece="<s>",
            eos_piece="<|eos|>",
            user_defined_symbols=["<|im_start|>", "<|im_end|>", "<think>", "</think>"],
            num_threads=max(os.cpu_count() or 1, 1),
        )
    finally:
        os.remove(corpus_path)

    spm_model_path = str(model_prefix) + ".model"
    _export_hf_tokenizer(spm_model_path, str(output_dir))

    print(f"Done. SentencePiece artifacts in ./{output_dir}")


if __name__ == "__main__":
    main()
