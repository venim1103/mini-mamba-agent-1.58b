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
import copy
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm

import train_tokenizer as base
from context_config import CONTEXT_LENGTH


DOMAINS = ("logic", "code", "tools", "web", "other")


def _resolve_profile(explicit_profile, profile_registry):
    if explicit_profile:
        if explicit_profile in profile_registry:
            return explicit_profile
        available = ", ".join(sorted(profile_registry))
        raise ValueError(f"Unknown profile {explicit_profile!r}. Available profiles: {available}")
    if os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        if "kaggle" in profile_registry:
            return "kaggle"
        return next(iter(sorted(profile_registry)))
    if "standard" in profile_registry:
        return "standard"
    return next(iter(sorted(profile_registry)))


def _parse_domain_map(text, name):
    result = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"{name} must use key=value entries. Got {item!r}.")
        key, raw_value = item.split("=", 1)
        key = key.strip().lower()
        if key not in DOMAINS:
            raise ValueError(f"Unsupported domain {key!r} in {name}. Expected one of: {', '.join(DOMAINS)}")
        result[key] = float(raw_value.strip())
    return result


def _normalize_domain_weights(raw_weights):
    for domain in DOMAINS:
        if domain not in raw_weights:
            raise ValueError(f"Missing domain weight for {domain!r}.")
        if raw_weights[domain] < 0:
            raise ValueError(f"Domain weight for {domain!r} must be non-negative.")
    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        raise ValueError("Domain weights must sum to a positive value.")
    return {domain: raw_weights[domain] / total_weight for domain in DOMAINS}


def _calculate_domain_quotas(total_lines, normalized_weights):
    if total_lines <= 0:
        raise ValueError("quota_total_lines must be > 0.")
    quotas = {domain: int(total_lines * normalized_weights[domain]) for domain in DOMAINS}
    assigned = sum(quotas.values())
    remainder = total_lines - assigned
    if remainder > 0:
        domains_by_fraction = sorted(
            DOMAINS,
            key=lambda domain: (total_lines * normalized_weights[domain]) - quotas[domain],
            reverse=True,
        )
        for idx in range(remainder):
            quotas[domains_by_fraction[idx % len(domains_by_fraction)]] += 1
    return quotas


def _load_profile_registry(profile_file):
    profile_registry = copy.deepcopy(SPM_PROFILE_DEFAULTS)
    if not profile_file:
        return profile_registry

    with open(profile_file, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    custom_profiles = raw.get("profiles", raw)
    if not isinstance(custom_profiles, dict):
        raise ValueError("Profile file must contain a JSON object (or {'profiles': {...}}).")

    for name, settings in custom_profiles.items():
        if not isinstance(settings, dict):
            raise ValueError(f"Profile {name!r} must map to an object.")
        profile_registry[name] = settings

    return profile_registry


def _resolve_profile_settings(args, profile_name, profile_registry):
    defaults = profile_registry[profile_name]
    input_sentence_size = args.input_sentence_size if args.input_sentence_size is not None else defaults["input_sentence_size"]
    max_sentence_length = args.max_sentence_length if args.max_sentence_length is not None else defaults["max_sentence_length"]

    quota_total_lines = (
        args.quota_total_lines
        if args.quota_total_lines is not None
        else defaults.get("quota_total_lines", sum(defaults.get("domain_quota", {}).values()))
    )

    if "domain_weights" in defaults:
        domain_weights = dict(defaults["domain_weights"])
    else:
        legacy_quota = defaults.get("domain_quota", {})
        legacy_total = sum(legacy_quota.get(domain, 0) for domain in DOMAINS)
        if legacy_total <= 0:
            raise ValueError(f"Profile {profile_name!r} does not define usable domain quotas/weights.")
        domain_weights = {domain: legacy_quota.get(domain, 0) / legacy_total for domain in DOMAINS}

    if args.quota_weights:
        domain_weights.update(_parse_domain_map(args.quota_weights, "--quota-weights"))
    normalized_weights = _normalize_domain_weights(domain_weights)
    quotas = _calculate_domain_quotas(quota_total_lines, normalized_weights)

    if args.quota_overrides:
        for domain, value in _parse_domain_map(args.quota_overrides, "--quota-overrides").items():
            quotas[domain] = max(0, int(value))

    return {
        "input_sentence_size": input_sentence_size,
        "max_sentence_length": max_sentence_length,
        "quota_total_lines": quota_total_lines,
        "domain_weights": normalized_weights,
        "domain_quotas": quotas,
    }


def _auto_tune_input_sentence_size(profile, input_sentence_size):
    """Scale SPM sampling upward on higher-RAM machines when using kaggle profile.

    This keeps the kaggle defaults safe for low-RAM environments while making
    better use of local machines that intentionally run with kaggle settings.
    """
    if profile != "kaggle":
        return input_sentence_size

    raw_ram = os.getenv("TOKENIZER_MAX_RAM_GB")
    if not raw_ram:
        return input_sentence_size

    try:
        ram_gb = float(raw_ram)
    except ValueError:
        return input_sentence_size

    base = SPM_PROFILE_DEFAULTS["kaggle"]["input_sentence_size"]
    if ram_gb <= 13:
        return input_sentence_size

    # Linear scale from 13 GB -> 30 GB, capped for OOM safety.
    # With current settings, 30 GB machines were underutilized (~11 GB peak),
    # so this default cap is intentionally more aggressive.
    target_cap = 2_800_000
    raw_cap = os.getenv("TOKENIZER_SPM_AUTO_MAX_INPUT_SENTENCE_SIZE")
    if raw_cap:
        try:
            target_cap = max(base, int(raw_cap))
        except ValueError:
            pass

    scaled = int(base + (ram_gb - 13.0) * (target_cap - base) / (30.0 - 13.0))
    tuned = max(base, min(target_cap, scaled))

    if tuned != input_sentence_size:
        print(
            "Auto-tuning input_sentence_size for kaggle profile based on "
            f"TOKENIZER_MAX_RAM_GB={ram_gb:g}: {input_sentence_size:,} -> {tuned:,}"
        )
    return tuned


SPM_PROFILE_DEFAULTS = {
    "standard": {
        "input_sentence_size": 750_000,
        "max_sentence_length": 2048,
        "quota_total_lines": 2_100_000,
        "domain_weights": {
            "logic": 0.1667,
            "code": 0.3095,
            "tools": 0.1190,
            "web": 0.3571,
            "other": 0.0476,
        },
    },
    "kaggle": {
        "input_sentence_size": 500_000,
        "max_sentence_length": 1600,
        "quota_total_lines": 1_000_000,
        "domain_weights": {
            "logic": 0.22,
            "code": 0.30,
            "tools": 0.16,
            "web": 0.28,
            "other": 0.04,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a low-RAM SentencePiece tokenizer.")
    parser.add_argument("--profile", help="Training profile name (built-in or from --profile-file).")
    parser.add_argument(
        "--profile-file",
        help="Optional JSON file with custom profiles. Accepts either {name: {...}} or {'profiles': {name: {...}}}.",
    )
    parser.add_argument("--vocab-size", type=int, default=64_000)
    parser.add_argument("--model-type", choices=["bpe", "unigram"], default="unigram")
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument(
        "--byte-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable SentencePiece byte fallback. Recommended for code so unseen bytes "
            "are represented as byte pieces instead of <|unk|>."
        ),
    )
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
    parser.add_argument(
        "--quota-total-lines",
        type=int,
        help="Total sampled corpus lines before SPM training. Per-domain quotas are derived from normalized weights.",
    )
    parser.add_argument(
        "--quota-weights",
        help=(
            "Comma-separated domain weights (logic=0.2,code=0.4,tools=0.1,web=0.25,other=0.05). "
            "Values are normalized automatically."
        ),
    )
    parser.add_argument(
        "--quota-overrides",
        help="Comma-separated absolute per-domain quotas (e.g. code=400000,web=500000).",
    )
    parser.add_argument(
        "--code-fidelity-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Preserve code whitespace structure better by avoiding newline flattening in corpus "
            "construction and disabling aggressive whitespace normalization in SPM."
        ),
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


def _build_temp_corpus(profile, quotas, max_sentence_length, code_fidelity_mode=False):
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

                if code_fidelity_mode:
                    # Keep line structure/indentation signal for code-heavy text.
                    text = text.replace("\r\n", "\n").replace("\r", "\n")
                    if not text.strip():
                        continue
                else:
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


def _resolve_model_max_length():
    raw = os.getenv("TOKENIZER_MODEL_MAX_LENGTH")
    if raw:
        try:
            parsed = int(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return CONTEXT_LENGTH


def _export_hf_tokenizer(spm_model_path, output_dir, model_max_length=CONTEXT_LENGTH):
    processor = spm.SentencePieceProcessor(model_file=spm_model_path)

    try:
        import sys

        import sentencepiece.sentencepiece_model_pb2 as sp_pb2
        from tokenizers import Tokenizer
        from tokenizers import decoders
        from tokenizers.decoders import Metaspace as MetaspaceDecoder
        from tokenizers.implementations import SentencePieceUnigramTokenizer
        from tokenizers.models import Unigram
        from tokenizers.normalizers import NFKC
        from tokenizers.pre_tokenizers import Metaspace
        from transformers import PreTrainedTokenizerFast
    except Exception as exc:  # pragma: no cover
        print(f"Skipping HuggingFace export (fast tokenizer dependencies unavailable): {exc}")
        return

    try:
        # Preferred path: import directly from the .model to preserve SentencePiece behavior.
        sys.modules.setdefault("sentencepiece_model_pb2", sp_pb2)
        impl = SentencePieceUnigramTokenizer.from_spm(spm_model_path)
        impl._tokenizer.decoder = decoders.Sequence(
            [
                decoders.ByteFallback(),
                decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True),
            ]
        )

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=impl._tokenizer,
            bos_token="<s>",
            eos_token="<|eos|>",
            unk_token="<|unk|>",
            pad_token="<|pad|>",
            model_max_length=model_max_length,
        )
        tokenizer.save_pretrained(output_dir)
        print(f"Saved HuggingFace tokenizer files to ./{output_dir}")
        return
    except Exception as spm_exc:
        print(f"Direct SentencePiece export path unavailable, falling back to manual conversion: {spm_exc}")

    try:
        byte_fallback = False
        try:
            from sentencepiece import sentencepiece_model_pb2 as sp_pb2

            model_proto = sp_pb2.ModelProto()
            with open(spm_model_path, "rb") as fh:
                model_proto.ParseFromString(fh.read())
            byte_fallback = bool(model_proto.trainer_spec.byte_fallback)
        except Exception:
            # Best-effort: if protobuf parsing is unavailable, keep legacy default.
            pass

        vocab = [(processor.id_to_piece(i), processor.get_score(i)) for i in range(processor.vocab_size())]
        backend = Tokenizer(Unigram(vocab, unk_id=processor.unk_id(), byte_fallback=byte_fallback))
        backend.normalizer = NFKC()
        backend.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="first")
        backend.decoder = decoders.Sequence(
            [
                decoders.ByteFallback(),
                MetaspaceDecoder(replacement="▁", prepend_scheme="first"),
            ]
        )

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token="<s>",
            eos_token="<|eos|>",
            unk_token="<|unk|>",
            pad_token="<|pad|>",
            model_max_length=model_max_length,
        )
        tokenizer.save_pretrained(output_dir)
        print(f"Saved HuggingFace tokenizer files to ./{output_dir}")
    except Exception as exc:  # pragma: no cover
        print(f"Error during HuggingFace tokenizer export: {exc}")
        raise


def main():
    args = parse_args()
    profile_registry = _load_profile_registry(args.profile_file)
    profile = _resolve_profile(args.profile, profile_registry)
    settings = _resolve_profile_settings(args, profile, profile_registry)
    input_sentence_size = settings["input_sentence_size"]
    max_sentence_length = settings["max_sentence_length"]

    if args.input_sentence_size is None:
        input_sentence_size = _auto_tune_input_sentence_size(profile, input_sentence_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path, total_lines = _build_temp_corpus(
        profile=profile,
        quotas=settings["domain_quotas"],
        max_sentence_length=max_sentence_length,
        code_fidelity_mode=args.code_fidelity_mode,
    )

    try:
        if total_lines == 0:
            raise RuntimeError("No training text found for SentencePiece corpus.")

        model_prefix = output_dir / "spm"

        print(
            f"Training SentencePiece ({args.model_type}) with vocab={args.vocab_size:,}, "
            f"input_sentence_size={input_sentence_size:,}, max_sentence_length={max_sentence_length}, "
            f"byte_fallback={args.byte_fallback}, code_fidelity_mode={args.code_fidelity_mode}."
        )
        print(
            "Quota design: total_lines="
            f"{settings['quota_total_lines']:,}, normalized_weights={settings['domain_weights']}, "
            f"derived_quotas={settings['domain_quotas']}"
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
            byte_fallback=args.byte_fallback,
            split_by_whitespace=not args.code_fidelity_mode,
            remove_extra_whitespaces=not args.code_fidelity_mode,
            num_threads=max(os.cpu_count() or 1, 1),
        )
    finally:
        os.remove(corpus_path)

    spm_model_path = str(model_prefix) + ".model"
    _export_hf_tokenizer(
        spm_model_path,
        str(output_dir),
        model_max_length=_resolve_model_max_length(),
    )

    print(f"Done. SentencePiece artifacts in ./{output_dir}")


if __name__ == "__main__":
    main()
