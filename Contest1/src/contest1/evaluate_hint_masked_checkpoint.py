#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import hint_masked_lib as hml
from .data_policy import reject_forbidden_evaluation_path
from .paths import ARTIFACTS_DIR, DATA_DIR, TRAIN_DIR
from .train_hint_masked import atomic_text_write, validate_dev_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a hint-masked checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ARTIFACTS_DIR / "gpt2-keyboard-script/best_dev_step_126000.pt",
    )
    parser.add_argument(
        "--train-path", type=Path, default=TRAIN_DIR / "train.src.tok"
    )
    parser.add_argument("--dev-path", type=Path, default=DATA_DIR / "devv_eval.csv")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACTS_DIR / "gpt2-keyboard-script/analysis_step_126000",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--max-train-lines", type=int, default=None)
    parser.add_argument("--rerank-words-per-root", type=int, default=1)
    parser.add_argument("--rerank-root-beam", type=int, default=None)
    parser.add_argument("--rerank-batch-size", type=int, default=64)
    parser.add_argument(
        "--rerank-mode", choices=("within_root", "cross_root"), default="cross_root"
    )
    parser.add_argument("--rerank-suffix-weight", type=float, default=1.0)
    parser.add_argument("--rerank-suffix-length-penalty", type=float, default=0.0)
    parser.add_argument("--rerank-frequency-weight", type=float, default=0.0)
    parser.add_argument("--rerank-boundary-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_forbidden_evaluation_path(args.dev_path)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.train_path.is_file():
        raise FileNotFoundError(args.train_path)
    if not args.dev_path.is_file():
        raise FileNotFoundError(args.dev_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.config.pad_token_id = tokenizer.pad_token_id
    torch.nn.Module.to(model, device)

    print(f"Building candidate lexicon from {args.train_path} ...", flush=True)
    lexicon = hml.build_training_lexicon(
        tokenizer,
        args.train_path,
        max_lines=args.max_train_lines,
    )
    dev = pd.read_csv(args.dev_path, keep_default_na=False)
    validate_dev_frame(dev)
    boundary_token_ids = (
        hml.build_boundary_token_ids(tokenizer)
        if args.rerank_boundary_weight
        else None
    )
    print(f"Evaluating {len(dev):,} full-dev examples ...", flush=True)
    metrics, details, per_hint = hml.evaluate_model(
        model,
        tokenizer,
        dev,
        candidate_ids_by_hint=lexicon.candidate_ids_by_hint,
        hint_token_word=lexicon.hint_token_word,
        fallback_by_hint=lexicon.fallback_by_hint,
        predictable_words=lexicon.predictable_words,
        hint_root_words=lexicon.hint_root_words,
        word_bpe=lexicon.word_bpe,
        word_counts=lexicon.word_counts,
        rerank_words_per_root=args.rerank_words_per_root,
        rerank_root_beam=args.rerank_root_beam,
        rerank_batch_size=args.rerank_batch_size,
        rerank_mode=args.rerank_mode,
        rerank_suffix_weight=args.rerank_suffix_weight,
        rerank_suffix_length_penalty=args.rerank_suffix_length_penalty,
        rerank_frequency_weight=args.rerank_frequency_weight,
        rerank_boundary_weight=args.rerank_boundary_weight,
        boundary_token_ids=boundary_token_ids,
        label=f"checkpoint-step-{checkpoint['step']}",
        top_k=args.top_k,
        eval_batch_size=args.eval_batch_size,
        max_context_tokens=args.max_context_tokens,
    )
    metrics["checkpoint_step"] = int(checkpoint["step"])
    metrics["selection_metrics"] = checkpoint.get("metrics")
    metrics["rerank_config"] = {
        "mode": args.rerank_mode,
        "words_per_root": args.rerank_words_per_root,
        "root_beam": args.rerank_root_beam,
        "suffix_weight": args.rerank_suffix_weight,
        "suffix_length_penalty": args.rerank_suffix_length_penalty,
        "frequency_weight": args.rerank_frequency_weight,
        "boundary_weight": args.rerank_boundary_weight,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text_write(
        json.dumps(metrics, indent=2) + "\n",
        args.output_dir / "full_dev_metrics.json",
    )
    details.to_csv(args.output_dir / "full_dev_predictions.csv", index=False)
    per_hint.to_csv(args.output_dir / "full_dev_per_hint.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Analysis saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
