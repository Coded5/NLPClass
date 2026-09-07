#!/usr/bin/env python3
"""Exact, cache-first analysis of hint-restricted GPT-2 checkpoints.

The expensive model pass and the deterministic analysis pass are deliberately
separate.  ``cache`` stores every allowed root log probability in one flat NPY
array; ``analyze`` can then compare checkpoints, sweep ensembles, build a
leakage-safe top-five reranker, and regenerate all reports without a GPU.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import hint_masked_lib as hml
from .data_policy import reject_forbidden_evaluation_path
from .paths import DATA_DIR, TRAIN_DIR
from .train_hint_masked import atomic_text_write, validate_dev_frame


CACHE_VERSION = 2
DEFAULT_ALPHAS = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"
RERANK_FEATURES = [
    "model_log_probability",
    "model_rank",
    "top1_margin",
    "log_unigram_count",
    "letter_unigram_probability",
    "letter_unigram_rank",
    "log_bigram_count",
    "log_trigram_count",
    "character_length",
    "bpe_length",
    "context_word_count",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_forbidden_dev(path: Path) -> None:
    reject_forbidden_evaluation_path(path)


def group_identifier(context: str, hint: str) -> str:
    return hashlib.sha256(f"{context}\x1f{hint}".encode("utf-8")).hexdigest()[:20]


def parse_float_list(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or any(not 0.0 <= item <= 1.0 for item in values):
        raise ValueError("alphas must be a non-empty comma list in [0, 1]")
    return values


def checkpoint_payload(path: Path) -> tuple[dict, int]:
    import torch

    # Resume snapshots contain Python/NumPy RNG state and therefore cannot use
    # PyTorch's tensor-only loader. Only pass trusted local training artifacts.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise ValueError(f"Checkpoint has no 'model' state: {path}")
    step = payload.get("step", payload.get("global_step"))
    if step is None:
        raise ValueError(f"Checkpoint has neither 'step' nor 'global_step': {path}")
    return payload, int(step)


def _model_source(checkpoint: Path, configured: str | None) -> str:
    if configured:
        return configured
    if (checkpoint.parent / "config.json").is_file():
        return str(checkpoint.parent)
    return "gpt2"


def _candidate_document(lexicon: hml.TrainingLexicon) -> dict:
    hints = {}
    for hint, token_ids in lexicon.candidate_ids_by_hint.items():
        hints[hint] = [
            {
                "token_id": int(token_id),
                "word": lexicon.hint_token_word[hint][token_id],
                "frequency": int(
                    lexicon.word_counts[lexicon.hint_token_word[hint][token_id]]
                ),
                "bpe_length": len(
                    lexicon.word_bpe[lexicon.hint_token_word[hint][token_id]]
                ),
            }
            for token_id in token_ids
        ]
    return {
        "hints": hints,
        "word_frequencies": {word: int(count) for word, count in lexicon.word_counts.items()},
        "word_roots": {
            word: int(token_ids[0])
            for word, token_ids in lexicon.word_bpe.items()
            if token_ids
        },
    }


def build_score_cache(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    reject_forbidden_dev(args.dev_path)
    for path in (args.checkpoint, args.train_path, args.dev_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to replace non-empty {args.output_dir}; use --overwrite")

    dev = pd.read_csv(args.dev_path, keep_default_na=False)
    validate_dev_frame(dev)
    source = _model_source(args.checkpoint, args.model_source)
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    lexicon = hml.build_training_lexicon(
        tokenizer, args.train_path, max_lines=args.max_train_lines
    )
    missing_hints = sorted(set(dev["first letter"]) - set(lexicon.candidate_ids_by_hint))
    if missing_hints:
        raise ValueError(f"No training candidates for hints: {missing_hints}")

    payload, step = checkpoint_payload(args.checkpoint)
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(source))
    model.load_state_dict(payload["model"])
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    counts = np.fromiter(
        (len(lexicon.candidate_ids_by_hint[hint]) for hint in dev["first letter"]),
        dtype=np.int64,
        count=len(dev),
    )
    offsets = np.empty(len(dev) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_scores = args.output_dir / "scores.tmp.npy"
    score_dtype = np.float16 if args.score_dtype == "float16" else np.float32
    scores = np.lib.format.open_memmap(
        temporary_scores, mode="w+", dtype=score_dtype, shape=(int(offsets[-1]),)
    )
    candidate_tensors = {
        hint: torch.tensor(ids, dtype=torch.long, device=device)
        for hint, ids in lexicon.candidate_ids_by_hint.items()
    }
    model_limit = getattr(model.config, "max_position_embeddings", args.max_context_tokens)
    context_limit = min(args.max_context_tokens, int(model_limit))
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(dev), args.batch_size):
            stop = min(start + args.batch_size, len(dev))
            chunk = dev.iloc[start:stop]
            encoded = tokenizer(
                chunk["context"].tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=context_limit,
            ).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    **encoded,
                    position_ids=hml._position_ids(encoded["attention_mask"]),
                    logits_to_keep=1,
                ).logits[:, -1, :]
            grouped_rows: dict[str, list[int]] = defaultdict(list)
            for local_row, hint in enumerate(chunk["first letter"]):
                grouped_rows[str(hint)].append(local_row)
            for hint, local_rows in grouped_rows.items():
                row_tensor = torch.tensor(local_rows, dtype=torch.long, device=device)
                restricted = logits.index_select(0, row_tensor).index_select(
                    1, candidate_tensors[hint]
                )
                values = restricted.float().log_softmax(1).cpu().numpy()
                for value_row, local_row in zip(values, local_rows):
                    row = start + local_row
                    scores[offsets[row] : offsets[row + 1]] = value_row.astype(
                        score_dtype
                    )
            scores.flush()
            print(f"Cached {stop:,}/{len(dev):,}", flush=True)
    del scores
    os.replace(temporary_scores, args.output_dir / "scores.npy")
    np.save(args.output_dir / "offsets.npy", offsets)
    examples = dev.copy()
    examples.insert(0, "example_id", np.arange(len(dev), dtype=np.int64))
    examples["group_id"] = [
        group_identifier(context, hint)
        for context, hint in zip(examples["context"], examples["first letter"])
    ]
    run_config_path = args.checkpoint.parent / "run_config.json"
    run_config = (
        json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config_path.is_file()
        else {}
    )
    selection_size = args.selection_size
    if selection_size is None:
        selection_size = run_config.get("eval_subset_size")
    selection_seed = args.selection_seed
    if selection_seed is None:
        selection_seed = run_config.get("seed")
    contaminated_groups: set[str] = set()
    if selection_size is not None:
        selection_size = int(selection_size)
        if not 0 <= selection_size <= len(examples) or selection_seed is None:
            raise ValueError("checkpoint selection size/seed are incomplete or invalid")
        selected = examples.sample(n=selection_size, random_state=int(selection_seed))
        contaminated_groups = set(selected["group_id"])
    examples["checkpoint_selection_contaminated"] = examples["group_id"].isin(
        contaminated_groups
    )
    examples.to_csv(args.output_dir / "examples.csv", index=False)
    candidates = _candidate_document(lexicon)
    atomic_text_write(
        json.dumps(candidates, ensure_ascii=True, separators=(",", ":")) + "\n",
        args.output_dir / "candidates.json",
    )
    candidate_sha = file_sha256(args.output_dir / "candidates.json")
    metadata = {
        "cache_version": CACHE_VERSION,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": step,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_metrics": payload.get("metrics"),
        "dev_path": str(args.dev_path),
        "dev_sha256": file_sha256(args.dev_path),
        "train_path": str(args.train_path),
        "train_sha256": file_sha256(args.train_path),
        "candidate_sha256": candidate_sha,
        "examples": len(dev),
        "score_values": int(offsets[-1]),
        "score_dtype": args.score_dtype,
        "restricted_log_probabilities": True,
        "model_source": source,
        "checkpoint_selection_size": selection_size,
        "checkpoint_selection_seed": selection_seed,
        "checkpoint_selection_contaminated_examples": int(
            examples["checkpoint_selection_contaminated"].sum()
        ),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_text_write(
        json.dumps(metadata, indent=2) + "\n", args.output_dir / "metadata.json"
    )


class ScoreCache:
    def __init__(self, label: str, directory: Path) -> None:
        self.label = label
        self.directory = directory
        self.metadata = json.loads((directory / "metadata.json").read_text("utf-8"))
        if self.metadata.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"Unsupported cache version in {directory}")
        self.examples = pd.read_csv(directory / "examples.csv", keep_default_na=False)
        self.offsets = np.load(directory / "offsets.npy", mmap_mode="r")
        self.scores = np.load(directory / "scores.npy", mmap_mode="r")
        self.candidates = json.loads((directory / "candidates.json").read_text("utf-8"))
        self.word_positions = {
            hint: {candidate["word"]: index for index, candidate in enumerate(candidates)}
            for hint, candidates in self.candidates["hints"].items()
        }
        self.token_positions = {
            hint: {
                int(candidate["token_id"]): index
                for index, candidate in enumerate(candidates)
            }
            for hint, candidates in self.candidates["hints"].items()
        }
        if len(self.offsets) != len(self.examples) + 1 or self.offsets[-1] != len(self.scores):
            raise ValueError(f"Corrupt offsets in {directory}")

    def row(self, index: int) -> np.ndarray:
        return np.asarray(
            self.scores[int(self.offsets[index]) : int(self.offsets[index + 1])],
            dtype=np.float64,
        )


def parse_cache_specs(specs: Sequence[str]) -> dict[str, ScoreCache]:
    result = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError("cache specifications must be LABEL=PATH")
        label, raw_path = spec.split("=", 1)
        if not label or label in result:
            raise ValueError(f"Invalid or duplicate cache label: {label!r}")
        result[label] = ScoreCache(label, Path(raw_path))
    if not result:
        raise ValueError("at least one --cache is required")
    reference = next(iter(result.values()))
    for cache in list(result.values())[1:]:
        for field in ("dev_sha256", "train_sha256", "candidate_sha256", "examples"):
            if cache.metadata[field] != reference.metadata[field]:
                raise ValueError(f"Cache {cache.label} has incompatible {field}")
        if not cache.offsets.shape == reference.offsets.shape or not np.array_equal(
            cache.offsets, reference.offsets
        ):
            raise ValueError(f"Cache {cache.label} has incompatible offsets")
    return result


def ranked_positions(scores: np.ndarray, top_k: int = 5) -> np.ndarray:
    count = min(top_k, len(scores))
    if count == len(scores):
        return np.argsort(-scores, kind="stable")
    selected = np.argpartition(-scores, count - 1)[:count]
    return selected[np.argsort(-scores[selected], kind="stable")]


def entropy_from_log_probs(scores: np.ndarray) -> float:
    probabilities = np.exp(scores)
    total = probabilities.sum()
    if total <= 0:
        return 0.0
    probabilities /= total
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log(positive)).sum())


def _hint_candidates(cache: ScoreCache, hint: str) -> list[dict]:
    return cache.candidates["hints"][hint]


def collect_predictions(
    cache: ScoreCache,
    score_rows: Iterable[np.ndarray] | None = None,
    *,
    checkpoint_label: str | None = None,
) -> pd.DataFrame:
    records = []
    rows = score_rows if score_rows is not None else (cache.row(i) for i in range(len(cache.examples)))
    for source, values in zip(cache.examples.to_dict("records"), rows):
        values = np.asarray(values, dtype=np.float64)
        hint = str(source["first letter"])
        choices = _hint_candidates(cache, hint)
        positions = ranked_positions(values)
        words = [choices[position]["word"] for position in positions]
        answer = str(source["answer"])
        true_word_position = cache.word_positions[hint].get(answer)
        true_root_id = cache.candidates["word_roots"].get(answer)
        true_root_position = cache.token_positions[hint].get(true_root_id)
        true_root_score = (
            float(values[true_root_position])
            if true_root_position is not None
            else math.nan
        )
        true_score = (
            float(values[true_word_position])
            if true_word_position is not None
            else math.nan
        )

        def rank(position: int | None) -> int | None:
            if position is None:
                return None
            return int(
                1
                + np.sum(values > values[position])
                + np.sum(values[:position] == values[position])
            )

        true_word_rank = (
            rank(true_word_position)
            if true_word_position == true_root_position
            else None
        )
        true_root_rank = rank(true_root_position)
        top1_score = float(values[positions[0]])
        top1_margin = (
            float(top1_score - values[positions[1]])
            if len(positions) > 1
            else math.inf
        )
        top1_minus_true = (
            float(top1_score - true_score) if math.isfinite(true_score) else math.nan
        )
        top_columns = {}
        for rank_number in range(1, 6):
            if rank_number <= len(positions):
                position = int(positions[rank_number - 1])
                top_columns[f"top{rank_number}_word"] = choices[position]["word"]
                top_columns[f"top{rank_number}_log_probability"] = float(values[position])
                top_columns[f"top{rank_number}_training_frequency"] = int(
                    choices[position]["frequency"]
                )
            else:
                top_columns[f"top{rank_number}_word"] = ""
                top_columns[f"top{rank_number}_log_probability"] = math.nan
                top_columns[f"top{rank_number}_training_frequency"] = 0
        true_training_frequency = int(
            cache.candidates["word_frequencies"].get(answer, 0)
        )
        records.append(
            {
                **source,
                "checkpoint": checkpoint_label or cache.label,
                "checkpoint_step": cache.metadata.get("checkpoint_step"),
                **top_columns,
                "top5_predictions": json.dumps(words),
                "top5_log_probabilities": json.dumps([float(values[p]) for p in positions]),
                "top5_frequencies": json.dumps([int(choices[p]["frequency"]) for p in positions]),
                "top5_ranks": json.dumps(list(range(1, len(positions) + 1))),
                "true_training_frequency": true_training_frequency,
                "true_root_id": true_root_id,
                "true_score": true_score,
                "true_root_score": true_root_score,
                "true_word_rank": true_word_rank,
                "true_root_rank": true_root_rank,
                "root_collision": true_word_position != true_root_position
                and true_root_position is not None,
                "top1_margin": top1_margin,
                "top1_minus_true_score_margin": top1_minus_true,
                "prediction": words[0],
                "correct": words[0] == answer,
                "top5_correct": answer in words,
                "true_word_in_top1": words[0] == answer,
                "true_word_in_top5": answer in words,
                "entropy": entropy_from_log_probs(values),
            }
        )
    return pd.DataFrame.from_records(records)


def prediction_metrics(details: pd.DataFrame) -> dict:
    return {
        "examples": len(details),
        "top1_correct": int(details["correct"].sum()),
        "top1_accuracy": float(details["correct"].mean()),
        "top5_correct": int(details["top5_correct"].sum()),
        "top5_accuracy": float(details["top5_correct"].mean()),
        "mean_entropy": float(details["entropy"].mean()),
        "rank2_to_5_errors": {
            str(rank): int(
                ((~details["correct"]) & details["true_word_rank"].eq(rank)).sum()
            )
            for rank in range(2, 6)
        },
    }


def per_letter_metrics(details: pd.DataFrame, cache: ScoreCache) -> pd.DataFrame:
    frame = details.copy()
    frequencies = cache.candidates["word_frequencies"]
    training_words_by_hint: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for word, count in frequencies.items():
        if word:
            training_words_by_hint[word[0]].append((word, int(count)))
    mode_by_hint = {
        hint: min(words, key=lambda item: (-item[1], item[0]))[0]
        for hint, words in training_words_by_hint.items()
    }
    frame["answer_frequency"] = frame["answer"].map(frequencies).fillna(0)
    frame["prediction_frequency"] = frame["prediction"].map(frequencies).fillna(0)
    frame["mode_correct"] = [
        answer == mode_by_hint[hint]
        for answer, hint in zip(frame["answer"], frame["first letter"])
    ]
    rows = []
    for hint, group in frame.groupby("first letter", sort=True):
        prediction_distribution = group["prediction"].value_counts(normalize=True)
        target_counts = group["answer"].value_counts()
        target_distribution = target_counts.to_numpy(dtype=np.float64) / len(group)
        target_entropy = float(
            -(target_distribution * np.log(target_distribution)).sum()
        )
        target_sorted = sorted(
            ((str(word), int(count)) for word, count in target_counts.items()),
            key=lambda item: (-item[1], item[0]),
        )
        most_common_target, most_common_count = target_sorted[0]
        rows.append(
            {
                "letter": hint,
                "examples": len(group),
                "top1_accuracy": float(group["correct"].mean()),
                "top5_accuracy": float(group["top5_correct"].mean()),
                "mean_candidate_entropy": float(group["entropy"].mean()),
                "unique_top1_predictions": int(group["prediction"].nunique()),
                "top1_prediction_entropy": float(
                    -(prediction_distribution * np.log(prediction_distribution)).sum()
                ),
                "target_entropy": target_entropy,
                "unique_target_words": int(group["answer"].nunique()),
                "most_common_target": most_common_target,
                "most_common_target_count": most_common_count,
                "most_common_target_share": most_common_count / len(group),
                "most_common_target_training_frequency": int(
                    frequencies.get(most_common_target, 0)
                ),
                "top5_target_cumulative_share": float(
                    sum(count for _word, count in target_sorted[:5]) / len(group)
                ),
                "top20_target_cumulative_share": float(
                    sum(count for _word, count in target_sorted[:20]) / len(group)
                ),
                "training_mode": mode_by_hint[hint],
                "training_mode_accuracy": float(group["mode_correct"].mean()),
                "candidate_count": len(cache.candidates["hints"][hint]),
                "mean_target_training_frequency": float(
                    group["answer_frequency"].mean()
                ),
                "median_target_training_frequency": float(
                    group["answer_frequency"].median()
                ),
                "mean_prediction_frequency": float(
                    group["prediction_frequency"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def frequency_metrics(details: pd.DataFrame, cache: ScoreCache) -> list[dict]:
    frequencies = cache.candidates["word_frequencies"]
    frame = details.copy()
    frame["answer_frequency"] = frame["answer"].map(frequencies).fillna(0).astype(int)
    bins = [-1, 0, 10, 100, 1_000, math.inf]
    labels = ["unrepresented", "1-10", "11-100", "101-1000", "1001+"]
    frame["frequency_band"] = pd.cut(frame["answer_frequency"], bins=bins, labels=labels)
    return [
        {
            "band": str(band),
            "examples": len(group),
            "top1_accuracy": float(group["correct"].mean()),
            "top5_accuracy": float(group["top5_correct"].mean()),
        }
        for band, group in frame.groupby("frequency_band", observed=True, sort=False)
    ]


def rank2_to_5_diagnostics(details: pd.DataFrame) -> dict[str, object]:
    errors = details.loc[~details["correct"]]
    recoverable = errors.loc[errors["true_word_rank"].between(2, 5)].copy()
    total = len(recoverable)
    rank_distribution = []
    for rank, group in recoverable.groupby("true_word_rank", sort=True):
        rank_distribution.append(
            {
                "rank": int(rank),
                "examples": len(group),
                "share_of_rank2_to_5": len(group) / total if total else 0.0,
                "share_of_all_examples": len(group) / len(details),
                "mean_score_gap": float(
                    group["top1_minus_true_score_margin"].mean()
                ),
                "median_score_gap": float(
                    group["top1_minus_true_score_margin"].median()
                ),
            }
        )
    per_letter = []
    for hint, group in details.groupby("first letter", sort=True):
        letter_errors = group.loc[~group["correct"]]
        letter_recoverable = letter_errors["true_word_rank"].between(2, 5)
        recoverable_group = letter_errors.loc[letter_recoverable]
        count = int(letter_recoverable.sum())
        per_letter.append(
            {
                "letter": hint,
                "examples": len(group),
                "errors": len(letter_errors),
                "rank2_to_5": count,
                "recoverable_share_of_errors": (
                    count / len(letter_errors) if len(letter_errors) else 0.0
                ),
                "recoverable_share_of_examples": count / len(group),
                "mean_score_gap": float(
                    recoverable_group["top1_minus_true_score_margin"].mean()
                ),
                "median_score_gap": float(
                    recoverable_group["top1_minus_true_score_margin"].median()
                ),
            }
        )
    confusions = (
        recoverable.groupby(["prediction", "answer"], as_index=False)
        .agg(
            examples=("example_id", "size"),
            mean_score_gap=("top1_minus_true_score_margin", "mean"),
            median_score_gap=("top1_minus_true_score_margin", "median"),
        )
        .sort_values(["examples", "prediction", "answer"], ascending=[False, True, True])
    )
    ordered = recoverable.sort_values(
        ["top1_minus_true_score_margin", "example_id"], ascending=[True, True]
    )
    return {
        "total": total,
        "total_examples": len(details),
        "total_errors": len(errors),
        "percentage_of_all_examples": total / len(details),
        "percentage_of_errors": total / len(errors) if len(errors) else 0.0,
        "mean_score_gap": float(
            recoverable["top1_minus_true_score_margin"].mean()
        ),
        "median_score_gap": float(
            recoverable["top1_minus_true_score_margin"].median()
        ),
        "rank_distribution": rank_distribution,
        "per_letter": per_letter,
        "recoverable": recoverable,
        "confusions": confusions,
        "smallest_margins": ordered.iloc[:100],
        "largest_margins": ordered.iloc[-100:].sort_values(
            ["top1_minus_true_score_margin", "example_id"],
            ascending=[False, True],
        ),
    }


def weighted_score_rows(
    caches: Sequence[ScoreCache], weights: Sequence[float]
) -> Iterable[np.ndarray]:
    if not caches or len(caches) != len(weights):
        raise ValueError("weighted ensembles need one weight per cache")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("ensemble weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("at least one ensemble weight must be positive")
    normalized = [weight / total for weight in weights]
    for index in range(len(caches[0].examples)):
        combined = np.zeros_like(caches[0].row(index), dtype=np.float64)
        for cache, weight in zip(caches, normalized):
            combined += weight * cache.row(index)
        yield combined


def comparison_metrics(details: pd.DataFrame, primary: pd.DataFrame) -> dict:
    if not details["example_id"].equals(primary["example_id"]):
        raise ValueError("comparison rows are not aligned by example_id")
    baseline = primary["correct"].to_numpy(dtype=bool)
    evaluated = details["correct"].to_numpy(dtype=bool)
    rank_bucket = (
        ~baseline
        & primary["true_word_rank"].between(2, 5).to_numpy(dtype=bool)
    )
    improvements = int((~baseline & evaluated).sum())
    breakages = int((baseline & ~evaluated).sum())
    bucket_denominator = int(rank_bucket.sum())
    bucket_promotions = int((rank_bucket & evaluated).sum())
    breakage_denominator = int(baseline.sum())
    return {
        "top1_correct": int(evaluated.sum()),
        "top1_accuracy": float(evaluated.mean()),
        "top5_correct": int(details["top5_correct"].sum()),
        "top5_accuracy": float(details["top5_correct"].mean()),
        "top1_accuracy_change": float(evaluated.mean() - baseline.mean()),
        "improvements": improvements,
        "breakages": breakages,
        "net": improvements - breakages,
        "primary_rank2_to_5_denominator": bucket_denominator,
        "primary_rank2_to_5_promotions": bucket_promotions,
        "primary_rank2_to_5_promotion_rate": (
            bucket_promotions / bucket_denominator if bucket_denominator else 0.0
        ),
        "breakage_denominator": breakage_denominator,
        "breakage_rate": (
            breakages / breakage_denominator if breakage_denominator else 0.0
        ),
    }


def ensemble_sweep(
    left: ScoreCache,
    right: ScoreCache,
    alphas: Sequence[float],
    primary_details: pd.DataFrame,
) -> pd.DataFrame:
    summary = []
    for alpha in alphas:
        rows = weighted_score_rows([left, right], [alpha, 1.0 - alpha])
        details = collect_predictions(
            left, rows, checkpoint_label=f"{alpha:g}*{left.label}+{1-alpha:g}*{right.label}"
        )
        summary.append(
            {
                "ensemble": f"alpha={alpha:g}",
                "weights": json.dumps(
                    {left.label: alpha, right.label: 1.0 - alpha}, sort_keys=True
                ),
                "alpha": alpha,
                **comparison_metrics(details, primary_details),
            }
        )
    return pd.DataFrame(summary)


def parse_weighted_ensemble(
    spec: str, caches: dict[str, ScoreCache]
) -> tuple[list[ScoreCache], list[float], dict[str, float]]:
    mapping = {}
    for item in spec.split(","):
        if "=" not in item:
            raise ValueError("weighted ensemble must be LABEL=WEIGHT comma entries")
        label, raw_weight = item.split("=", 1)
        if label not in caches or label in mapping:
            raise ValueError(f"unknown or duplicate ensemble cache: {label!r}")
        mapping[label] = float(raw_weight)
    selected = [caches[label] for label in mapping]
    weights = list(mapping.values())
    # Trigger all validation before the expensive analysis loop.
    if any(not math.isfinite(weight) or weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("ensemble weights must be non-negative with a positive sum")
    return selected, weights, mapping


def assign_grouped_split(
    group_ids: Sequence[str], contaminated: Sequence[bool] | None = None
) -> list[str]:
    """Stable 60/20/20 split; identical context/hint groups cannot cross splits."""
    result = []
    flags = contaminated if contaminated is not None else [False] * len(group_ids)
    for group_id, is_contaminated in zip(group_ids, flags):
        if is_contaminated:
            result.append("contaminated")
            continue
        bucket = int(hashlib.sha256(group_id.encode("ascii")).hexdigest()[:8], 16) / 2**32
        result.append("train" if bucket < 0.6 else "validation" if bucket < 0.8 else "test")
    return result


def top5_reranker_rows(
    details: pd.DataFrame, score_rows: Iterable[np.ndarray], cache: ScoreCache
) -> pd.DataFrame:
    rows = []
    hint_totals = {
        hint: sum(int(candidate["frequency"]) for candidate in candidates)
        for hint, candidates in cache.candidates["hints"].items()
    }
    hint_frequency_ranks = {
        hint: {
            candidate["word"]: rank
            for rank, candidate in enumerate(
                sorted(candidates, key=lambda item: (-item["frequency"], item["word"])), 1
            )
        }
        for hint, candidates in cache.candidates["hints"].items()
    }
    for source, values in zip(details.to_dict("records"), score_rows):
        hint = str(source["first letter"])
        candidates = _hint_candidates(cache, hint)
        positions = ranked_positions(values)
        context_words = str(source["context"]).split()
        prev1 = context_words[-1] if context_words else ""
        prev2 = context_words[-2] if len(context_words) > 1 else ""
        is_contaminated = source.get("checkpoint_selection_contaminated", False)
        if isinstance(is_contaminated, str):
            is_contaminated = is_contaminated.casefold() == "true"
        for rank, position in enumerate(positions, 1):
            candidate = candidates[int(position)]
            count = int(candidate["frequency"])
            rows.append(
                {
                    "example_id": int(source["example_id"]),
                    "group_id": source["group_id"],
                    "split": assign_grouped_split(
                        [source["group_id"]], [bool(is_contaminated)]
                    )[0],
                    "hint": hint,
                    "answer": source["answer"],
                    "baseline_prediction": source["prediction"],
                    "true_word_rank": source["true_word_rank"],
                    "candidate": candidate["word"],
                    "label": int(candidate["word"] == source["answer"]),
                    "model_log_probability": float(values[position]),
                    "model_rank": rank,
                    "top1_margin": source["top1_margin"],
                    "unigram_count": count,
                    "log_unigram_count": math.log1p(count),
                    "letter_unigram_probability": count / max(1, hint_totals[hint]),
                    "letter_unigram_rank": hint_frequency_ranks[hint][candidate["word"]],
                    "previous_word": prev1,
                    "previous_two_word": prev2,
                    "bigram_count": 0,
                    "trigram_count": 0,
                    "character_length": len(candidate["word"]),
                    "bpe_length": int(candidate["bpe_length"]),
                    "context_word_count": len(context_words),
                }
            )
    result = pd.DataFrame.from_records(rows)
    result["group_has_positive"] = result.groupby("example_id")["label"].transform(
        "any"
    )
    result["ranking_group_size"] = result.groupby("example_id")[
        "example_id"
    ].transform("size")
    return result


def add_targeted_ngram_counts(rows: pd.DataFrame, train_path: Path) -> None:
    targets2 = set(zip(rows["previous_word"], rows["candidate"]))
    targets3 = set(zip(rows["previous_two_word"], rows["previous_word"], rows["candidate"]))
    counts2: Counter = Counter()
    counts3: Counter = Counter()
    with train_path.open("r", encoding="utf-8") as corpus:
        for line in corpus:
            words = line.split()
            for index in range(1, len(words)):
                pair = (words[index - 1], words[index])
                if pair in targets2:
                    counts2[pair] += 1
                if index >= 2:
                    triple = (words[index - 2], words[index - 1], words[index])
                    if triple in targets3:
                        counts3[triple] += 1
    rows["bigram_count"] = [counts2[pair] for pair in zip(rows["previous_word"], rows["candidate"])]
    rows["trigram_count"] = [
        counts3[triple]
        for triple in zip(rows["previous_two_word"], rows["previous_word"], rows["candidate"])
    ]
    rows["log_bigram_count"] = np.log1p(rows["bigram_count"])
    rows["log_trigram_count"] = np.log1p(rows["trigram_count"])


def transition_metrics(frame: pd.DataFrame, prediction_column: str) -> dict:
    baseline = frame["baseline_prediction"].eq(frame["answer"])
    reranked = frame[prediction_column].eq(frame["answer"])
    promotion = int((~baseline & reranked).sum())
    breakage = int((baseline & ~reranked).sum())
    eligible = (
        frame["group_has_positive"].astype(bool)
        if "group_has_positive" in frame
        else ~baseline
    )
    promotion_denominator = int((~baseline & eligible).sum())
    breakage_denominator = int(baseline.sum())
    return {
        "examples": len(frame),
        "baseline_accuracy": float(baseline.mean()),
        "reranked_accuracy": float(reranked.mean()),
        "promotion": promotion,
        "promotion_denominator": promotion_denominator,
        "promotion_rate": (
            promotion / promotion_denominator if promotion_denominator else 0.0
        ),
        "breakage": breakage,
        "breakage_denominator": breakage_denominator,
        "breakage_rate": (
            breakage / breakage_denominator if breakage_denominator else 0.0
        ),
        "net": promotion - breakage,
        "net_accuracy_points": float((promotion - breakage) / len(frame)) if len(frame) else 0.0,
    }


def train_and_evaluate_reranker(rows: pd.DataFrame, seed: int) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    train = rows.loc[rows["split"] == "train"]
    labels = train["label"].to_numpy()
    if train.empty or len(np.unique(labels)) != 2:
        raise ValueError("Reranker train split needs positive and negative rows")
    group_sizes = train.groupby("example_id")["example_id"].transform("size").to_numpy()
    positives = max(1, int(labels.sum()))
    negatives = max(1, len(labels) - positives)
    class_weights = np.where(labels == 1, len(labels) / (2 * positives), len(labels) / (2 * negatives))
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(train[RERANK_FEATURES], labels, sample_weight=class_weights / group_sizes)
    results = {"features": RERANK_FEATURES, "training_rows": len(train)}
    for split in ("validation", "test"):
        pool = rows.loc[rows["split"] == split].copy()
        pool["reranker_score"] = model.predict_proba(pool[RERANK_FEATURES])[:, 1]
        chosen = (
            pool.sort_values(
                ["example_id", "reranker_score", "model_rank", "candidate"],
                ascending=[True, False, True, True],
            )
            .drop_duplicates("example_id")
            .loc[
                :,
                [
                    "example_id",
                    "answer",
                    "baseline_prediction",
                    "candidate",
                    "group_has_positive",
                ],
            ]
            .rename(columns={"candidate": "reranked_prediction"})
        )
        results[split] = transition_metrics(chosen, "reranked_prediction")
    return results


def make_plots(
    output_dir: Path,
    details: pd.DataFrame,
    per_letter: pd.DataFrame,
    checkpoints: dict[str, dict],
    ensemble: pd.DataFrame | None,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    def save_bar(column: str, ylabel: str, title: str, filename: str) -> None:
        fig, axis = plt.subplots(figsize=(10, 4))
        axis.bar(per_letter["letter"], 100 * per_letter[column])
        axis.set(xlabel="First-character hint", ylabel=ylabel, title=title)
        fig.tight_layout()
        path = plots / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(output_dir)))

    save_bar("top1_accuracy", "Top-1 accuracy (%)", "Top-1 Accuracy by Letter", "top1_by_letter.png")
    save_bar("top5_accuracy", "Top-5 accuracy (%)", "Top-5 Accuracy by Letter", "top5_by_letter.png")

    for x_column, x_label, title, filename in (
        ("target_entropy", "Target entropy (nats)", "Target Entropy vs Top-1 Accuracy", "target_entropy_vs_accuracy.png"),
        ("examples", "Examples", "Examples vs Top-1 Accuracy", "examples_vs_accuracy.png"),
    ):
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(per_letter[x_column], 100 * per_letter["top1_accuracy"])
        for row in per_letter.to_dict("records"):
            axis.annotate(row["letter"], (row[x_column], 100 * row["top1_accuracy"]), fontsize=8)
        axis.set(xlabel=x_label, ylabel="Top-1 accuracy (%)", title=title)
        fig.tight_layout()
        path = plots / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(output_dir)))

    margins = details["top1_minus_true_score_margin"].replace([np.inf, -np.inf], np.nan).dropna()
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.hist(margins, bins=80)
    axis.set(
        xlabel="Top-1 log probability minus true-root log probability",
        ylabel="Examples",
        title="True-Word Score Margin Distribution",
    )
    fig.tight_layout()
    path = plots / "true_word_score_margin_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(output_dir)))

    labels = list(checkpoints)
    top1 = [100 * checkpoints[label]["metrics"]["top1_accuracy"] for label in labels]
    top5 = [100 * checkpoints[label]["metrics"]["top5_accuracy"] for label in labels]
    if ensemble is not None and not ensemble.empty:
        best = ensemble.sort_values(["top1_accuracy", "top5_accuracy"], ascending=False).iloc[0]
        labels.append(str(best["ensemble"]))
        top1.append(100 * float(best["top1_accuracy"]))
        top5.append(100 * float(best["top5_accuracy"]))
    positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(max(7, len(labels) * 1.4), 4))
    axis.bar(positions - 0.2, top1, width=0.4, label="top-1")
    axis.bar(positions + 0.2, top5, width=0.4, label="top-5")
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set(ylabel="Accuracy (%)", title="Checkpoint and Ensemble Accuracy Comparison")
    axis.legend()
    fig.tight_layout()
    path = plots / "checkpoint_ensemble_accuracy_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path.relative_to(output_dir)))
    return paths


def markdown_report(report: dict) -> str:
    lines = ["# Comprehensive Hint-Restricted Checkpoint Evaluation", ""]
    lines.extend([
        "## Provenance",
        "",
        f"- Development examples: {report['provenance']['examples']:,}",
        f"- Development SHA-256: `{report['provenance']['dev_sha256']}`",
        f"- Training SHA-256: `{report['provenance']['train_sha256']}`",
        "- Probabilities are normalized over every root candidate allowed by the supplied hint.",
        "- Reranker n-grams were counted only in the training corpus; context/hint duplicates share one split.",
        "",
        "## Checkpoints",
        "",
        "| Checkpoint | Step | Top-1 | Top-5 | Mean entropy |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, result in report["checkpoints"].items():
        metric = result["metrics"]
        lines.append(
            f"| {label} | {result['step']:,} | {metric['top1_accuracy']:.4%} | "
            f"{metric['top5_accuracy']:.4%} | {metric['mean_entropy']:.4f} |"
        )
    primary = report["primary"]
    lines.extend(["", "## Rank 2-5 Errors", "", "| Rank | Errors |", "|---:|---:|"])
    for rank, count in report["checkpoints"][primary]["metrics"]["rank2_to_5_errors"].items():
        lines.append(f"| {rank} | {count:,} |")
    if report.get("ensemble"):
        best = max(report["ensemble"]["sweep"], key=lambda row: row["top1_accuracy"])
        lines.extend([
            "",
            "## Ensemble Sweep",
            "",
            f"Descriptively best ensemble is **{best['ensemble']}** ({best['weights']}) with top-1 accuracy "
            f"**{best['top1_accuracy']:.4%}**. All requested alphas are retained in JSON/CSV; "
            "this full-dev comparison is not treated as a held-out model-selection estimate.",
        ])
    reranker = report.get("reranker")
    if reranker:
        test = reranker["test"]
        lines.extend([
            "",
            "## Held-Out Top-5 Reranker",
            "",
            f"- Baseline accuracy: {test['baseline_accuracy']:.4%}",
            f"- Reranked accuracy: {test['reranked_accuracy']:.4%}",
            f"- Promotions: {test['promotion']:,}",
            f"- Promotion rate: {test['promotion']:,}/{test['promotion_denominator']:,} "
            f"({test['promotion_rate']:.2%})",
            f"- Breakages: {test['breakage']:,}",
            f"- Breakage rate: {test['breakage']:,}/{test['breakage_denominator']:,} "
            f"({test['breakage_rate']:.2%})",
            f"- Net: {test['net']:+,} ({test['net_accuracy_points']:+.4%})",
        ])
    lines.extend(["", "## Evidence-Based Interpretation", ""])
    for category, text in report["interpretation"].items():
        lines.append(f"- **{category.title()}:** {text}")
    lines.extend(["", "## Prioritized Recommendations", ""])
    lines.extend(
        f"{index}. {recommendation}"
        for index, recommendation in enumerate(report["recommendations"], 1)
    )
    lines.extend(["", "## Artifacts", ""])
    lines.extend(f"- `{path}`" for path in report["artifacts"])
    return "\n".join(lines) + "\n"


def evidence_interpretation(
    primary_details: pd.DataFrame,
    primary_metrics: dict,
    per_letter: pd.DataFrame,
    frequency: list[dict],
    rank_diagnostics: dict,
    ensemble: pd.DataFrame | None,
    disagreements: dict[str, int],
) -> tuple[dict[str, str], list[str]]:
    rank_total = int(rank_diagnostics["total"])
    rare_rows = [row for row in frequency if row["band"] in ("unrepresented", "1-10", "11-100")]
    rare_examples = sum(row["examples"] for row in rare_rows)
    rare_correct = sum(row["examples"] * row["top1_accuracy"] for row in rare_rows)
    rare_accuracy = rare_correct / rare_examples if rare_examples else 0.0
    highest_entropy = per_letter.sort_values("target_entropy", ascending=False).iloc[0]
    lowest_entropy = per_letter.sort_values("target_entropy").iloc[0]
    collisions = int(primary_details["root_collision"].sum())
    instability = (
        ", ".join(f"{label}: {count:,}" for label, count in disagreements.items())
        if disagreements
        else "no secondary checkpoint was supplied"
    )
    if ensemble is not None and not ensemble.empty:
        best = ensemble.sort_values(["top1_accuracy", "top5_accuracy"], ascending=False).iloc[0]
        instability += (
            f"; best evaluated ensemble {best['ensemble']} changed top-1 by "
            f"{best['top1_accuracy_change']:+.4%}, with {int(best['improvements']):,} "
            f"improvements and {int(best['breakages']):,} breakages"
        )
    interpretation = {
        "ranking": (
            f"{rank_total:,} examples ({rank_diagnostics['percentage_of_errors']:.2%} of "
            "top-1 errors) have the exact target representative at ranks 2-5; these are "
            "direct ranking opportunities rather than vocabulary misses."
        ),
        "modeling": (
            f"The primary checkpoint reaches {primary_metrics['top1_accuracy']:.2%} top-1 and "
            f"{primary_metrics['top5_accuracy']:.2%} top-5 accuracy. {collisions:,} targets "
            "share a root with a different representative, separating root modeling from "
            "full-word decoding failures."
        ),
        "data": (
            f"Target distributions vary from {float(lowest_entropy['target_entropy']):.2f} to "
            f"{float(highest_entropy['target_entropy']):.2f} nats by letter; the all-training-word "
            "mode baseline and target concentration columns quantify how much can be explained "
            "without context."
        ),
        "rare": (
            f"Targets seen at most 100 times or absent from training account for {rare_examples:,} "
            f"examples and have {rare_accuracy:.2%} top-1 accuracy, indicating the measured "
            "rare-word data regime."
        ),
        "high-entropy": (
            f"Letter {highest_entropy['letter']!r} has the highest empirical target entropy "
            f"({float(highest_entropy['target_entropy']):.2f} nats) and "
            f"{float(highest_entropy['top1_accuracy']):.2%} top-1 accuracy, so per-letter "
            "difficulty should be interpreted against target diversity."
        ),
        "instability": f"Top-1 disagreements versus the primary checkpoint are {instability}.",
    }
    recommendations = [
        f"Prioritize a leakage-safe reranker on the {rank_total:,} rank-2-to-5 cases, using held-out promotion and breakage rates as the acceptance criterion.",
        f"Add collision-aware full-word scoring for the {collisions:,} collision targets; root-only ranking cannot choose the correct surface word in those cases.",
        f"Target augmentation or vocabulary/tokenizer changes at the {rare_examples:,} rare-target examples rather than treating them as ordinary reranking errors.",
        f"Stratify future diagnostics by target entropy, beginning with letter {highest_entropy['letter']!r}, to avoid attributing distributional difficulty entirely to model quality.",
        "Adopt an ensemble only if its locked promotion gain exceeds its measured breakage and operational cost; the full-dev alpha sweep is descriptive, not a selection-safe estimate.",
    ]
    return interpretation, recommendations


def analyze(args: argparse.Namespace) -> None:
    reject_forbidden_dev(args.dev_path)
    caches = parse_cache_specs(args.cache)
    primary_label = args.primary or next(iter(caches))
    if primary_label not in caches:
        raise ValueError(f"Unknown primary cache: {primary_label}")
    primary = caches[primary_label]
    if file_sha256(args.dev_path) != primary.metadata["dev_sha256"]:
        raise ValueError("--dev-path does not match cache provenance")
    if file_sha256(args.train_path) != primary.metadata["train_sha256"]:
        raise ValueError("--train-path does not match cache provenance")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_details = []
    details_by_label = {}
    checkpoint_report = {}
    primary_details = None
    for label, cache in caches.items():
        details = collect_predictions(cache)
        all_details.append(details)
        details_by_label[label] = details
        metrics = prediction_metrics(details)
        checkpoint_report[label] = {
            "step": int(cache.metadata["checkpoint_step"]),
            "checkpoint_sha256": cache.metadata["checkpoint_sha256"],
            "metrics": metrics,
            "frequency_metrics": frequency_metrics(details, cache),
        }
        if label == primary_label:
            primary_details = details
    assert primary_details is not None
    pd.concat(all_details, ignore_index=True).to_csv(
        args.output_dir / "top5_predictions.csv", index=False
    )
    per_letter = per_letter_metrics(primary_details, primary)
    per_letter.to_csv(args.output_dir / "per_letter_metrics.csv", index=False)
    rank_diagnostics = rank2_to_5_diagnostics(primary_details)
    rank_diagnostics["recoverable"].to_csv(
        args.output_dir / "rank2_to_5_errors.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "total": rank_diagnostics["total"],
                "total_examples": rank_diagnostics["total_examples"],
                "total_errors": rank_diagnostics["total_errors"],
                "percentage_of_all_examples": rank_diagnostics[
                    "percentage_of_all_examples"
                ],
                "percentage_of_errors": rank_diagnostics["percentage_of_errors"],
                "mean_score_gap": rank_diagnostics["mean_score_gap"],
                "median_score_gap": rank_diagnostics["median_score_gap"],
            }
        ]
    ).to_csv(args.output_dir / "rank2_to_5_overall.csv", index=False)
    pd.DataFrame(rank_diagnostics["rank_distribution"]).to_csv(
        args.output_dir / "rank2_to_5_summary.csv", index=False
    )
    pd.DataFrame(rank_diagnostics["per_letter"]).to_csv(
        args.output_dir / "rank2_to_5_per_letter.csv", index=False
    )
    rank_diagnostics["confusions"].to_csv(
        args.output_dir / "rank2_to_5_confusion_pairs.csv", index=False
    )
    rank_diagnostics["smallest_margins"].to_csv(
        args.output_dir / "rank2_to_5_smallest_margins.csv", index=False
    )
    rank_diagnostics["largest_margins"].to_csv(
        args.output_dir / "rank2_to_5_largest_margins.csv", index=False
    )

    ensemble_frame = None
    ensemble_report = None
    ensemble_parts = []
    if args.ensemble:
        pair = [item.strip() for item in args.ensemble.split(",")]
        if len(pair) != 2 or any(label not in caches for label in pair):
            raise ValueError("--ensemble must name two loaded cache labels")
        ensemble_parts.append(
            ensemble_sweep(
                caches[pair[0]],
                caches[pair[1]],
                parse_float_list(args.alphas),
                primary_details,
            )
        )
    for index, spec in enumerate(args.weighted_ensemble or [], 1):
        selected, weights, mapping = parse_weighted_ensemble(spec, caches)
        label = f"weighted-{index}"
        details = collect_predictions(
            selected[0],
            weighted_score_rows(selected, weights),
            checkpoint_label=label,
        )
        ensemble_parts.append(
            pd.DataFrame(
                [
                    {
                        "ensemble": label,
                        "weights": json.dumps(mapping, sort_keys=True),
                        "alpha": "",
                        **comparison_metrics(details, primary_details),
                    }
                ]
            )
        )
    if ensemble_parts:
        ensemble_frame = pd.concat(ensemble_parts, ignore_index=True)
        ensemble_frame.to_csv(args.output_dir / "ensemble_results.csv", index=False)
        if args.ensemble:
            ensemble_frame.loc[ensemble_frame["alpha"].ne("")].to_csv(
                args.output_dir / "ensemble_alpha_sweep.csv", index=False
            )
        ensemble_report = {"sweep": ensemble_frame.to_dict("records")}

    comparison_rows = []
    for label, details in details_by_label.items():
        comparison_rows.append(
            {
                "kind": "checkpoint",
                "name": label,
                "step": checkpoint_report[label]["step"],
                "weights": "",
                "alpha": "",
                **comparison_metrics(details, primary_details),
            }
        )
    if ensemble_frame is not None:
        for result in ensemble_frame.to_dict("records"):
            row = dict(result)
            name = row.pop("ensemble")
            comparison_rows.append(
                {"kind": "ensemble", "name": name, "step": "", **row}
            )
    pd.DataFrame(comparison_rows).to_csv(
        args.output_dir / "checkpoint_ensemble_summary.csv", index=False
    )

    reranker_rows = top5_reranker_rows(
        primary_details,
        (primary.row(index) for index in range(len(primary.examples))),
        primary,
    )
    add_targeted_ngram_counts(reranker_rows, args.train_path)
    reranker_rows.to_csv(args.output_dir / "top5_reranker_rows.csv", index=False)
    reranker_report = (
        None
        if args.skip_reranker
        else train_and_evaluate_reranker(reranker_rows, args.seed)
    )

    frequency = checkpoint_report[primary_label]["frequency_metrics"]
    pd.DataFrame(frequency).to_csv(
        args.output_dir / "frequency_metrics.csv", index=False
    )
    disagreements = {
        label: int((~details["prediction"].eq(primary_details["prediction"])).sum())
        for label, details in details_by_label.items()
        if label != primary_label
    }
    plots = make_plots(
        args.output_dir,
        primary_details,
        per_letter,
        checkpoint_report,
        ensemble_frame,
    )
    artifacts = [
        "top5_predictions.csv",
        "rank2_to_5_errors.csv",
        "rank2_to_5_overall.csv",
        "rank2_to_5_summary.csv",
        "rank2_to_5_per_letter.csv",
        "rank2_to_5_confusion_pairs.csv",
        "rank2_to_5_smallest_margins.csv",
        "rank2_to_5_largest_margins.csv",
        "per_letter_metrics.csv",
        "frequency_metrics.csv",
        "top5_reranker_rows.csv",
        "checkpoint_ensemble_summary.csv",
        *(["ensemble_alpha_sweep.csv"] if args.ensemble else []),
        *(["ensemble_results.csv"] if ensemble_frame is not None else []),
        *plots,
    ]
    rank_report = {
        key: rank_diagnostics[key]
        for key in (
            "total",
            "total_examples",
            "total_errors",
            "percentage_of_all_examples",
            "percentage_of_errors",
            "mean_score_gap",
            "median_score_gap",
            "rank_distribution",
            "per_letter",
        )
    }
    interpretation, recommendations = evidence_interpretation(
        primary_details,
        checkpoint_report[primary_label]["metrics"],
        per_letter,
        frequency,
        rank_report,
        ensemble_frame,
        disagreements,
    )
    report = {
        "provenance": {
            "examples": len(primary.examples),
            "dev_path": str(args.dev_path),
            "dev_sha256": primary.metadata["dev_sha256"],
            "train_path": str(args.train_path),
            "train_sha256": primary.metadata["train_sha256"],
            "candidate_sha256": primary.metadata["candidate_sha256"],
        },
        "primary": primary_label,
        "checkpoints": checkpoint_report,
        "per_letter": per_letter.to_dict("records"),
        "rank2_to_5": rank_report,
        "ensemble": ensemble_report,
        "checkpoint_disagreements_vs_primary": disagreements,
        "reranker": reranker_report,
        "interpretation": interpretation,
        "recommendations": recommendations,
        "split_counts": {
            str(split): int(count)
            for split, count in reranker_rows.drop_duplicates("example_id")["split"]
            .value_counts()
            .items()
        },
        "artifacts": artifacts + ["report.json", "report.md"],
    }
    atomic_text_write(
        json.dumps(report, indent=2) + "\n", args.output_dir / "report.json"
    )
    atomic_text_write(markdown_report(report), args.output_dir / "report.md")
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "report.json"),
                "artifacts": report["artifacts"],
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache = subparsers.add_parser(
        "cache", help="cache every hint-restricted log probability"
    )
    cache.add_argument("--checkpoint", type=Path, required=True)
    cache.add_argument("--output-dir", type=Path, required=True)
    cache.add_argument("--train-path", type=Path, default=TRAIN_DIR / "train.src.tok")
    cache.add_argument("--dev-path", type=Path, default=DATA_DIR / "devv_eval.csv")
    cache.add_argument(
        "--model-source",
        help="local Hugging Face directory or model name; defaults to checkpoint directory",
    )
    cache.add_argument("--device")
    cache.add_argument("--batch-size", type=int, default=64)
    cache.add_argument("--max-context-tokens", type=int, default=256)
    cache.add_argument("--max-train-lines", type=int)
    cache.add_argument(
        "--selection-size",
        type=int,
        help="checkpoint-selection row count; defaults to run_config.json",
    )
    cache.add_argument(
        "--selection-seed",
        type=int,
        help="checkpoint-selection seed; defaults to run_config.json",
    )
    cache.add_argument("--score-dtype", choices=("float16", "float32"), default="float32")
    cache.add_argument("--overwrite", action="store_true")

    analysis = subparsers.add_parser(
        "analyze", help="analyze compatible score caches without loading models"
    )
    analysis.add_argument("--cache", action="append", required=True, metavar="LABEL=PATH")
    analysis.add_argument(
        "--primary", help="cache label used for detailed diagnostics and reranking"
    )
    analysis.add_argument("--ensemble", help="two cache labels, e.g. step126,step258")
    analysis.add_argument(
        "--weighted-ensemble",
        action="append",
        metavar="LABEL=WEIGHT,...",
        help="additional arbitrary weighted log-probability ensemble",
    )
    analysis.add_argument("--alphas", default=DEFAULT_ALPHAS)
    analysis.add_argument("--train-path", type=Path, default=TRAIN_DIR / "train.src.tok")
    analysis.add_argument("--dev-path", type=Path, default=DATA_DIR / "devv_eval.csv")
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--seed", type=int, default=42)
    analysis.add_argument(
        "--skip-reranker",
        action="store_true",
        help="write feature rows but do not fit sklearn model",
    )
    args = parser.parse_args()
    if getattr(args, "batch_size", 1) < 1 or getattr(args, "max_context_tokens", 1) < 1:
        parser.error("batch size and max context tokens must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "cache":
        build_score_cache(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
