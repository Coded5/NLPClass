#!/usr/bin/env python3
"""Two-stage, shared-context neural reranking experiment.

The GPT-2 score cache is treated as immutable input.  MiniLM is only used to
encode text; the trainable component is an MLP over cached embeddings and
scalar features, not a cross-encoder.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import pickle
import resource
import time
from typing import Iterable, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import hint_masked_lib as hml
from .comprehensive_checkpoint_evaluation import (
    ScoreCache,
    entropy_from_log_probs,
    group_identifier,
    ranked_positions,
)
from .data_policy import reject_forbidden_evaluation_path
from .paths import ARTIFACTS_DIR, DATA_DIR, TRAIN_DIR
from .rerank_experiment import file_sha256, grouped_fold


DEFAULT_CACHE = ARTIFACTS_DIR / "comprehensive-evaluation/cache-step258"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SCHEMA_VERSION = 3
SCALAR_COLUMNS = [
    "root_log_probability", "root_rank", "within_root_rank", "root_margin",
    "root_entropy", "log_word_count", "root_candidate_count", "bpe_length",
    "char_length", "is_numeric", "is_alpha", "is_representative",
]
REQUIRED_POOL_COLUMNS = {
    "example_id", "group_id", "context", "hint", "answer", "contamination",
    "baseline_prediction", "candidate", "candidate_correct", "root_id",
    "root_rank", "root_log_probability", "root_margin", "root_entropy",
    "within_root_rank", "word_count", "log_word_count", "root_candidate_count", "bpe_length",
    "char_length", "is_numeric", "is_alpha",
    "is_representative", "candidate_miss",
}
PREPARED_FILES = (
    "candidate_pool.parquet",
    "context_embeddings.npy",
    "candidate_embeddings.npy",
    "contexts.csv",
    "words.csv",
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def environment_provenance() -> dict:
    return {
        "packages": {
            name: version(name)
            for name in (
                "numpy", "pandas", "pyarrow", "scikit-learn", "torch",
                "transformers", "huggingface-hub",
            )
        },
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def reject_forbidden_path(path: Path | str) -> None:
    reject_forbidden_evaluation_path(path)


def validate_cache_provenance(cache: ScoreCache, train_path: Path, dev_path: Path) -> None:
    reject_forbidden_path(dev_path)
    reject_forbidden_path(cache.metadata.get("dev_path", ""))
    if not train_path.is_file() or not dev_path.is_file():
        raise FileNotFoundError("train/dev provenance file is missing")
    if file_sha256(train_path) != cache.metadata.get("train_sha256"):
        raise ValueError("training corpus hash does not match score cache")
    if file_sha256(dev_path) != cache.metadata.get("dev_sha256"):
        raise ValueError("development file hash does not match score cache")
    if cache.metadata.get("dev_path"):
        reject_forbidden_path(cache.metadata["dev_path"])
    if not cache.metadata.get("restricted_log_probabilities"):
        raise ValueError("score cache is not restricted-root log-probability data")
    if cache.metadata.get("checkpoint_step") != 258_000:
        raise ValueError("neural reranking requires the step-258000 score cache")
    if file_sha256(cache.directory / "candidates.json") != cache.metadata.get(
        "candidate_sha256"
    ):
        raise ValueError("candidate file does not match cache provenance")
    if len(cache.examples) != cache.metadata.get("examples"):
        raise ValueError("cache example count does not match metadata")
    if int(cache.offsets[-1]) != cache.metadata.get("score_values"):
        raise ValueError("cache score count does not match metadata")

    expected_ids = np.arange(len(cache.examples), dtype=np.int64)
    if not np.array_equal(cache.examples["example_id"].to_numpy(), expected_ids):
        raise ValueError("cache example IDs are not contiguous and ordered")
    development = pd.read_csv(dev_path, keep_default_na=False)
    for column in ("context", "first letter", "answer"):
        if column not in development or not cache.examples[column].equals(
            development[column]
        ):
            raise ValueError(
                f"cached examples do not match development column {column!r}"
            )
    expected_groups = np.asarray(
        [
            group_identifier(str(context), str(hint))
            for context, hint in zip(
                development["context"], development["first letter"]
            )
        ]
    )
    if not np.array_equal(cache.examples["group_id"].to_numpy(), expected_groups):
        raise ValueError("cached group IDs do not match context/hint groups")

    hints = cache.examples["first letter"].astype(str)
    unknown_hints = sorted(set(hints) - set(cache.candidates["hints"]))
    if unknown_hints:
        raise ValueError(f"cache has examples with unknown hints: {unknown_hints[:5]}")
    row_widths = np.diff(cache.offsets)
    expected_widths = hints.map(
        lambda hint: len(cache.candidates["hints"][hint])
    ).to_numpy()
    if not np.array_equal(row_widths, expected_widths):
        raise ValueError("cache score rows do not align with hint candidate lists")


def validate_lexicon_alignment(
    cache: ScoreCache, lexicon: hml.TrainingLexicon
) -> None:
    cached_frequencies = {
        word: int(count)
        for word, count in cache.candidates["word_frequencies"].items()
    }
    rebuilt_frequencies = {
        word: int(count) for word, count in lexicon.word_counts.items()
    }
    if cached_frequencies != rebuilt_frequencies:
        raise ValueError("rebuilt word frequencies do not match score-cache candidates")

    cached_roots = {
        word: int(root) for word, root in cache.candidates["word_roots"].items()
    }
    rebuilt_roots = {
        word: int(token_ids[0])
        for word, token_ids in lexicon.word_bpe.items()
        if token_ids
    }
    if cached_roots != rebuilt_roots:
        raise ValueError("rebuilt tokenizer roots do not match score-cache candidates")

    for hint, candidates in cache.candidates["hints"].items():
        cached = [
            (
                int(candidate["token_id"]),
                str(candidate["word"]),
                int(candidate["frequency"]),
                int(candidate["bpe_length"]),
            )
            for candidate in candidates
        ]
        rebuilt = [
            (
                int(token_id),
                lexicon.hint_token_word[hint][token_id],
                int(lexicon.word_counts[lexicon.hint_token_word[hint][token_id]]),
                len(lexicon.word_bpe[lexicon.hint_token_word[hint][token_id]]),
            )
            for token_id in lexicon.candidate_ids_by_hint[hint]
        ]
        if cached != rebuilt:
            raise ValueError(f"rebuilt candidates do not match cache for hint {hint!r}")


def _bool(value: object) -> bool:
    return str(value).casefold() in {"1", "true", "yes"}


def _root_stats(values: np.ndarray, positions: np.ndarray) -> tuple[float, float]:
    margin = (
        float(values[int(positions[0])] - values[int(positions[1])])
        if len(positions) > 1
        else 0.0
    )
    return margin, entropy_from_log_probs(values)


def build_candidate_pool(
    cache: ScoreCache,
    lexicon: hml.TrainingLexicon,
    *,
    roots_per_example: int = 10,
    words_per_root: int = 5,
) -> pd.DataFrame:
    """Build the full-word pool for alphanumeric hints without label-selected roots."""
    columns = [
        "example_id", "group_id", "context", "hint", "answer", "contamination",
        "baseline_prediction", "candidate", "candidate_correct", "root_id",
        "root_rank", "root_log_probability", "root_margin", "root_entropy",
        "within_root_rank", "word_count", "log_word_count",
        "root_candidate_count", "bpe_length", "char_length", "is_numeric",
        "is_alpha", "is_representative", "candidate_miss",
    ]
    records: list[tuple] = []
    answer_roots = {
        word: int(ids[0]) for word, ids in lexicon.word_bpe.items() if ids
    }
    for source_index, source in enumerate(cache.examples.to_dict("records")):
        hint = str(source["first letter"])
        if not hint.isalnum():
            continue
        values = cache.row(source_index)
        positions = ranked_positions(values, roots_per_example)
        margin, entropy = _root_stats(values, positions)
        choices = cache.candidates["hints"][hint]
        root_info = []
        for root_rank, position in enumerate(positions, 1):
            root_id = int(choices[int(position)]["token_id"])
            words = list(
                lexicon.hint_root_words.get(hint, {}).get(root_id, [])[
                    :words_per_root
                ]
            )
            root_info.append((root_rank, root_id, float(values[int(position)]), margin, entropy, words))
        if not root_info:
            continue
        baseline = str(lexicon.hint_token_word.get(hint, {}).get(root_info[0][1], ""))
        contamination = _bool(source.get("checkpoint_selection_contaminated", False))
        answer = str(source["answer"])
        answer_root = answer_roots.get(answer)
        selected_roots = {root_id for _rank, root_id, *_rest in root_info}
        selected_words = {
            word for _rank, _root_id, _score, _margin, _entropy, words in root_info
            for word in words
        }
        if answer in selected_words:
            candidate_miss = "reachable"
        elif answer_root is None:
            candidate_miss = "answer_unrepresented"
        elif answer_root not in selected_roots:
            candidate_miss = "root_outside_beam"
        else:
            candidate_miss = "word_outside_per_root_limit"
        for root_rank, root_id, root_logprob, margin, entropy, words in root_info:
            for within_rank, word in enumerate(words, 1):
                bpe = tuple(lexicon.word_bpe[word])
                word_count = int(lexicon.word_counts[word])
                records.append(
                    (
                        int(source["example_id"]), str(source["group_id"]),
                        str(source["context"]), hint, answer, contamination, baseline,
                        str(word), str(word) == answer, root_id, root_rank, root_logprob,
                        margin, entropy, within_rank, word_count,
                        math.log1p(word_count),
                        len(lexicon.hint_root_words.get(hint, {}).get(root_id, [])),
                        len(bpe), len(word), word.isnumeric(), word.isalpha(),
                        within_rank == 1, candidate_miss,
                    )
                )
    pool = pd.DataFrame.from_records(records, columns=columns)
    if pool.empty:
        raise ValueError("no alphanumeric candidates were produced")
    return pool.sort_values(["example_id", "root_rank", "within_root_rank", "candidate"]).reset_index(drop=True)


def deterministic_folds(group_ids: Iterable[str], folds: int) -> np.ndarray:
    if folds < 2:
        raise ValueError("folds must be at least two")
    return np.fromiter((grouped_fold(str(group), folds) for group in group_ids), dtype=np.int64)


class ScalarNormalizer:
    def __init__(self, columns: Sequence[str] = SCALAR_COLUMNS):
        self.columns = list(columns)
        self.mean = np.empty(0, dtype=np.float64)
        self.scale = np.empty(0, dtype=np.float64)

    def fit(self, frame: pd.DataFrame) -> "ScalarNormalizer":
        values = frame[self.columns].astype(float).to_numpy()
        if not np.isfinite(values).all():
            invalid = [
                column
                for index, column in enumerate(self.columns)
                if not np.isfinite(values[:, index]).all()
            ]
            raise ValueError(f"non-finite scalar features: {invalid}")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-8] = 1.0
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("non-finite scalar normalization statistics")
        self.mean = mean
        self.scale = scale
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not len(self.mean) or not len(self.scale):
            raise ValueError("normalizer has not been fitted")
        values = frame[self.columns].astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("non-finite scalar features during transform")
        return (values - self.mean) / self.scale


class SharedContextMLP(nn.Module):
    def __init__(self, embedding_dim: int, scalar_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 4 + scalar_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, context: torch.Tensor, candidate: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        if context.ndim == candidate.ndim - 1:
            context = context.unsqueeze(1).expand_as(candidate)
        shape = candidate.shape[:-1]
        joined = torch.cat(
            (context, candidate, context * candidate, torch.abs(context - candidate), scalar),
            dim=-1,
        )
        return self.network(joined.reshape(-1, joined.shape[-1])).reshape(*shape)


def listwise_batch_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = scores.masked_fill(~mask, -torch.inf)
    targets = labels.masked_fill(~mask, -torch.inf)
    target_index = targets.argmax(dim=1)
    valid = ((labels > 0) & mask).any(dim=1)
    return nn.functional.cross_entropy(masked[valid], target_index[valid]) if valid.any() else scores.sum() * 0.0


def deterministic_rank(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    result = frame[["example_id", "candidate", "candidate_correct", "root_rank", "within_root_rank"]].copy()
    result["score"] = np.asarray(scores, dtype=float)
    return result.sort_values(
        ["example_id", "score", "root_rank", "within_root_rank", "candidate"],
        ascending=[True, False, True, True, True], kind="mergesort",
    ).assign(rank=lambda x: x.groupby("example_id").cumcount() + 1)


def ranking_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    ranked = deterministic_rank(frame, scores)
    first_correct = ranked.loc[ranked.candidate_correct.astype(bool)].groupby("example_id")["rank"].min()
    examples = int(ranked.example_id.nunique())
    reachable = ranked.groupby("example_id").candidate_correct.any()
    return {
        "examples": examples,
        "candidate_oracle_coverage": float(reachable.mean()),
        "top1_accuracy": float((first_correct == 1).sum() / examples),
        "top5_accuracy": float((first_correct <= 5).sum() / examples),
        "mean_reciprocal_rank": float((1 / first_correct).sum() / examples),
    }


def grouped_bootstrap(frame: pd.DataFrame, *, samples: int, seed: int) -> dict:
    groups = frame.groupby("group_id")["correct"].agg(["sum", "size"])
    if not samples or len(groups) == 0:
        return {}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = groups.iloc[rng.integers(0, len(groups), size=len(groups))]
        draws[index] = sampled["sum"].sum() / sampled["size"].sum()
    return {"low": float(np.quantile(draws, .025)), "high": float(np.quantile(draws, .975))}


def paired_prediction_metrics(
    frame: pd.DataFrame,
    baseline_column: str,
    prediction_column: str,
    *,
    samples: int,
    seed: int,
) -> dict:
    baseline = frame[baseline_column].eq(frame["answer"])
    prediction = frame[prediction_column].eq(frame["answer"])
    result = {
        "examples": len(frame),
        "baseline_accuracy": float(baseline.mean()),
        "accuracy": float(prediction.mean()),
        "gain": float(prediction.mean() - baseline.mean()),
        "wrong_to_correct": int((~baseline & prediction).sum()),
        "correct_to_wrong": int((baseline & ~prediction).sum()),
    }
    if not samples:
        return result
    grouped = pd.DataFrame(
        {
            "group_id": frame["group_id"],
            "delta": prediction.astype(int) - baseline.astype(int),
        }
    ).groupby("group_id")["delta"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = grouped.iloc[rng.integers(0, len(grouped), size=len(grouped))]
        draws[index] = sampled["sum"].sum() / sampled["size"].sum()
    result["gain_ci_low"] = float(np.quantile(draws, 0.025))
    result["gain_ci_high"] = float(np.quantile(draws, 0.975))
    return result


def _encode_texts(
    texts: list[str], model_name: str, batch_size: int, max_length: int, device: str
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            output.append(pooled.cpu().numpy().astype(np.float16))
    return np.concatenate(output, axis=0) if output else np.empty((0, int(model.config.hidden_size)), dtype=np.float16)


def _model_manifest(model_name: str) -> dict:
    from huggingface_hub import snapshot_download

    model_path = Path(model_name)
    if not model_path.is_dir():
        model_path = Path(snapshot_download(model_name, local_files_only=True))
    files = {
        str(path.relative_to(model_path)): file_sha256(path)
        for path in sorted(model_path.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise ValueError(f"embedding model has no local files: {model_name}")
    return {"requested": model_name, "resolved_path": str(model_path), "files": files}


def _validate_prepare_args(args: argparse.Namespace) -> None:
    if args.roots < 1 or args.words < 1:
        raise ValueError("roots and words must be positive")
    if args.embedding_batch_size < 1 or args.max_embedding_tokens < 1:
        raise ValueError("embedding batch size and token limit must be positive")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")


def prepare(args: argparse.Namespace) -> None:
    _validate_prepare_args(args)
    reject_forbidden_path(args.dev_path)
    started = time.perf_counter()
    cache = ScoreCache("step258", args.cache)
    if int(cache.metadata.get("checkpoint_step", -1)) != 258000:
        raise ValueError("prepare requires the step258 GPT-2 score cache")
    validate_cache_provenance(cache, args.train_path, args.dev_path)
    model_source = cache.metadata.get("model_source")
    if not model_source:
        raise ValueError("cache metadata has no GPT tokenizer/model source")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True, local_files_only=True)
    lexicon = hml.build_training_lexicon(tokenizer, args.train_path)
    validate_lexicon_alignment(cache, lexicon)
    pool = build_candidate_pool(cache, lexicon, roots_per_example=args.roots, words_per_root=args.words)
    expected_example_ids = set(
        cache.examples.loc[
            cache.examples["first letter"].astype(str).map(str.isalnum), "example_id"
        ].astype(int)
    )
    if set(pool.example_id.astype(int)) != expected_example_ids:
        raise ValueError("candidate pool does not cover every alphanumeric-hint example")
    candidate_pool_seconds = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.output_dir / "candidate_pool.parquet"
    pool.to_parquet(pool_path, index=False)
    context_keys = sorted(pool.context.unique())
    word_keys = sorted(pool.candidate.unique())
    embedding_model_manifest = _model_manifest(args.embedding_model)
    embedding_started = time.perf_counter()
    context_embeddings = _encode_texts(
        context_keys,
        args.embedding_model,
        args.embedding_batch_size,
        args.max_embedding_tokens,
        args.device,
    )
    word_embeddings = _encode_texts(
        word_keys,
        args.embedding_model,
        args.embedding_batch_size,
        args.max_embedding_tokens,
        args.device,
    )
    if context_embeddings.shape[1:] != word_embeddings.shape[1:]:
        raise ValueError("context and candidate embedding dimensions disagree")
    if not np.isfinite(context_embeddings).all() or not np.isfinite(word_embeddings).all():
        raise ValueError("embedding model produced non-finite values")
    embedding_seconds = time.perf_counter() - embedding_started
    np.save(args.output_dir / "context_embeddings.npy", context_embeddings)
    np.save(args.output_dir / "candidate_embeddings.npy", word_embeddings)
    pd.DataFrame({"context_key": context_keys, "context_id": range(len(context_keys))}).to_csv(args.output_dir / "contexts.csv", index=False)
    pd.DataFrame({"candidate": word_keys, "candidate_id": range(len(word_keys))}).to_csv(args.output_dir / "words.csv", index=False)
    metadata = {
        "schema_version": SCHEMA_VERSION, "created_unix": time.time(),
        "cache_dir": str(args.cache.resolve()), "cache_metadata": cache.metadata,
        "cache_files_sha256": {name: file_sha256(args.cache / name) for name in ("metadata.json", "examples.csv", "candidates.json", "offsets.npy", "scores.npy")},
        "train_sha256": file_sha256(args.train_path), "dev_sha256": file_sha256(args.dev_path),
        "train_path": str(args.train_path.resolve()), "dev_path": str(args.dev_path.resolve()),
        "embedding_model": args.embedding_model,
        "embedding_model_manifest": embedding_model_manifest,
        "embedding_dim": int(context_embeddings.shape[1]),
        "max_embedding_tokens": args.max_embedding_tokens,
        "candidate_policy": {"roots": args.roots, "words_per_root": args.words, "alphanumeric_hints_only": True, "label_free_root_selection": True},
        "rows": len(pool), "examples": int(pool.example_id.nunique()),
        "elapsed_candidate_pool_seconds": candidate_pool_seconds,
        "elapsed_embedding_seconds": embedding_seconds,
        "elapsed_total_seconds": time.perf_counter() - started,
    }
    metadata["prepared_files_sha256"] = {
        name: file_sha256(args.output_dir / name)
        for name in PREPARED_FILES
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _load_prepared(directory: Path) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported prepared-pool schema")
    reject_forbidden_path(metadata.get("dev_path", ""))
    manifest = metadata.get("prepared_files_sha256")
    if not isinstance(manifest, dict) or set(manifest) != set(PREPARED_FILES):
        raise ValueError("prepared artifact manifest is incomplete")
    for name, expected in manifest.items():
        path = directory / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"prepared artifact hash mismatch for {name}")
    pool = pd.read_parquet(directory / "candidate_pool.parquet")
    missing = REQUIRED_POOL_COLUMNS - set(pool.columns)
    if missing:
        raise ValueError(f"prepared pool is missing columns: {sorted(missing)}")
    if len(pool) != metadata.get("rows") or pool.example_id.nunique() != metadata.get(
        "examples"
    ):
        raise ValueError("prepared pool dimensions do not match metadata")
    if pool[list(REQUIRED_POOL_COLUMNS)].isna().any().any():
        raise ValueError("prepared pool has missing required values")
    contexts = pd.read_csv(
        directory / "contexts.csv",
        dtype={"context_key": str, "context_id": np.int64},
        keep_default_na=False,
    )
    words = pd.read_csv(
        directory / "words.csv",
        dtype={"candidate": str, "candidate_id": np.int64},
        keep_default_na=False,
    )
    context_embeddings = np.load(directory / "context_embeddings.npy", mmap_mode="r")
    word_embeddings = np.load(directory / "candidate_embeddings.npy", mmap_mode="r")
    if len(contexts) != len(context_embeddings) or len(words) != len(word_embeddings):
        raise ValueError("embedding mapping and array lengths disagree")
    if (
        contexts.context_key.duplicated().any()
        or words.candidate.duplicated().any()
        or not np.array_equal(contexts.context_id.to_numpy(), np.arange(len(contexts)))
        or not np.array_equal(words.candidate_id.to_numpy(), np.arange(len(words)))
    ):
        raise ValueError("embedding mapping IDs are not a contiguous aligned range")
    expected_dimension = metadata.get("embedding_dim")
    if (
        context_embeddings.ndim != 2
        or word_embeddings.ndim != 2
        or context_embeddings.shape[1] != expected_dimension
        or word_embeddings.shape[1] != expected_dimension
        or context_embeddings.dtype.kind != "f"
        or word_embeddings.dtype.kind != "f"
        or not np.isfinite(context_embeddings).all()
        or not np.isfinite(word_embeddings).all()
    ):
        raise ValueError("embedding arrays have invalid shape, dtype, or values")
    cache_manifest = metadata.get("cache_files_sha256")
    if not isinstance(cache_manifest, dict) or not cache_manifest:
        raise ValueError("prepared cache manifest is missing")
    for name, expected in cache_manifest.items():
        actual_path = Path(metadata["cache_dir"]) / name
        if not actual_path.is_file() or file_sha256(actual_path) != expected:
            raise ValueError(f"prepared cache provenance mismatch for {name}")
    context_ids = dict(zip(contexts.context_key, contexts.context_id))
    word_ids = dict(zip(words.candidate, words.candidate_id))
    if any(value not in context_ids for value in pool.context) or any(value not in word_ids for value in pool.candidate):
        raise ValueError("pool contains text absent from embedding mappings")
    if pool.duplicated(["example_id", "candidate"]).any():
        raise ValueError("prepared pool has duplicate candidates")
    invariant_columns = [
        "group_id", "context", "hint", "answer", "contamination",
        "baseline_prediction", "candidate_miss",
    ]
    if pool.groupby("example_id")[invariant_columns].nunique(dropna=False).gt(1).any().any():
        raise ValueError("prepared pool has inconsistent rows within an example")
    expected_groups = np.asarray(
        [group_identifier(str(context), str(hint)) for context, hint in zip(pool.context, pool.hint)]
    )
    if not np.array_equal(pool.group_id.astype(str).to_numpy(), expected_groups):
        raise ValueError("prepared group IDs do not match context/hint groups")
    expected_labels = pool.candidate.astype(str).eq(pool.answer.astype(str))
    if not np.array_equal(pool.candidate_correct.astype(bool).to_numpy(), expected_labels.to_numpy()):
        raise ValueError("prepared candidate labels are inconsistent")
    scalar_values = pool[SCALAR_COLUMNS].astype(float).to_numpy()
    if not np.isfinite(scalar_values).all():
        raise ValueError("prepared pool has non-finite scalar features")
    pool["context_id"] = pool.context.map(context_ids).astype(int)
    pool["candidate_id"] = pool.candidate.map(word_ids).astype(int)
    return pool, metadata, np.asarray(context_embeddings), np.asarray(word_embeddings)


def _group_rows(frame: pd.DataFrame) -> list[tuple[int, np.ndarray]]:
    return [
        (int(example_id), np.asarray(indices, dtype=np.int64))
        for example_id, indices in frame.groupby("example_id", sort=True).groups.items()
    ]


def _padded_batches(
    grouped: Sequence[tuple[int, np.ndarray]],
    batch_examples: int,
    *,
    order: np.ndarray | None = None,
) -> Iterable[tuple[list[int], np.ndarray, np.ndarray]]:
    positions = np.arange(len(grouped)) if order is None else order
    for start in range(0, len(positions), batch_examples):
        groups = [grouped[int(position)] for position in positions[start:start + batch_examples]]
        width = max(len(indices) for _, indices in groups)
        indices = np.full((len(groups), width), -1, dtype=np.int64)
        mask = np.zeros((len(groups), width), dtype=bool)
        for row, (_example, group_indices) in enumerate(groups):
            indices[row, :len(group_indices)] = group_indices
            mask[row, :len(group_indices)] = True
        yield [int(example) for example, _ in groups], indices, mask


def _save_json(value: object, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _batch_tensors(
    frame: pd.DataFrame,
    indices: np.ndarray,
    mask: np.ndarray,
    context_embeddings: np.ndarray,
    word_embeddings: np.ndarray,
    normalizer: ScalarNormalizer,
    device: torch.device,
    *,
    include_labels: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    flat = indices[mask]
    first = indices[:, 0]
    context_ids = frame.loc[first, "context_id"].to_numpy(dtype=np.int64)
    context = torch.from_numpy(
        np.asarray(context_embeddings[context_ids], dtype=np.float32)
    ).to(device)
    candidate = torch.zeros(
        (mask.shape[0], mask.shape[1], word_embeddings.shape[1]), device=device
    )
    scalars = torch.zeros(
        (mask.shape[0], mask.shape[1], len(SCALAR_COLUMNS)), device=device
    )
    valid = torch.from_numpy(mask).to(device)
    candidate_ids = frame.loc[flat, "candidate_id"].to_numpy(dtype=np.int64)
    candidate[valid] = torch.from_numpy(
        np.asarray(word_embeddings[candidate_ids], dtype=np.float32)
    ).to(device)
    scalars[valid] = torch.from_numpy(
        normalizer.transform(frame.loc[flat]).astype(np.float32)
    ).to(device)
    labels = None
    if include_labels:
        labels = torch.zeros(mask.shape, device=device)
        labels[valid] = torch.from_numpy(
            frame.loc[flat, "candidate_correct"].astype(float).to_numpy(np.float32)
        ).to(device)
    return context, candidate, scalars, labels, valid


def _validate_train_args(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("folds must be at least two")
    if args.epochs < 1 or args.batch_size < 1 or args.hidden < 2:
        raise ValueError("epochs, batch size, and hidden width must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")


def train(args: argparse.Namespace) -> None:
    _validate_train_args(args)
    started = time.perf_counter()
    pool, metadata, context_embeddings, word_embeddings = _load_prepared(args.input_dir)
    if file_sha256(Path(metadata["train_path"])) != metadata["train_sha256"] or file_sha256(Path(metadata["dev_path"])) != metadata["dev_sha256"]:
        raise ValueError("prepared provenance hashes no longer match")
    contaminated = pool.groupby("group_id").contamination.transform("any")
    pool = pool.loc[~contaminated].copy().reset_index(drop=True)
    if pool.empty:
        raise ValueError("no uncontaminated examples remain")
    pool["fold"] = deterministic_folds(pool.group_id, args.folds)
    if pool.groupby("group_id")["fold"].nunique().gt(1).any():
        raise RuntimeError("a context/hint group crosses OOF folds")
    reachable_by_example = pool.groupby("example_id").candidate_correct.transform("any")
    example_folds = pool.drop_duplicates("example_id")["fold"].value_counts()
    missing_folds = sorted(set(range(args.folds)) - set(example_folds.index))
    if missing_folds:
        raise ValueError(f"folds have no validation examples: {missing_folds}")
    for fold in range(args.folds):
        training_examples = pool.loc[
            pool.fold.ne(fold) & reachable_by_example, "example_id"
        ].nunique()
        if not training_examples:
            raise ValueError(f"fold {fold} has no reachable training examples")
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    all_scores = np.full(len(pool), np.nan, dtype=np.float32)
    fold_stats = []
    total_training_seconds = 0.0
    total_inference_seconds = 0.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(args.folds):
        torch.manual_seed(args.seed + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + fold)
        train_frame = pool.loc[pool.fold.ne(fold) & reachable_by_example].copy()
        valid_frame = pool.loc[pool.fold.eq(fold)].copy()
        if train_frame.empty or not train_frame.candidate_correct.any():
            raise ValueError(f"fold {fold} has no reachable training examples")
        normalizer = ScalarNormalizer().fit(train_frame)
        model = SharedContextMLP(context_embeddings.shape[1], len(SCALAR_COLUMNS), args.hidden, args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        train_groups = _group_rows(train_frame)
        valid_groups = _group_rows(valid_frame)
        epoch_losses = []
        training_started = time.perf_counter()
        for epoch in range(args.epochs):
            model.train()
            rng = np.random.default_rng(args.seed + fold * 10_000 + epoch)
            order = rng.permutation(len(train_groups))
            loss_total = 0.0
            batches = 0
            for _examples, indices, mask in _padded_batches(
                train_groups, args.batch_size, order=order
            ):
                context, candidate, scalars, labels, valid = _batch_tensors(
                    train_frame,
                    indices,
                    mask,
                    context_embeddings,
                    word_embeddings,
                    normalizer,
                    device,
                    include_labels=True,
                )
                assert labels is not None
                loss = listwise_batch_loss(
                    model(context, candidate, scalars), labels, valid
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_total += float(loss.detach())
                batches += 1
            epoch_loss = loss_total / max(1, batches)
            epoch_losses.append(epoch_loss)
            print(
                f"Fold {fold + 1}/{args.folds}, epoch {epoch + 1}/{args.epochs}: "
                f"loss={epoch_loss:.6f}",
                flush=True,
            )
        training_seconds = time.perf_counter() - training_started
        total_training_seconds += training_seconds
        model.eval()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            for _examples, indices, mask in _padded_batches(
                valid_groups, args.batch_size
            ):
                flat = indices[mask]
                context, candidate, scalars, _labels, valid = _batch_tensors(
                    valid_frame,
                    indices,
                    mask,
                    context_embeddings,
                    word_embeddings,
                    normalizer,
                    device,
                    include_labels=False,
                )
                batch_scores = model(context, candidate, scalars).cpu().numpy()
                all_scores[flat] = batch_scores[mask]
        inference_seconds = time.perf_counter() - inference_started
        total_inference_seconds += inference_seconds
        model_path = args.output_dir / f"fold_{fold}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "normalizer_mean": normalizer.mean,
                "normalizer_scale": normalizer.scale,
                "scalar_columns": SCALAR_COLUMNS,
                "embedding_dim": int(context_embeddings.shape[1]),
                "hidden": args.hidden,
                "dropout": args.dropout,
            },
            model_path,
        )
        fold_stats.append(
            {
                "fold": fold,
                "train_examples": len(train_groups),
                "valid_examples": len(valid_groups),
                "train_rows": len(train_frame),
                "valid_rows": len(valid_frame),
                "reachable_valid_examples": int(
                    valid_frame.groupby("example_id").candidate_correct.any().sum()
                ),
                "epoch_losses": epoch_losses,
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                "normalizer_mean": normalizer.mean.tolist(),
                "normalizer_scale": normalizer.scale.tolist(),
                "model_sha256": file_sha256(model_path),
            }
        )
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not np.isfinite(all_scores).all():
        raise RuntimeError("OOF predictions are incomplete")
    ranked = deterministic_rank(pool, all_scores)
    per_example = pool.loc[pool.groupby("example_id").head(1).index, ["example_id", "group_id", "answer", "baseline_prediction"]].copy()
    top = ranked.drop_duplicates("example_id").set_index("example_id")
    per_example["prediction"] = per_example.example_id.map(top.candidate)
    per_example["correct"] = per_example.prediction.eq(per_example.answer)
    per_example["fold"] = per_example.example_id.map(pool.drop_duplicates("example_id").set_index("example_id").fold)
    metrics = {
        "evaluation_scope": "grouped out-of-fold development evidence, not untouched holdout",
        "candidate_ranking": ranking_metrics(pool, all_scores),
        "candidate_miss_reasons": {},
        "top1": {
            "accuracy": float(per_example.correct.mean()),
            "grouped_bootstrap_ci": grouped_bootstrap(
                per_example, samples=args.bootstrap_samples, seed=args.seed
            ),
        },
    }
    miss = pool.drop_duplicates("example_id").set_index("example_id")["candidate_miss"]
    metrics["candidate_miss_reasons"] = {
        str(name): int(count) for name, count in miss.value_counts().items()
    }
    metrics["comparisons"] = {
        "step258_root": paired_prediction_metrics(
            per_example,
            "baseline_prediction",
            "prediction",
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        **compare_existing(
            per_example,
            args.existing_fusion,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    metrics["timing_seconds"] = {
        "candidate_pool_preparation": metadata.get("elapsed_candidate_pool_seconds"),
        "embedding": metadata.get("elapsed_embedding_seconds"),
        "oof_training": total_training_seconds,
        "oof_inference": total_inference_seconds,
        "train_command_total": time.perf_counter() - started,
    }
    metrics["peak_memory_mib"] = {
        "process_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "cuda_allocated": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
        "cuda_reserved": (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
    }
    per_example.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    ranked.to_parquet(args.output_dir / "oof_candidate_scores.parquet", index=False)
    _save_json(
        {
            "metadata": metadata,
            "folds": fold_stats,
            "scalar_columns": SCALAR_COLUMNS,
            "model": (
                "shared-context MLP over frozen MiniLM mean-pooled embeddings; "
                "not a cross-encoder"
            ),
            "seed": args.seed,
            "parameters": {
                "folds": args.folds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden": args.hidden,
                "dropout": args.dropout,
                "learning_rate": args.learning_rate,
            },
            "environment": environment_provenance(),
        },
        args.output_dir / "provenance.json",
    )
    _save_json(metrics, args.output_dir / "metrics.json")
    (args.output_dir / "report.md").write_text(
        markdown_report(metrics, metadata), encoding="utf-8"
    )
    _save_json(
        {
            path.name: file_sha256(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "artifact_manifest.json"
        },
        args.output_dir / "artifact_manifest.json",
    )


def blend_candidate_scores(
    left: pd.DataFrame,
    right: pd.DataFrame,
    existing: pd.DataFrame,
    *,
    right_weight: float,
    fusion_bonus: float,
    control_bonus: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    identity = [
        "example_id", "candidate", "candidate_correct", "root_rank",
        "within_root_rank",
    ]
    left = left.sort_values(["example_id", "candidate"], kind="stable").reset_index(
        drop=True
    )
    right = right.sort_values(
        ["example_id", "candidate"], kind="stable"
    ).reset_index(drop=True)
    if left.duplicated(["example_id", "candidate"]).any() or right.duplicated(
        ["example_id", "candidate"]
    ).any():
        raise ValueError("ensemble inputs contain duplicate candidates")
    if not left[identity].equals(right[identity]):
        raise ValueError("ensemble candidate rows are not identically aligned")
    if existing.example_id.duplicated().any():
        raise ValueError("existing predictions contain duplicate example IDs")
    existing = existing.set_index("example_id")
    example_ids = left.example_id
    if not set(example_ids.unique()).issubset(existing.index):
        raise ValueError("existing predictions do not cover ensemble examples")
    fusion_predictions = example_ids.map(existing.treatment_prediction)
    control_predictions = example_ids.map(existing.control_prediction)
    scores = (
        (1.0 - right_weight) * left.score.to_numpy(dtype=np.float64)
        + right_weight * right.score.to_numpy(dtype=np.float64)
        + fusion_bonus
        * left.candidate.eq(fusion_predictions).to_numpy(dtype=np.float64)
        + control_bonus
        * left.candidate.eq(control_predictions).to_numpy(dtype=np.float64)
    )
    if not np.isfinite(scores).all():
        raise ValueError("ensemble produced non-finite candidate scores")
    return left, scores


def ensemble(args: argparse.Namespace) -> None:
    if not 0.0 <= args.right_weight <= 1.0:
        raise ValueError("right model weight must be in [0, 1]")
    if args.fusion_bonus < 0.0 or args.control_bonus < 0.0:
        raise ValueError("prediction bonuses must be non-negative")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    started = time.perf_counter()
    score_name = "oof_candidate_scores.parquet"
    prediction_name = "oof_predictions.csv"
    left_path = args.left_run / score_name
    right_path = args.right_run / score_name
    left = pd.read_parquet(left_path)
    right = pd.read_parquet(right_path)
    existing = pd.read_csv(args.existing_fusion, keep_default_na=False)
    for column in (
        "example_id", "treatment_prediction", "control_prediction"
    ):
        if column not in existing:
            raise ValueError(f"existing predictions are missing {column!r}")
    candidates, scores = blend_candidate_scores(
        left,
        right,
        existing,
        right_weight=args.right_weight,
        fusion_bonus=args.fusion_bonus,
        control_bonus=args.control_bonus,
    )
    ranked = deterministic_rank(candidates, scores)
    per_example = pd.read_csv(
        args.left_run / prediction_name, keep_default_na=False
    )[["example_id", "group_id", "answer", "baseline_prediction", "fold"]]
    if per_example.example_id.duplicated().any():
        raise ValueError("left-run predictions contain duplicate example IDs")
    right_predictions = pd.read_csv(
        args.right_run / prediction_name, keep_default_na=False
    )[["example_id", "group_id", "answer", "prediction"]].rename(
        columns={"prediction": "right_prediction"}
    )
    per_example = per_example.merge(
        right_predictions,
        on=["example_id", "group_id", "answer"],
        how="left",
        validate="one_to_one",
    )
    left_predictions = pd.read_csv(
        args.left_run / prediction_name, keep_default_na=False
    )[["example_id", "prediction"]].rename(
        columns={"prediction": "left_prediction"}
    )
    per_example = per_example.merge(
        left_predictions, on="example_id", how="left", validate="one_to_one"
    )
    top = ranked.drop_duplicates("example_id").set_index("example_id")
    per_example["prediction"] = per_example.example_id.map(top.candidate)
    if per_example[["left_prediction", "right_prediction", "prediction"]].isna().any().any():
        raise ValueError("ensemble predictions are incomplete")
    per_example["correct"] = per_example.prediction.eq(per_example.answer)
    ranking = ranking_metrics(candidates, scores)
    comparisons = {
        "step258_root": paired_prediction_metrics(
            per_example,
            "baseline_prediction",
            "prediction",
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "left_neural_model": paired_prediction_metrics(
            per_example,
            "left_prediction",
            "prediction",
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "right_neural_model": paired_prediction_metrics(
            per_example,
            "right_prediction",
            "prediction",
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        **compare_existing(
            per_example,
            args.existing_fusion,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    source_metrics = json.loads(
        (args.left_run / "metrics.json").read_text(encoding="utf-8")
    )
    metrics = {
        "evaluation_scope": (
            "grouped out-of-fold development tuning evidence; ensemble weights and "
            "bonuses were selected on these labels, not an untouched holdout"
        ),
        "candidate_ranking": ranking,
        "candidate_miss_reasons": source_metrics["candidate_miss_reasons"],
        "top1": {
            "accuracy": float(per_example.correct.mean()),
            "grouped_bootstrap_ci": grouped_bootstrap(
                per_example, samples=args.bootstrap_samples, seed=args.seed
            ),
        },
        "comparisons": comparisons,
        "timing_seconds": {"ensemble_command_total": time.perf_counter() - started},
        "peak_memory_mib": {
            "process_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_example.to_csv(args.output_dir / prediction_name, index=False)
    ranked.to_parquet(args.output_dir / score_name, index=False)
    provenance = {
        "method": (
            "weighted OOF neural-logit blend with bonuses for existing checkpoint-"
            "fusion and full-word-control predictions"
        ),
        "development_tuned": True,
        "parameters": {
            "right_weight": args.right_weight,
            "fusion_bonus": args.fusion_bonus,
            "control_bonus": args.control_bonus,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "sources": {
            "left_run": str(args.left_run.resolve()),
            "right_run": str(args.right_run.resolve()),
            "existing_fusion": str(args.existing_fusion.resolve()),
        },
        "source_sha256": {
            "left_scores": file_sha256(left_path),
            "left_predictions": file_sha256(args.left_run / prediction_name),
            "left_provenance": file_sha256(args.left_run / "provenance.json"),
            "right_scores": file_sha256(right_path),
            "right_predictions": file_sha256(args.right_run / prediction_name),
            "right_provenance": file_sha256(args.right_run / "provenance.json"),
            "existing_fusion": file_sha256(args.existing_fusion),
        },
        "environment": environment_provenance(),
    }
    _save_json(provenance, args.output_dir / "provenance.json")
    _save_json(metrics, args.output_dir / "metrics.json")
    lines = [
        "# Tuned Neural Reranker Ensemble",
        "",
        "This is grouped OOF development tuning evidence, not an untouched holdout estimate.",
        "The blend parameters were selected using these development labels.",
        "",
        "## Method",
        "",
        f"- Left neural-logit weight: {1.0 - args.right_weight:.3f}",
        f"- Right neural-logit weight: {args.right_weight:.3f}",
        f"- Checkpoint-fusion prediction bonus: {args.fusion_bonus:.3f}",
        f"- Full-word-control prediction bonus: {args.control_bonus:.3f}",
        "",
        "## Results",
        "",
        f"- Top-1 accuracy: **{metrics['top1']['accuracy']:.4%}**",
        f"- Top-5 accuracy: {ranking['top5_accuracy']:.4%}",
        f"- Candidate oracle coverage: {ranking['candidate_oracle_coverage']:.4%}",
        f"- MRR: {ranking['mean_reciprocal_rank']:.5f}",
        "",
        "## Paired comparisons",
        "",
    ]
    for name, comparison in comparisons.items():
        if name == "identical_examples":
            continue
        lines.append(
            f"- **{name}:** {comparison['accuracy']:.4%} ensemble vs "
            f"{comparison['baseline_accuracy']:.4%} baseline "
            f"({comparison['gain']:+.4%}; 95% CI "
            f"[{comparison['gain_ci_low']:+.4%}, "
            f"{comparison['gain_ci_high']:+.4%}])."
        )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    _save_json(
        {
            path.name: file_sha256(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "artifact_manifest.json"
        },
        args.output_dir / "artifact_manifest.json",
    )


def compare_existing(
    per_example: pd.DataFrame,
    fusion_path: Path | None,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    result = {}
    if fusion_path is None or not fusion_path.is_file():
        return result
    existing = pd.read_csv(fusion_path, keep_default_na=False)
    if "example_id" not in existing or "treatment_prediction" not in existing:
        return result
    if per_example.example_id.duplicated().any() or existing.example_id.duplicated().any():
        raise ValueError("comparison predictions have duplicate example IDs")
    columns = [
        column for column in (
            "treatment_prediction", "root_prediction_step258", "control_prediction"
        ) if column in existing
    ]
    identity_columns = ["example_id", "group_id", "answer"]
    if any(column not in existing for column in identity_columns):
        raise ValueError("existing comparisons lack group/answer identity columns")
    joined = per_example.merge(
        existing[[*identity_columns, *columns]],
        on="example_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_existing"),
    )
    if len(joined) != len(per_example):
        raise ValueError("existing comparison predictions do not cover identical examples")
    if joined[columns].isna().any().any():
        raise ValueError("existing comparison predictions do not cover identical examples")
    if not joined.group_id.eq(joined.group_id_existing).all() or not joined.answer.eq(
        joined.answer_existing
    ).all():
        raise ValueError("existing comparison example identities do not agree")
    result["identical_examples"] = len(joined)
    for column, name in (
        ("treatment_prediction", "checkpoint_fusion"),
        ("root_prediction_step258", "existing_root_step258"),
        ("control_prediction", "existing_full_word_control"),
    ):
        if column in joined:
            result[name] = paired_prediction_metrics(
                joined,
                column,
                "prediction",
                samples=bootstrap_samples,
                seed=seed,
            )
    return result


def markdown_report(metrics: dict, metadata: dict) -> str:
    ranking = metrics["candidate_ranking"]
    root = metrics["comparisons"]["step258_root"]
    lines = [
        "# Neural Candidate Reranker Experiment",
        "",
        "This is grouped out-of-fold development evidence, not an untouched holdout estimate.",
        "The model is a shared-context listwise MLP over frozen MiniLM embeddings and",
        "GPT-2/lexical scalar features; it is not a cross-encoder.",
        "",
        "## Configuration",
        "",
        f"- Generator checkpoint: step {metadata['cache_metadata']['checkpoint_step']:,}",
        f"- Root beam: {metadata['candidate_policy']['roots']}",
        f"- Words per root: {metadata['candidate_policy']['words_per_root']}",
        f"- Evaluated examples: {ranking['examples']:,} alphanumeric hints",
        "- Checkpoint-selection-contaminated groups excluded",
        "",
        "## Results",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Step-258 root-only top-1 | {root['baseline_accuracy']:.2%} |",
        f"| Neural reranker top-1 | **{root['accuracy']:.2%}** |",
        f"| Gain over step-258 | {root['gain']:+.3%} |",
        f"| Candidate oracle coverage | {ranking['candidate_oracle_coverage']:.2%} |",
        f"| Neural top-5 | {ranking['top5_accuracy']:.2%} |",
        f"| Mean reciprocal rank | {ranking['mean_reciprocal_rank']:.4f} |",
        "",
        "## Paired comparisons",
        "",
    ]
    for name, comparison in metrics["comparisons"].items():
        if name == "identical_examples":
            continue
        lines.append(
            f"- **{name}:** {comparison['accuracy']:.2%} neural vs "
            f"{comparison['baseline_accuracy']:.2%} baseline "
            f"({comparison['gain']:+.3%}; 95% CI "
            f"[{comparison.get('gain_ci_low', 0):+.3%}, "
            f"{comparison.get('gain_ci_high', 0):+.3%}])."
        )
    lines.extend(
        [
            "",
            "## Candidate misses",
            "",
            *[
                f"- {name}: {count:,}"
                for name, count in metrics["candidate_miss_reasons"].items()
            ],
            "",
            "## Timing",
            "",
            *[
                f"- {name}: {value:.2f} seconds"
                for name, value in metrics["timing_seconds"].items()
                if value is not None
            ],
            "",
        ]
    )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    prep.add_argument("--train-path", type=Path, default=TRAIN_DIR / "train.src.tok")
    prep.add_argument("--dev-path", type=Path, default=DATA_DIR / "devv_eval.csv")
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--embedding-model", default=DEFAULT_MODEL)
    prep.add_argument("--embedding-batch-size", type=int, default=128)
    prep.add_argument("--max-embedding-tokens", type=int, default=256)
    prep.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    prep.add_argument("--roots", type=int, default=10)
    prep.add_argument("--words", type=int, default=5)
    tr = sub.add_parser("train")
    tr.add_argument("--input-dir", type=Path, required=True)
    tr.add_argument("--output-dir", type=Path, required=True)
    tr.add_argument(
        "--existing-fusion",
        type=Path,
        default=ARTIFACTS_DIR
        / "gpt2-keyboard-script/rerank/checkpoint-fusion/grouped_oof_predictions.csv",
    )
    tr.add_argument("--folds", type=int, default=5)
    tr.add_argument("--epochs", type=int, default=5)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--hidden", type=int, default=256)
    tr.add_argument("--dropout", type=float, default=0.1)
    tr.add_argument("--learning-rate", type=float, default=1e-3)
    tr.add_argument("--bootstrap-samples", type=int, default=1_000)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ens = sub.add_parser("ensemble")
    ens.add_argument("--left-run", type=Path, required=True)
    ens.add_argument("--right-run", type=Path, required=True)
    ens.add_argument("--output-dir", type=Path, required=True)
    ens.add_argument(
        "--existing-fusion",
        type=Path,
        default=ARTIFACTS_DIR
        / "gpt2-keyboard-script/rerank/checkpoint-fusion/grouped_oof_predictions.csv",
    )
    ens.add_argument("--right-weight", type=float, default=0.55)
    ens.add_argument("--fusion-bonus", type=float, default=0.275)
    ens.add_argument("--control-bonus", type=float, default=0.325)
    ens.add_argument("--bootstrap-samples", type=int, default=1_000)
    ens.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    parsed = parser().parse_args(argv)
    {"prepare": prepare, "train": train, "ensemble": ensemble}[parsed.command](parsed)


if __name__ == "__main__":
    main()
