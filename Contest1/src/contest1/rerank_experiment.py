#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pickle
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import hint_masked_lib as hml
from .data_policy import reject_forbidden_evaluation_path
from .paths import ARTIFACTS_DIR, DATA_DIR, TRAIN_DIR
from .train_hint_masked import atomic_text_write, validate_dev_frame


SCORER_VERSION = 1
FINITE_SCORE_SENTINEL = 1_000_000.0
FEATURE_COLUMNS = [
    "root_log_probability",
    "root_rank",
    "within_root_rank",
    "root_margin",
    "root_entropy",
    "suffix_log_probability",
    "suffix_mean_log_probability",
    "suffix_token_count",
    "boundary_log_probability",
    "log_word_count",
    "root_candidate_count",
    "is_representative",
    "is_numeric",
    "is_alpha",
]
CHEAP_FEATURE_COLUMNS = [
    "root_log_probability",
    "root_rank",
    "within_root_rank",
    "root_margin",
    "root_entropy",
    "log_word_count",
    "root_candidate_count",
    "is_representative",
    "is_numeric",
    "is_alpha",
]
SUFFIX_FEATURE_COLUMNS = [
    column for column in FEATURE_COLUMNS if column != "boundary_log_probability"
]
REQUIRED_CACHE_COLUMNS = set(FEATURE_COLUMNS) | {
    "example_id",
    "group_id",
    "split",
    "answer",
    "baseline_prediction",
    "candidate",
    "candidate_correct",
}
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_identifier(context: str, hint: str) -> str:
    value = "\x1f".join((context, hint)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:20]


def reject_forbidden_dev(path: Path | str) -> None:
    reject_forbidden_evaluation_path(path)


def assign_grouped_splits(
    frame: pd.DataFrame,
    *,
    selection_size: int = 2_000,
    tune_fraction: float = 0.2,
    tune_train_fraction: float = 0.75,
    seed: int = 42,
) -> pd.DataFrame:
    if not 0 < tune_fraction < 1 or not 0 < tune_train_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if selection_size < 0 or selection_size > len(frame):
        raise ValueError("selection_size is outside the frame")

    split_frame = frame.copy()
    split_frame["example_id"] = np.arange(len(split_frame), dtype=np.int64)
    split_frame["group_id"] = [
        group_identifier(context, hint)
        for context, hint in zip(
            split_frame["context"],
            split_frame["first letter"],
        )
    ]
    selected = (
        split_frame.sample(n=selection_size, random_state=seed).index
        if selection_size
        else pd.Index([])
    )
    contaminated_groups = set(split_frame.loc[selected, "group_id"])
    split_frame["split"] = "locked"
    split_frame.loc[
        split_frame["group_id"].isin(list(contaminated_groups)), "split"
    ] = "contaminated"

    available = split_frame.loc[split_frame["split"] == "locked"]
    group_rows = (
        available.groupby(["group_id", "first letter"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
    )
    rng = np.random.default_rng(seed)
    tune_groups: set[str] = set()
    for _hint, hint_groups in group_rows.groupby("first letter", sort=True):
        order = rng.permutation(len(hint_groups))
        target_rows = max(1, round(int(hint_groups["rows"].sum()) * tune_fraction))
        accumulated = 0
        for position in order:
            row = hint_groups.iloc[int(position)]
            if accumulated >= target_rows:
                break
            tune_groups.add(str(row["group_id"]))
            accumulated += int(row["rows"])

    tune_group_list = sorted(tune_groups)
    rng.shuffle(tune_group_list)
    train_count = round(len(tune_group_list) * tune_train_fraction)
    tune_train_groups = set(tune_group_list[:train_count])
    tune_valid_groups = set(tune_group_list[train_count:])
    split_frame.loc[split_frame["group_id"].isin(list(tune_train_groups)), "split"] = (
        "tune_train"
    )
    split_frame.loc[split_frame["group_id"].isin(list(tune_valid_groups)), "split"] = (
        "tune_valid"
    )
    return split_frame


def load_model_and_lexicon(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    lexicon = hml.build_training_lexicon(
        tokenizer, args.train_path, max_lines=args.max_train_lines
    )
    return device, tokenizer, model, checkpoint, lexicon


def load_cache(
    path: Path,
    *,
    required_words_per_root: int | None = None,
    require_boundary: bool = False,
) -> tuple[pd.DataFrame, dict]:
    metadata_path = path.with_suffix(".meta.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Candidate cache or metadata is missing for {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dev_path"):
        reject_forbidden_dev(metadata["dev_path"])
    if metadata.get("scorer_version") != SCORER_VERSION:
        raise ValueError("candidate cache scorer version does not match this code")
    if (
        required_words_per_root is not None
        and required_words_per_root > int(metadata["words_per_root"])
    ):
        raise ValueError("requested words-per-root exceeds the cached candidate pool")
    if require_boundary and not metadata.get("boundary_score"):
        raise ValueError("non-zero boundary weights require a boundary-scored cache")
    candidates = pd.read_csv(path, keep_default_na=False, na_values=[""])
    missing = REQUIRED_CACHE_COLUMNS.difference(candidates.columns)
    if missing:
        raise ValueError(f"candidate cache is missing columns: {sorted(missing)}")
    for column in ("baseline_correct", "candidate_correct", "is_representative"):
        if column in candidates and candidates[column].dtype == object:
            candidates[column] = candidates[column].map(
                {"True": True, "False": False, "true": True, "false": False}
            )
    return candidates, metadata


def build_cache(args: argparse.Namespace) -> None:
    reject_forbidden_dev(args.dev_path)
    for path in (args.checkpoint, args.train_path, args.dev_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.is_file() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {args.output}; pass --overwrite")

    dev = pd.read_csv(args.dev_path, keep_default_na=False)
    validate_dev_frame(dev)
    run_config_path = args.checkpoint.parent / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(
            f"Cannot verify checkpoint-selection split without {run_config_path}"
        )
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    expected_selection = int(run_config.get("eval_subset_size") or len(dev))
    expected_seed = int(run_config.get("seed", -1))
    if args.selection_size != expected_selection or args.seed != expected_seed:
        raise ValueError(
            "selection-size/seed do not match the checkpoint run configuration"
        )
    split_frame = assign_grouped_splits(
        dev,
        selection_size=args.selection_size,
        tune_fraction=args.tune_fraction,
        tune_train_fraction=args.tune_train_fraction,
        seed=args.seed,
    )
    device, tokenizer, model, checkpoint, lexicon = load_model_and_lexicon(args)
    boundary_ids = None
    if args.boundary_score:
        boundary_ids = hml.build_boundary_token_ids(tokenizer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    wrote_header = False
    started = time.perf_counter()
    for start in range(0, len(split_frame), args.chunk_size):
        chunk = split_frame.iloc[start : start + args.chunk_size]
        scored_rows = hml.score_candidates(
            model,
            tokenizer,
            chunk["context"].tolist(),
            chunk["first letter"].tolist(),
            candidate_ids_by_hint=lexicon.candidate_ids_by_hint,
            hint_root_words=lexicon.hint_root_words,
            word_bpe=lexicon.word_bpe,
            word_counts=lexicon.word_counts,
            words_per_root=args.words_per_root,
            root_beam=args.root_beam,
            mode="cross_root",
            batch_size=args.eval_batch_size,
            candidate_batch_size=args.candidate_batch_size,
            max_context_tokens=args.max_context_tokens,
            boundary_token_ids=boundary_ids,
        )
        if any(not candidates for candidates in scored_rows):
            missing_rows = [
                int(example_id)
                for example_id, candidates in zip(chunk["example_id"], scored_rows)
                if not candidates
            ]
            raise ValueError(f"candidate generation failed for examples {missing_rows[:10]}")
        records = []
        for source, candidates in zip(chunk.to_dict(orient="records"), scored_rows):
            hint = str(source["first letter"])
            answer = str(source["answer"])
            baseline = next(
                (
                    candidate.word
                    for candidate in candidates
                    if candidate.root_rank == 1 and candidate.within_root_rank == 1
                ),
                lexicon.fallback_by_hint.get(hint, hint),
            )
            for candidate in candidates:
                word = candidate.word
                records.append(
                    {
                        "example_id": source["example_id"],
                        "group_id": source["group_id"],
                        "split": source["split"],
                        "hint": hint,
                        "answer": answer,
                        "baseline_prediction": baseline,
                        "baseline_correct": baseline == answer,
                        "candidate": word,
                        "candidate_correct": word == answer,
                        "root_id": candidate.root_id,
                        "root_rank": candidate.root_rank,
                        "within_root_rank": candidate.within_root_rank,
                        "root_logit": candidate.root_logit,
                        "root_log_probability": candidate.root_log_probability,
                        "root_margin": candidate.root_margin,
                        "root_entropy": candidate.root_entropy,
                        "suffix_log_probability": candidate.suffix_log_probability,
                        "suffix_mean_log_probability": candidate.suffix_mean_log_probability,
                        "suffix_token_count": candidate.suffix_token_count,
                        "boundary_log_probability": candidate.boundary_log_probability,
                        "word_count": candidate.word_count,
                        "log_word_count": math.log1p(candidate.word_count),
                        "root_candidate_count": len(
                            lexicon.hint_root_words[hint][candidate.root_id]
                        ),
                        "is_representative": candidate.within_root_rank == 1,
                        "is_numeric": word.isnumeric(),
                        "is_alpha": word.isalpha(),
                    }
                )
        pd.DataFrame.from_records(records).to_csv(
            temporary, mode="a", header=not wrote_header, index=False
        )
        wrote_header = True
        print(f"Cached {min(start + len(chunk), len(split_frame)):,}/{len(split_frame):,}", flush=True)
    temporary.replace(args.output)

    metadata = {
        "scorer_version": SCORER_VERSION,
        "created_unix": time.time(),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "dev_path": str(args.dev_path),
        "dev_sha256": file_sha256(args.dev_path),
        "train_path": str(args.train_path),
        "train_sha256": file_sha256(args.train_path),
        "tokenizer_vocab_sha256": hashlib.sha256(
            json.dumps(tokenizer.get_vocab(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "model_name": args.model_name,
        "root_beam": args.root_beam,
        "words_per_root": args.words_per_root,
        "boundary_score": args.boundary_score,
        "rows_by_split": split_frame["split"].value_counts().to_dict(),
    }
    atomic_text_write(
        json.dumps(metadata, indent=2) + "\n", args.output.with_suffix(".meta.json")
    )


def parse_number_list(value: str, cast):
    return [cast(item) for item in value.split(",") if item]


def choose_predictions(
    candidates: pd.DataFrame,
    *,
    words_per_root: int,
    suffix_weight: float,
    length_penalty: float,
    frequency_weight: float,
    boundary_weight: float,
    confidence_gate: float | None,
) -> pd.DataFrame:
    pool = candidates.loc[candidates["within_root_rank"] <= words_per_root].copy()
    suffix_scale = pool["suffix_token_count"].clip(lower=1).pow(length_penalty)
    pool["experiment_score"] = (
        pool["root_log_probability"]
        + suffix_weight * pool["suffix_log_probability"] / suffix_scale
        + frequency_weight * pool["log_word_count"]
        + boundary_weight * pool["boundary_log_probability"].fillna(0.0)
    )
    root_one = pool.loc[pool["root_rank"] == 1]
    within = (
        root_one.sort_values(
            ["example_id", "experiment_score", "within_root_rank", "candidate"],
            ascending=[True, False, True, True],
        )
        .drop_duplicates("example_id")
        .set_index("example_id")["candidate"]
    )
    if confidence_gate is None:
        prediction = within
    else:
        cross = (
            pool.sort_values(
                [
                    "example_id",
                    "experiment_score",
                    "root_rank",
                    "within_root_rank",
                    "candidate",
                ],
                ascending=[True, False, True, True, True],
            )
            .drop_duplicates("example_id")
            .set_index("example_id")["candidate"]
        )
        confidence = root_one.loc[root_one["within_root_rank"] == 1].set_index(
            "example_id"
        )["root_log_probability"].map(math.exp)
        prediction = within.where(confidence > confidence_gate, cross)
    examples = pool.loc[
        (pool["root_rank"] == 1) & (pool["within_root_rank"] == 1)
    ].set_index("example_id")
    return pd.DataFrame(
        {
            "group_id": examples["group_id"],
            "answer": examples["answer"],
            "baseline_prediction": examples["baseline_prediction"],
            "prediction": prediction,
        }
    ).dropna(subset=["prediction"])


def paired_metrics(predictions: pd.DataFrame, *, bootstrap_samples: int, seed: int) -> dict:
    baseline = predictions["baseline_prediction"].eq(predictions["answer"])
    reranked = predictions["prediction"].eq(predictions["answer"])
    gained = int((~baseline & reranked).sum())
    lost = int((baseline & ~reranked).sum())
    result = {
        "examples": len(predictions),
        "baseline_accuracy": float(baseline.mean()),
        "accuracy": float(reranked.mean()),
        "gain": float(reranked.mean() - baseline.mean()),
        "wrong_to_correct": gained,
        "correct_to_wrong": lost,
    }
    if not bootstrap_samples:
        return result
    grouped = pd.DataFrame(
        {"group_id": predictions["group_id"], "delta": reranked.astype(int) - baseline.astype(int)}
    ).groupby("group_id")["delta"].agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    values = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        rows = grouped.iloc[sampled]
        values[index] = rows["sum"].sum() / rows["size"].sum()
    result["gain_ci_low"] = float(np.quantile(values, 0.025))
    result["gain_ci_high"] = float(np.quantile(values, 0.975))
    return result


def evaluate_heuristics(args: argparse.Namespace) -> None:
    words_per_root = parse_number_list(args.words_per_root, int)
    boundary_weights = parse_number_list(args.boundary_weights, float)
    candidates, _metadata = load_cache(
        args.cache,
        required_words_per_root=max(words_per_root),
        require_boundary=any(weight != 0 for weight in boundary_weights),
    )
    grids = itertools.product(
        words_per_root,
        parse_number_list(args.suffix_weights, float),
        parse_number_list(args.length_penalties, float),
        parse_number_list(args.frequency_weights, float),
        boundary_weights,
        [None] + parse_number_list(args.confidence_gates, float),
    )
    tuning = candidates.loc[candidates["split"] == "tune_valid"]
    rows = []
    for values in grids:
        names = (
            "words_per_root",
            "suffix_weight",
            "length_penalty",
            "frequency_weight",
            "boundary_weight",
            "confidence_gate",
        )
        config = dict(zip(names, values))
        predictions = choose_predictions(tuning, **config)
        rows.append({**config, **paired_metrics(predictions, bootstrap_samples=0, seed=args.seed)})
    results = pd.DataFrame(rows).sort_values(
        ["accuracy", "correct_to_wrong", "words_per_root"],
        ascending=[False, True, True],
    )
    best = results.iloc[0].to_dict()
    best_config = {
        name: (None if name == "confidence_gate" and pd.isna(best[name]) else best[name])
        for name in (
            "words_per_root",
            "suffix_weight",
            "length_penalty",
            "frequency_weight",
            "boundary_weight",
            "confidence_gate",
        )
    }
    best_config["words_per_root"] = int(best_config["words_per_root"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "heuristic_tuning.csv", index=False)
    summary = {
        "best_config": best_config,
        "validation_metrics": {
            key: best[key]
            for key in (
                "examples",
                "baseline_accuracy",
                "accuracy",
                "gain",
                "wrong_to_correct",
                "correct_to_wrong",
            )
        },
    }
    atomic_text_write(
        json.dumps(summary, indent=2) + "\n",
        args.output_dir / "heuristic_selection.json",
    )
    print(json.dumps(summary, indent=2), flush=True)


def model_predictions(candidates: pd.DataFrame, scores: np.ndarray, gate: float) -> pd.DataFrame:
    pool = candidates.copy()
    pool["experiment_score"] = scores
    within = (
        pool.loc[pool["root_rank"] == 1]
        .sort_values(
            ["example_id", "experiment_score", "within_root_rank", "candidate"],
            ascending=[True, False, True, True],
        )
        .drop_duplicates("example_id")
        .set_index("example_id")["candidate"]
    )
    cross = (
        pool.sort_values(
            ["example_id", "experiment_score", "root_rank", "within_root_rank", "candidate"],
            ascending=[True, False, True, True, True],
        )
        .drop_duplicates("example_id")
        .set_index("example_id")["candidate"]
    )
    representatives = pool.loc[
        (pool["root_rank"] == 1) & (pool["within_root_rank"] == 1)
    ].set_index("example_id")
    confidence = representatives["root_log_probability"].map(math.exp)
    prediction = within.where(confidence > gate, cross)
    return pd.DataFrame(
        {
            "group_id": representatives["group_id"],
            "answer": representatives["answer"],
            "baseline_prediction": representatives["baseline_prediction"],
            "prediction": prediction,
        }
    ).dropna(subset=["prediction"])


def validate_fusion_inputs(
    candidates: pd.DataFrame,
    candidate_metadata: dict,
    score_caches: dict,
    base_label: str,
) -> pd.DataFrame:
    if base_label not in score_caches:
        raise ValueError(f"unknown base score cache: {base_label}")
    if candidates.duplicated(["example_id", "candidate"]).any():
        raise ValueError("candidate cache has duplicate candidates within an example")
    for cache in score_caches.values():
        if cache.metadata["dev_sha256"] != candidate_metadata["dev_sha256"]:
            raise ValueError(f"score cache {cache.label} has a different development set")
        if cache.metadata["train_sha256"] != candidate_metadata["train_sha256"]:
            raise ValueError(f"score cache {cache.label} has a different training corpus")
        if cache.metadata.get("dev_path"):
            reject_forbidden_dev(cache.metadata["dev_path"])
    if (
        score_caches[base_label].metadata["checkpoint_sha256"]
        != candidate_metadata["checkpoint_sha256"]
    ):
        raise ValueError("base score cache does not match the full-word candidate checkpoint")

    representatives = candidates.loc[
        (candidates["root_rank"] == 1) & (candidates["within_root_rank"] == 1)
    ].copy()
    if representatives["example_id"].duplicated().any():
        raise ValueError("candidate cache has duplicate baseline representatives")
    representatives = representatives.sort_values("example_id").reset_index(drop=True)
    reference = score_caches[base_label].examples.reset_index(drop=True)
    if len(representatives) != len(reference):
        raise ValueError("candidate and score caches have different example counts")
    expected_ids = np.arange(len(reference), dtype=np.int64)
    if not np.array_equal(representatives["example_id"].to_numpy(), expected_ids):
        raise ValueError("candidate cache example IDs are not complete and ordered")
    checks = (
        ("group_id", "group_id"),
        ("hint", "first letter"),
        ("answer", "answer"),
    )
    contamination_masks = []
    reference_candidate_hash = score_caches[base_label].metadata.get("candidate_sha256")
    for label, cache in score_caches.items():
        cache_examples = cache.examples.reset_index(drop=True)
        if not np.array_equal(cache_examples["example_id"].to_numpy(), expected_ids):
            raise ValueError(f"score cache {label} is not physically ordered by example ID")
        for candidate_column, score_column in checks:
            if not representatives[candidate_column].astype(str).equals(
                cache_examples[score_column].astype(str)
            ):
                raise ValueError(
                    f"candidate and score cache {label} disagree on {candidate_column}"
                )
        contaminated = cache_examples["checkpoint_selection_contaminated"]
        if contaminated.dtype == object:
            contaminated = contaminated.astype(str).str.casefold().eq("true")
        contamination_masks.append(contaminated.astype(bool).to_numpy())
        if (
            reference_candidate_hash is not None
            and cache.metadata.get("candidate_sha256") != reference_candidate_hash
        ):
            raise ValueError(f"score cache {label} has an incompatible root lexicon")
    if not representatives["split"].eq("contaminated").equals(
        pd.Series(contamination_masks[0])
    ):
        raise ValueError("candidate and score caches disagree on contaminated groups")
    representatives["fusion_contaminated"] = np.logical_or.reduce(contamination_masks)

    base = score_caches[base_label]
    roots = candidates.loc[candidates["within_root_rank"].eq(1), ["hint", "root_id", "candidate"]]
    roots = roots.drop_duplicates()
    for row in roots.itertuples(index=False):
        position = base.token_positions.get(str(row.hint), {}).get(int(row.root_id))
        if position is None:
            raise ValueError(f"base score cache is missing root {row.root_id}")
        expected_word = base.candidates["hints"][str(row.hint)][position]["word"]
        if str(row.candidate) != expected_word:
            raise ValueError(
                f"candidate representative for root {row.root_id} does not match base cache"
            )
    return representatives


def build_checkpoint_root_features(
    candidates: pd.DataFrame, score_caches: dict
) -> tuple[pd.DataFrame, dict[str, str]]:
    sidecar = (
        candidates[["example_id", "hint", "root_id"]]
        .drop_duplicates()
        .sort_values(["example_id", "root_id"])
        .reset_index(drop=True)
    )
    feature_names = {}
    used_names = set()
    for label, cache in score_caches.items():
        safe_label = "".join(character if character.isalnum() else "_" for character in label)
        column = f"checkpoint_{safe_label}_root_log_probability"
        if column in used_names:
            raise ValueError("score cache labels do not produce unique feature names")
        used_names.add(column)
        feature_names[label] = column
        values = np.empty(len(sidecar), dtype=np.float64)
        for example_id, group in sidecar.groupby("example_id", sort=True):
            hint = str(group["hint"].iloc[0])
            if not group["hint"].astype(str).eq(hint).all():
                raise ValueError(f"example {example_id} has inconsistent hints")
            if str(cache.examples.iloc[int(example_id)]["first letter"]) != hint:
                raise ValueError(f"score cache {label} hint mismatch at example {example_id}")
            positions = cache.token_positions.get(hint, {})
            root_positions = []
            for root_id in group["root_id"]:
                position = positions.get(int(root_id))
                if position is None:
                    raise ValueError(
                        f"score cache {label} is missing root {root_id} for hint {hint!r}"
                    )
                root_positions.append(position)
            values[group.index.to_numpy()] = cache.row(int(example_id))[root_positions]
        sidecar[column] = values
    return sidecar, feature_names


def merge_checkpoint_root_features(
    candidates: pd.DataFrame, sidecar: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    merged = candidates.merge(
        sidecar[["example_id", "hint", "root_id", *feature_columns]],
        on=["example_id", "hint", "root_id"],
        how="left",
        validate="many_to_one",
    )
    if len(merged) != len(candidates) or merged[feature_columns].isna().any().any():
        raise ValueError("checkpoint feature join was incomplete")
    return merged


def grouped_fold(group_id: str, folds: int) -> int:
    return int(hashlib.sha256(group_id.encode("ascii")).hexdigest()[:8], 16) % folds


def fit_candidate_ranker(
    train: pd.DataFrame,
    feature_columns: list[str],
    *,
    learning_rate: float,
    max_iter: int,
    max_leaf_nodes: int,
    l2_regularization: float,
    seed: int,
):
    from sklearn.ensemble import HistGradientBoostingClassifier

    labels = train["candidate_correct"].astype(int).to_numpy()
    if train.empty or len(np.unique(labels)) != 2:
        raise ValueError("ranker training requires non-empty positive and negative classes")
    group_size = train.groupby("example_id")["example_id"].transform("size").to_numpy()
    positives = max(1, int(labels.sum()))
    negatives = max(1, len(labels) - positives)
    class_weight = np.where(
        labels == 1,
        len(labels) / (2 * positives),
        len(labels) / (2 * negatives),
    )
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        random_state=seed,
    )
    model.fit(train[feature_columns], labels, sample_weight=class_weight / group_size)
    return model


def ranking_metrics(candidates: pd.DataFrame, scores: np.ndarray) -> dict:
    ranked = candidates[
        ["example_id", "candidate_correct", "candidate", "root_rank", "within_root_rank"]
    ].copy()
    ranked["score"] = scores
    ranked = ranked.sort_values(
        ["example_id", "score", "root_rank", "within_root_rank", "candidate"],
        ascending=[True, False, True, True, True],
    )
    ranked["rank"] = ranked.groupby("example_id").cumcount() + 1
    correct = (
        ranked.loc[ranked["candidate_correct"].astype(bool)]
        .sort_values(["example_id", "rank"])
        .drop_duplicates("example_id")
    )
    total = int(ranked["example_id"].nunique())
    reciprocal = 1.0 / correct.set_index("example_id")["rank"]
    return {
        "examples": total,
        "candidate_oracle_coverage": float(correct["example_id"].nunique() / total),
        "top5_accuracy": float(correct["rank"].le(5).sum() / total),
        "mean_reciprocal_rank": float(reciprocal.sum() / total),
    }


def paired_columns_metrics(
    frame: pd.DataFrame,
    baseline_column: str,
    prediction_column: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    comparison = frame[["group_id", "answer", baseline_column, prediction_column]].rename(
        columns={baseline_column: "baseline_prediction", prediction_column: "prediction"}
    )
    return paired_metrics(
        comparison, bootstrap_samples=bootstrap_samples, seed=seed
    )


def sanitize_ranker_features(frame: pd.DataFrame, feature_columns: list[str]) -> None:
    frame[feature_columns] = frame[feature_columns].replace(
        {np.inf: FINITE_SCORE_SENTINEL, -np.inf: -FINITE_SCORE_SENTINEL}
    )


def fusion_example_diagnostics(
    candidates: pd.DataFrame, representatives: pd.DataFrame, score_caches: dict, base_label: str
) -> pd.DataFrame:
    base = score_caches[base_label]
    examples = representatives[
        ["example_id", "group_id", "split", "hint", "answer", "baseline_prediction"]
    ].copy()
    true_roots = []
    collisions = []
    frequencies = []
    for row in examples.itertuples(index=False):
        root_id = base.candidates["word_roots"].get(row.answer)
        true_roots.append(root_id)
        representative = None
        if root_id is not None and int(root_id) in base.token_positions.get(row.hint, {}):
            position = base.token_positions[row.hint][int(root_id)]
            representative = base.candidates["hints"][row.hint][position]["word"]
        collisions.append(representative is not None and representative != row.answer)
        frequencies.append(int(base.candidates["word_frequencies"].get(row.answer, 0)))
    examples["true_root_id"] = true_roots
    examples["root_collision"] = collisions
    examples["answer_training_frequency"] = frequencies
    examples["frequency_band"] = pd.cut(
        examples["answer_training_frequency"],
        bins=[-1, 0, 10, 100, 1_000, math.inf],
        labels=["unrepresented", "1-10", "11-100", "101-1000", "1001+"],
    ).astype(str)
    oracle = candidates.groupby("example_id")["candidate_correct"].any()
    selected_roots = candidates.groupby("example_id")["root_id"].agg(set)
    examples["candidate_oracle"] = examples["example_id"].map(oracle).astype(bool)
    examples["true_root_in_beam"] = [
        root_id is not None and int(root_id) in selected_roots[example_id]
        for example_id, root_id in zip(examples["example_id"], examples["true_root_id"])
    ]
    examples["candidate_miss"] = np.select(
        [
            examples["candidate_oracle"],
            examples["true_root_id"].isna(),
            ~examples["true_root_in_beam"],
        ],
        ["reachable", "answer_unrepresented", "root_outside_beam"],
        default="word_outside_per_root_limit",
    )
    top_roots = []
    for label, cache in score_caches.items():
        roots = []
        predictions = []
        for example_id, hint in zip(examples["example_id"], examples["hint"]):
            position = int(np.argmax(cache.row(int(example_id))))
            candidate = cache.candidates["hints"][hint][position]
            roots.append(int(candidate["token_id"]))
            predictions.append(str(candidate["word"]))
        top_roots.append(roots)
        safe_label = "".join(
            character if character.isalnum() else "_" for character in label
        )
        examples[f"root_prediction_{safe_label}"] = predictions
    examples["checkpoints_agree"] = [
        len(set(values)) == 1 for values in zip(*top_roots)
    ]
    return examples


def run_checkpoint_fusion(args: argparse.Namespace) -> None:
    from . import comprehensive_checkpoint_evaluation as comprehensive

    if args.folds < 2:
        raise ValueError("--folds must be at least two")
    candidates, candidate_metadata = load_cache(args.cache)
    score_caches = comprehensive.parse_cache_specs(args.score_cache)
    representatives = validate_fusion_inputs(
        candidates, candidate_metadata, score_caches, args.base_label
    )
    sidecar, feature_names = build_checkpoint_root_features(candidates, score_caches)
    treatment_features = [
        feature_names[label] for label in score_caches if label != args.base_label
    ]
    if not treatment_features:
        raise ValueError("checkpoint fusion requires at least one non-base score cache")
    candidates = merge_checkpoint_root_features(
        candidates, sidecar, list(feature_names.values())
    )
    candidates["boundary_log_probability"] = candidates[
        "boundary_log_probability"
    ].fillna(0.0)
    contaminated_by_id = representatives.set_index("example_id")["fusion_contaminated"]
    candidates["fusion_contaminated"] = candidates["example_id"].map(
        contaminated_by_id
    )
    candidates = candidates.loc[~candidates["fusion_contaminated"]].copy()
    candidates["fold"] = candidates["group_id"].map(
        lambda value: grouped_fold(str(value), args.folds)
    )
    reachable = candidates.groupby("example_id")["candidate_correct"].transform("any")
    control_scores = np.full(len(candidates), np.nan, dtype=np.float64)
    treatment_scores = np.full(len(candidates), np.nan, dtype=np.float64)
    control_models = []
    treatment_models = []
    control_features = list(FEATURE_COLUMNS)
    treatment_feature_columns = control_features + treatment_features
    sanitize_ranker_features(candidates, treatment_feature_columns)
    for fold in range(args.folds):
        train_mask = candidates["fold"].ne(fold) & reachable
        valid_mask = candidates["fold"].eq(fold)
        train = candidates.loc[train_mask]
        valid = candidates.loc[valid_mask]
        control = fit_candidate_ranker(
            train,
            control_features,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            seed=args.seed + fold,
        )
        treatment = fit_candidate_ranker(
            train,
            treatment_feature_columns,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            seed=args.seed + fold,
        )
        control_scores[valid_mask.to_numpy()] = control.predict_proba(
            valid[control_features]
        )[:, 1]
        treatment_scores[valid_mask.to_numpy()] = treatment.predict_proba(
            valid[treatment_feature_columns]
        )[:, 1]
        control_models.append(control)
        treatment_models.append(treatment)
        print(f"Completed grouped fold {fold + 1}/{args.folds}", flush=True)
    if not np.isfinite(control_scores).all() or not np.isfinite(treatment_scores).all():
        raise RuntimeError("out-of-fold predictions are incomplete")

    control_predictions = model_predictions(candidates, control_scores, args.confidence_gate)
    treatment_predictions = model_predictions(
        candidates, treatment_scores, args.confidence_gate
    )
    diagnostics = fusion_example_diagnostics(
        candidates,
        representatives.loc[~representatives["fusion_contaminated"]],
        score_caches,
        args.base_label,
    ).set_index("example_id")
    diagnostics["fold"] = diagnostics["group_id"].map(
        lambda value: grouped_fold(str(value), args.folds)
    )
    diagnostics["control_prediction"] = control_predictions["prediction"]
    diagnostics["treatment_prediction"] = treatment_predictions["prediction"]
    if diagnostics[["control_prediction", "treatment_prediction"]].isna().any().any():
        raise RuntimeError("fusion predictions do not cover every non-contaminated example")

    metrics = {
        "control_vs_root_baseline": paired_columns_metrics(
            diagnostics,
            "baseline_prediction",
            "control_prediction",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "treatment_vs_root_baseline": paired_columns_metrics(
            diagnostics,
            "baseline_prediction",
            "treatment_prediction",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "treatment_vs_control": paired_columns_metrics(
            diagnostics,
            "control_prediction",
            "treatment_prediction",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "control_ranking": ranking_metrics(candidates, control_scores),
        "treatment_ranking": ranking_metrics(candidates, treatment_scores),
        "root_baselines": {},
        "subgroups": {},
    }
    for label in score_caches:
        safe_label = "".join(
            character if character.isalnum() else "_" for character in label
        )
        column = f"root_prediction_{safe_label}"
        metrics["root_baselines"][label] = {
            "accuracy": float(diagnostics[column].eq(diagnostics["answer"]).mean()),
            "treatment_comparison": paired_columns_metrics(
                diagnostics,
                column,
                "treatment_prediction",
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            ),
        }
    subgroup_masks = {
        "collision": diagnostics["root_collision"],
        "non_collision": ~diagnostics["root_collision"],
        "checkpoints_agree": diagnostics["checkpoints_agree"],
        "checkpoints_disagree": ~diagnostics["checkpoints_agree"],
    }
    subgroup_masks.update(
        {
            f"frequency_{band}": diagnostics["frequency_band"].eq(band)
            for band in sorted(diagnostics["frequency_band"].unique())
        }
    )
    for name, mask in subgroup_masks.items():
        subset = diagnostics.loc[mask]
        if subset.empty:
            continue
        metrics["subgroups"][name] = {
            "examples": len(subset),
            "treatment_vs_control": paired_columns_metrics(
                subset,
                "control_prediction",
                "treatment_prediction",
                bootstrap_samples=0,
                seed=args.seed,
            ),
            "treatment_vs_root_baseline": paired_columns_metrics(
                subset,
                "baseline_prediction",
                "treatment_prediction",
                bootstrap_samples=0,
                seed=args.seed,
            ),
        }
    metrics["candidate_misses"] = {
        str(name): int(count)
        for name, count in diagnostics["candidate_miss"].value_counts().items()
    }
    base_root_seconds = float(score_caches[args.base_label].metadata["elapsed_seconds"])
    full_word_seconds = float(candidate_metadata["elapsed_seconds"])
    additional_root_seconds = sum(
        float(cache.metadata["elapsed_seconds"])
        for label, cache in score_caches.items()
        if label != args.base_label
    )
    estimated_fusion_seconds = full_word_seconds + additional_root_seconds
    metrics["inference_cost_estimate"] = {
        "examples": len(representatives),
        "root_only_seconds": base_root_seconds,
        "full_word_seconds": full_word_seconds,
        "additional_checkpoint_root_seconds": additional_root_seconds,
        "fusion_seconds": estimated_fusion_seconds,
        "full_word_vs_root_ratio": full_word_seconds / base_root_seconds,
        "fusion_vs_root_ratio": estimated_fusion_seconds / base_root_seconds,
        "fusion_examples_per_second": len(representatives) / estimated_fusion_seconds,
        "qualification": "sum of separate full-development cache timings; excludes model loading, score merging, and classifier inference",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sidecar.to_csv(args.output_dir / "checkpoint_root_features.csv", index=False)
    diagnostics.reset_index().to_csv(
        args.output_dir / "grouped_oof_predictions.csv", index=False
    )
    with (args.output_dir / "oof_rankers.pkl").open("wb") as file:
        pickle.dump(
            {
                "control_models": control_models,
                "treatment_models": treatment_models,
                "control_features": control_features,
                "treatment_features": treatment_feature_columns,
                "confidence_gate": args.confidence_gate,
                "folds": args.folds,
            },
            file,
        )
    provenance = {
        "candidate_cache": str(args.cache),
        "candidate_cache_sha256": file_sha256(args.cache),
        "candidate_metadata": candidate_metadata,
        "base_label": args.base_label,
        "score_caches": {
            label: {
                "directory": str(cache.directory),
                "metadata": cache.metadata,
                "scores_sha256": file_sha256(cache.directory / "scores.npy"),
                "offsets_sha256": file_sha256(cache.directory / "offsets.npy"),
                "examples_sha256": file_sha256(cache.directory / "examples.csv"),
                "candidates_sha256": file_sha256(cache.directory / "candidates.json"),
            }
            for label, cache in score_caches.items()
        },
        "folds": args.folds,
        "fold_policy": "sha256(group_id) modulo folds",
        "excluded_contaminated_examples": int(
            representatives["fusion_contaminated"].sum()
        ),
        "evaluation_scope": "grouped out-of-fold development comparison; not a deployable final model or untouched test estimate",
        "confidence_gate": args.confidence_gate,
        "model_parameters": {
            "learning_rate": args.learning_rate,
            "max_iter": args.max_iter,
            "max_leaf_nodes": args.max_leaf_nodes,
            "l2_regularization": args.l2_regularization,
            "seed": args.seed,
        },
        "control_features": control_features,
        "treatment_features": treatment_feature_columns,
    }
    atomic_text_write(
        json.dumps(provenance, indent=2) + "\n", args.output_dir / "provenance.json"
    )
    atomic_text_write(
        json.dumps(metrics, indent=2) + "\n", args.output_dir / "metrics.json"
    )
    print(json.dumps(metrics, indent=2), flush=True)


def train_ranker(args: argparse.Namespace) -> None:
    candidates, _metadata = load_cache(args.cache)
    candidates["boundary_log_probability"] = candidates[
        "boundary_log_probability"
    ].fillna(0.0)
    feature_columns = {
        "all": FEATURE_COLUMNS,
        "suffix": SUFFIX_FEATURE_COLUMNS,
        "cheap": CHEAP_FEATURE_COLUMNS,
    }[args.feature_set]
    sanitize_ranker_features(candidates, feature_columns)
    reachable = candidates.groupby("example_id")["candidate_correct"].transform("any")
    train = candidates.loc[(candidates["split"] == "tune_train") & reachable].copy()
    labels = train["candidate_correct"].astype(int).to_numpy()
    model = fit_candidate_ranker(
        train,
        feature_columns,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        seed=args.seed,
    )

    valid = candidates.loc[candidates["split"] == "tune_valid"].copy()
    valid_scores = model.predict_proba(valid[feature_columns])[:, 1]
    gate_results = []
    for gate in parse_number_list(args.confidence_gates, float):
        predictions = model_predictions(valid, valid_scores, gate)
        gate_results.append(
            {"confidence_gate": gate, **paired_metrics(predictions, bootstrap_samples=0, seed=args.seed)}
        )
    gate_frame = pd.DataFrame(gate_results).sort_values(
        ["accuracy", "correct_to_wrong"], ascending=[False, True]
    )
    best_gate = float(gate_frame.iloc[0]["confidence_gate"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidate_ranker.pkl").open("wb") as file:
        pickle.dump({"model": model, "features": feature_columns, "gate": best_gate}, file)
    gate_frame.to_csv(args.output_dir / "ranker_gate_tuning.csv", index=False)
    summary = {
        "features": feature_columns,
        "training_candidate_rows": len(train),
        "training_positive_rows": int(labels.sum()),
        "best_confidence_gate": best_gate,
        "validation_metrics": gate_frame.iloc[0].to_dict(),
    }
    atomic_text_write(
        json.dumps(summary, indent=2) + "\n", args.output_dir / "ranker_selection.json"
    )
    print(json.dumps(summary, indent=2), flush=True)


def finalize_evaluation(args: argparse.Namespace) -> None:
    candidates, metadata = load_cache(args.cache)
    marker = args.cache.with_suffix(".finalized.json")
    if marker.exists():
        raise FileExistsError(
            f"Locked evaluation was already finalized; see {marker}"
        )
    locked = candidates.loc[candidates["split"] == "locked"].copy()
    if locked.empty:
        raise ValueError("candidate cache has no locked examples")

    if args.method == "heuristic":
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        config = selection["best_config"]
        predictions = choose_predictions(locked, **config)
    else:
        with args.selection.open("rb") as file:
            payload = pickle.load(file)
        feature_columns = payload.get("features")
        if feature_columns not in (
            FEATURE_COLUMNS,
            SUFFIX_FEATURE_COLUMNS,
            CHEAP_FEATURE_COLUMNS,
        ):
            raise ValueError("ranker feature schema does not match this code")
        locked["boundary_log_probability"] = locked[
            "boundary_log_probability"
        ].fillna(0.0)
        scores = payload["model"].predict_proba(locked[feature_columns])[:, 1]
        predictions = model_predictions(locked, scores, float(payload["gate"]))

    expected = set(locked["example_id"].unique())
    if set(predictions.index) != expected:
        raise RuntimeError("final predictions do not cover every locked example exactly once")
    metrics = paired_metrics(
        predictions, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    frozen = {
        "method": args.method,
        "selection_sha256": file_sha256(args.selection),
        "cache_metadata": metadata,
        "metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.reset_index().to_csv(
        args.output_dir / "locked_predictions.csv", index=False
    )
    atomic_text_write(
        json.dumps(frozen, indent=2) + "\n", args.output_dir / "locked_metrics.json"
    )
    atomic_text_write(json.dumps(frozen, indent=2) + "\n", marker)
    print(json.dumps(frozen, indent=2), flush=True)


def benchmark_latency(args: argparse.Namespace) -> None:
    reject_forbidden_dev(args.dev_path)
    for path in (args.checkpoint, args.train_path, args.dev_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    dev = pd.read_csv(args.dev_path, keep_default_na=False)
    validate_dev_frame(dev)
    sample = dev.sample(n=min(args.examples, len(dev)), random_state=args.seed)
    _device, tokenizer, model, checkpoint, lexicon = load_model_and_lexicon(args)
    contexts = sample["context"].tolist()
    hints = sample["first letter"].tolist()
    boundary_ids = (
        hml.build_boundary_token_ids(tokenizer)
        if args.feature_set == "all"
        else None
    )

    warmup_count = min(128, len(sample))
    hml.predict_top_k(
        model,
        tokenizer,
        contexts[:warmup_count],
        hints[:warmup_count],
        candidate_ids_by_hint=lexicon.candidate_ids_by_hint,
        hint_token_word=lexicon.hint_token_word,
        fallback_by_hint=lexicon.fallback_by_hint,
        top_k=1,
        batch_size=args.eval_batch_size,
        max_context_tokens=args.max_context_tokens,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    hml.predict_top_k(
        model,
        tokenizer,
        contexts,
        hints,
        candidate_ids_by_hint=lexicon.candidate_ids_by_hint,
        hint_token_word=lexicon.hint_token_word,
        fallback_by_hint=lexicon.fallback_by_hint,
        top_k=1,
        batch_size=args.eval_batch_size,
        max_context_tokens=args.max_context_tokens,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    baseline_seconds = time.perf_counter() - started

    hml.score_candidates(
        model,
        tokenizer,
        contexts[:warmup_count],
        hints[:warmup_count],
        candidate_ids_by_hint=lexicon.candidate_ids_by_hint,
        hint_root_words=lexicon.hint_root_words,
        word_bpe=lexicon.word_bpe,
        word_counts=lexicon.word_counts,
        words_per_root=args.words_per_root,
        root_beam=args.root_beam,
        cross_root_confidence_gate=args.confidence_gate,
        batch_size=args.eval_batch_size,
        candidate_batch_size=args.candidate_batch_size,
        max_context_tokens=args.max_context_tokens,
        boundary_token_ids=boundary_ids,
        score_suffixes=args.feature_set != "cheap",
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    candidate_rows = hml.score_candidates(
        model,
        tokenizer,
        contexts,
        hints,
        candidate_ids_by_hint=lexicon.candidate_ids_by_hint,
        hint_root_words=lexicon.hint_root_words,
        word_bpe=lexicon.word_bpe,
        word_counts=lexicon.word_counts,
        words_per_root=args.words_per_root,
        root_beam=args.root_beam,
        cross_root_confidence_gate=args.confidence_gate,
        batch_size=args.eval_batch_size,
        candidate_batch_size=args.candidate_batch_size,
        max_context_tokens=args.max_context_tokens,
        boundary_token_ids=boundary_ids,
        score_suffixes=args.feature_set != "cheap",
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rerank_seconds = time.perf_counter() - started
    result = {
        "checkpoint_step": int(checkpoint["step"]),
        "examples": len(sample),
        "baseline_seconds": baseline_seconds,
        "rerank_feature_seconds": rerank_seconds,
        "latency_ratio": rerank_seconds / baseline_seconds,
        "baseline_examples_per_second": len(sample) / baseline_seconds,
        "rerank_examples_per_second": len(sample) / rerank_seconds,
        "mean_candidates_scored": float(np.mean([len(row) for row in candidate_rows])),
        "confidence_gate": args.confidence_gate,
        "feature_set": args.feature_set,
    }
    print(json.dumps(result, indent=2), flush=True)


def add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ARTIFACTS_DIR / "gpt2-keyboard-script/best_dev_step_126000.pt",
    )
    parser.add_argument("--train-path", type=Path, default=TRAIN_DIR / "train.src.tok")
    parser.add_argument("--dev-path", type=Path, default=DATA_DIR / "devv_eval.csv")
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "gpt2-keyboard-script/rerank/candidates.csv",
    )
    parser.add_argument("--root-beam", type=int, default=5)
    parser.add_argument("--words-per-root", type=int, default=5)
    parser.add_argument("--selection-size", type=int, default=2_000)
    parser.add_argument("--tune-fraction", type=float, default=0.2)
    parser.add_argument("--tune-train-fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--candidate-batch-size", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--max-train-lines", type=int, default=None)
    parser.add_argument("--boundary-score", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe reranking experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache = subparsers.add_parser("cache", help="Score and cache candidate features")
    add_cache_arguments(cache)

    heuristic = subparsers.add_parser("heuristics", help="Tune heuristic rerank scores")
    heuristic.add_argument("--cache", type=Path, required=True)
    heuristic.add_argument("--output-dir", type=Path, required=True)
    heuristic.add_argument("--words-per-root", default="2,3,5")
    heuristic.add_argument("--suffix-weights", default="0.1,0.25,0.5,1")
    heuristic.add_argument("--length-penalties", default="0.5,1")
    heuristic.add_argument("--frequency-weights", default="0,0.05")
    heuristic.add_argument("--boundary-weights", default="0,0.25")
    heuristic.add_argument("--confidence-gates", default="0.3,0.5")
    heuristic.add_argument("--bootstrap-samples", type=int, default=2_000)
    heuristic.add_argument("--seed", type=int, default=42)

    ranker = subparsers.add_parser("train", help="Train a lightweight candidate ranker")
    ranker.add_argument("--cache", type=Path, required=True)
    ranker.add_argument("--output-dir", type=Path, required=True)
    ranker.add_argument("--confidence-gates", default="0.3,0.5,0.7,1")
    ranker.add_argument(
        "--feature-set", choices=("all", "suffix", "cheap"), default="all"
    )
    ranker.add_argument("--learning-rate", type=float, default=0.05)
    ranker.add_argument("--max-iter", type=int, default=200)
    ranker.add_argument("--max-leaf-nodes", type=int, default=31)
    ranker.add_argument("--l2-regularization", type=float, default=1.0)
    ranker.add_argument("--bootstrap-samples", type=int, default=2_000)
    ranker.add_argument("--seed", type=int, default=42)
    fusion = subparsers.add_parser(
        "checkpoint-fusion",
        help="Compare full-word ranking with and without additional checkpoint root scores",
    )
    fusion.add_argument("--cache", type=Path, required=True)
    fusion.add_argument(
        "--score-cache", action="append", required=True, metavar="LABEL=PATH"
    )
    fusion.add_argument("--base-label", required=True)
    fusion.add_argument("--output-dir", type=Path, required=True)
    fusion.add_argument("--folds", type=int, default=5)
    fusion.add_argument("--confidence-gate", type=float, default=0.5)
    fusion.add_argument("--learning-rate", type=float, default=0.05)
    fusion.add_argument("--max-iter", type=int, default=200)
    fusion.add_argument("--max-leaf-nodes", type=int, default=31)
    fusion.add_argument("--l2-regularization", type=float, default=1.0)
    fusion.add_argument("--bootstrap-samples", type=int, default=2_000)
    fusion.add_argument("--seed", type=int, default=42)
    finalize = subparsers.add_parser(
        "finalize", help="Run one locked evaluation after freezing a configuration"
    )
    finalize.add_argument("--cache", type=Path, required=True)
    finalize.add_argument("--method", choices=("heuristic", "ranker"), required=True)
    finalize.add_argument("--selection", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--bootstrap-samples", type=int, default=2_000)
    finalize.add_argument("--seed", type=int, default=42)
    benchmark = subparsers.add_parser(
        "benchmark", help="Compare baseline and selective-reranker latency"
    )
    benchmark.add_argument(
        "--checkpoint",
        type=Path,
        default=ARTIFACTS_DIR / "gpt2-keyboard-script/best_dev_step_126000.pt",
    )
    benchmark.add_argument("--train-path", type=Path, default=TRAIN_DIR / "train.src.tok")
    benchmark.add_argument("--dev-path", type=Path, default=DATA_DIR / "devv_eval.csv")
    benchmark.add_argument("--model-name", default="gpt2")
    benchmark.add_argument("--examples", type=int, default=4_096)
    benchmark.add_argument("--root-beam", type=int, default=5)
    benchmark.add_argument("--words-per-root", type=int, default=5)
    benchmark.add_argument("--confidence-gate", type=float, default=0.5)
    benchmark.add_argument(
        "--feature-set", choices=("all", "suffix", "cheap"), default="all"
    )
    benchmark.add_argument("--eval-batch-size", type=int, default=64)
    benchmark.add_argument("--candidate-batch-size", type=int, default=32)
    benchmark.add_argument("--max-context-tokens", type=int, default=256)
    benchmark.add_argument("--max-train-lines", type=int, default=None)
    benchmark.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "cache":
        build_cache(args)
    elif args.command == "heuristics":
        evaluate_heuristics(args)
    elif args.command == "train":
        train_ranker(args)
    elif args.command == "finalize":
        finalize_evaluation(args)
    elif args.command == "checkpoint-fusion":
        run_checkpoint_fusion(args)
    else:
        benchmark_latency(args)


if __name__ == "__main__":
    main()
