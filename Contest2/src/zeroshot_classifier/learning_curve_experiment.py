from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import shutil
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from tqdm.auto import tqdm

from .absa_training import (
    AspectDataset,
    ExperimentConfig as TrainingConfig,
    PolarityDataset,
    _attach_tokenizer,
    group_aspect_targets,
    train_stage,
)
from .checkpoint import RunLock
from .data import LabeledRow, read_labeled_csv
from .deberta_experiment import load_tokenizer
from .deberta_three_experiments import average_backbone_logits
from .evaluation import calculate_metrics
from .imbalance_losses import LossSpec
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .joint_absa_experiment import (
    JointExperimentConfig,
    _group_accuracy,
    _pair_class_metrics,
    _reshape,
    _separate_logits,
    decode_pairs,
    train_separate_polarity,
    tune_pair_thresholds,
)
from .labels import ASPECTS, POLARITIES
from .multilabel_aspect_experiment import (
    AspectExperimentConfig,
    _predict_checkpoint_probabilities,
    aspect_predictions,
    calculate_aspect_metrics,
    paired_bootstrap_delta,
    tune_thresholds,
)
from .roberta_training import _precision_settings, _prepare_mlflow_tracking_uri, _seed_everything


LOGGER = logging.getLogger(__name__)
FRACTIONS = (25, 50, 75, 100)
PRIMARY_GAIN = 0.01


@dataclass(frozen=True)
class LearningCurveConfig:
    input: Path = Path('artifacts/training/roberta-aspect-exp1/splits/train.csv')
    run_dir: Path = Path('artifacts/experiments/learning-curve-v1')
    evaluator: Path = Path('scripts/evaluate.py')
    roberta_model: str = 'FacebookAI/roberta-base'
    deberta_model: str = 'microsoft/deberta-v3-base'
    folds: int = 5
    selection_folds: int = 10
    seed: int = 42
    epochs: int = 50
    evaluation_interval: int = 5
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
    max_length: int = 256
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    checkpoint_steps: int = 100
    log_steps: int = 10
    mixed_precision: str = 'auto'
    bootstrap_samples: int = 10_000
    benchmark_minutes: float = 10.0
    mlflow_tracking_uri: str = (
        'sqlite:////home/kami/Projects/NLP/Contest2/artifacts/mlflow.db'
    )
    mlflow_experiment: str = 'contest2-learning-curve-v1'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the grouped ABSA learning curve.')
    parser.add_argument('mode', choices=('run', 'benchmark', 'report'))
    for name, field in LearningCurveConfig.__dataclass_fields__.items():
        default = field.default
        option = f'--{name.replace("_", "-")}'
        if isinstance(default, Path):
            parser.add_argument(option, type=Path, default=default)
        else:
            parser.add_argument(option, type=type(default), default=default)
    return parser


def config_from_args(argv: list[str] | None = None) -> tuple[str, LearningCurveConfig]:
    values = vars(build_parser().parse_args(argv))
    mode = values.pop('mode')
    config = LearningCurveConfig(**values)
    validate_config(config)
    return mode, config


def validate_config(config: LearningCurveConfig) -> None:
    positive = {
        'folds': config.folds,
        'selection-folds': config.selection_folds,
        'epochs': config.epochs,
        'evaluation-interval': config.evaluation_interval,
        'early-stopping-patience': config.early_stopping_patience,
        'max-length': config.max_length,
        'train-batch-size': config.train_batch_size,
        'eval-batch-size': config.eval_batch_size,
        'gradient-accumulation-steps': config.gradient_accumulation_steps,
        'log-steps': config.log_steps,
        'bootstrap-samples': config.bootstrap_samples,
        'benchmark-minutes': config.benchmark_minutes,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f'{", ".join(invalid)} must be positive')
    if config.folds < 2 or config.selection_folds < 2:
        raise ValueError('folds and selection-folds must be at least 2')
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError('learning-rate must be positive and weight-decay nonnegative')
    if config.early_stopping_min_delta < 0 or not 0 <= config.warmup_ratio < 1:
        raise ValueError('invalid early-stopping-min-delta or warmup-ratio')


def grouped_label_folds(
    rows: Sequence[LabeledRow], folds: int, seed: int,
) -> list[set[str]]:
    """Assign complete review IDs to approximately multilabel-stratified folds."""
    grouped: dict[str, list[LabeledRow]] = defaultdict(list)
    for row in rows:
        grouped[row.item_id].append(row)
    if folds < 2 or folds > len(grouped):
        raise ValueError('fold count must be between 2 and the unique-ID count')

    feature_count = len(ASPECTS) + len(POLARITIES) + 1
    features: dict[str, list[int]] = {}
    for item_id, values in grouped.items():
        aspects = {row.aspect for row in values}
        polarities = {row.polarity for row in values}
        features[item_id] = [
            *[int(label in aspects) for label in ASPECTS],
            *[int(label in polarities) for label in POLARITIES],
            int(len(aspects) > 1),
        ]
    totals = [sum(vector[index] for vector in features.values())
              for index in range(feature_count)]
    targets = [max(total / folds, 1.0) for total in totals]
    target_size = len(grouped) / folds
    rng = random.Random(seed)
    tie = {item_id: rng.random() for item_id in grouped}
    ordered = sorted(grouped, key=lambda item_id: (
        -sum(value / max(total, 1) for value, total in zip(features[item_id], totals)),
        -sum(features[item_id]), tie[item_id], item_id,
    ))
    assignments = [set() for _ in range(folds)]
    counts = [[0] * feature_count for _ in range(folds)]
    for item_id in ordered:
        vector = features[item_id]
        destination = min(range(folds), key=lambda fold: (
            sum(((counts[fold][index] + vector[index]) / targets[index]) ** 2
                for index in range(feature_count)),
            ((len(assignments[fold]) + 1) / target_size) ** 2,
            len(assignments[fold]), fold,
        ))
        assignments[destination].add(item_id)
        counts[destination] = [left + right for left, right in zip(counts[destination], vector)]
    return assignments


def rows_for_ids(rows: Sequence[LabeledRow], item_ids: set[str]) -> list[LabeledRow]:
    return [row for row in rows if row.item_id in item_ids]


def learning_curve_splits(
    rows: Sequence[LabeledRow], folds: int, selection_folds: int, seed: int,
) -> list[dict[str, Any]]:
    outer_folds = grouped_label_folds(rows, folds, seed)
    all_ids = {row.item_id for row in rows}
    output = []
    for fold_index, heldout_ids in enumerate(outer_folds, 1):
        outer_train_ids = all_ids - heldout_ids
        outer_train = rows_for_ids(rows, outer_train_ids)
        selection_ids = grouped_label_folds(
            outer_train, selection_folds, seed + fold_index,
        )[0]
        fit_ids = outer_train_ids - selection_ids
        fit_rows = rows_for_ids(rows, fit_ids)
        blocks = grouped_label_folds(fit_rows, 4, seed + 1_000 + fold_index)
        subsets: dict[int, set[str]] = {}
        accumulated: set[str] = set()
        for fraction, block in zip(FRACTIONS, blocks):
            accumulated = accumulated | block
            subsets[fraction] = set(accumulated)
        output.append({
            'fold': fold_index,
            'heldout_ids': set(heldout_ids),
            'selection_ids': set(selection_ids),
            'fit_ids': set(fit_ids),
            'subsets': subsets,
        })
    return output


def validate_split_plan(rows: Sequence[LabeledRow], plan: Sequence[dict[str, Any]]) -> None:
    all_ids = {row.item_id for row in rows}
    heldout_union: set[str] = set()
    for fold in plan:
        heldout = fold['heldout_ids']
        selection = fold['selection_ids']
        fit = fold['fit_ids']
        if heldout & selection or heldout & fit or selection & fit:
            raise RuntimeError(f"ID leakage in fold {fold['fold']}")
        if heldout | selection | fit != all_ids:
            raise RuntimeError(f"Incomplete ID partition in fold {fold['fold']}")
        previous: set[str] = set()
        for fraction in FRACTIONS:
            current = fold['subsets'][fraction]
            if not previous <= current or not current <= fit:
                raise RuntimeError(f"Non-nested subset in fold {fold['fold']}")
            previous = current
        if previous != fit:
            raise RuntimeError(f"100% subset is incomplete in fold {fold['fold']}")
        if heldout_union & heldout:
            raise RuntimeError('Outer heldout folds overlap')
        heldout_union |= heldout
    if heldout_union != all_ids:
        raise RuntimeError('Outer folds do not cover every ID')


def serialized_split_plan(plan: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        'fold': fold['fold'],
        'heldout_ids': sorted(fold['heldout_ids']),
        'selection_ids': sorted(fold['selection_ids']),
        'fit_ids': sorted(fold['fit_ids']),
        'subsets': {str(key): sorted(value) for key, value in fold['subsets'].items()},
    } for fold in plan]


def gold_aspect_polarity_metrics(
    rows: Sequence[LabeledRow], candidate_predictions: Sequence[Sequence[int]],
) -> dict[str, Any]:
    reviews = group_aspect_targets(rows)
    if len(reviews) != len(candidate_predictions):
        raise ValueError('Polarity predictions do not match grouped reviews')
    review_index = {review.item_id: index for index, review in enumerate(reviews)}
    gold: list[int] = []
    predicted: list[int] = []
    for row in rows:
        values = candidate_predictions[review_index[row.item_id]]
        if len(values) != len(ASPECTS):
            raise ValueError('Each review requires one polarity prediction per aspect')
        gold.append(POLARITIES.index(row.polarity))
        predicted.append(int(values[ASPECTS.index(row.aspect)]))
    return classification_metrics(gold, predicted)


def classification_metrics(gold: Sequence[int], predicted: Sequence[int]) -> dict[str, Any]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError('Gold and predicted polarity values must have equal nonzero length')
    if any(value < 0 or value >= len(POLARITIES) for value in [*gold, *predicted]):
        raise ValueError('Polarity index is outside the configured label range')
    confusion = [[0] * len(POLARITIES) for _ in POLARITIES]
    classes = {}
    for expected, actual in zip(gold, predicted):
        confusion[expected][actual] += 1
    for index, label in enumerate(POLARITIES):
        true_positive = confusion[index][index]
        false_positive = sum(confusion[row][index] for row in range(len(POLARITIES))) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        classes[label] = {
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'support': sum(confusion[index]),
        }
    return {
        'accuracy': sum(expected == actual for expected, actual in zip(gold, predicted)) / len(gold),
        'macro_f1': statistics.fmean(value['f1'] for value in classes.values()),
        'classes': classes,
        'confusion_matrix': confusion,
        'label_order': list(POLARITIES),
        'support': len(gold),
    }


def gold_polarity_rows(
    rows: Sequence[LabeledRow], candidate_predictions: Sequence[Sequence[int]],
) -> list[list[str]]:
    reviews = group_aspect_targets(rows)
    if len(reviews) != len(candidate_predictions):
        raise ValueError('Polarity predictions do not match grouped reviews')
    review_index = {review.item_id: index for index, review in enumerate(reviews)}
    return [[
        row.item_id, row.aspect, row.polarity,
        POLARITIES[int(candidate_predictions[review_index[row.item_id]][ASPECTS.index(row.aspect)])],
    ] for row in rows]


def pair_micro_f1(rows: Sequence[LabeledRow], predictions: Iterable[tuple[str, str, str]]) -> float:
    metrics = calculate_metrics(rows, predictions)
    return float(metrics['source_of_truth_metrics']['overall']['micro']['f1'])


def paired_pair_bootstrap(
    rows: Sequence[LabeledRow], contender: Sequence[tuple[str, str, str]],
    baseline: Sequence[tuple[str, str, str]], samples: int, seed: int,
) -> dict[str, float | int]:
    by_id: dict[str, list[LabeledRow]] = defaultdict(list)
    for row in rows:
        by_id[row.item_id].append(row)
    contender_by_id: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    baseline_by_id: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for value in contender:
        contender_by_id[value[0]].append(value)
    for value in baseline:
        baseline_by_id[value[0]].append(value)
    item_ids = sorted(by_id)
    if not item_ids:
        raise ValueError('Cannot bootstrap an empty dataset')
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        selected = rng.choices(item_ids, k=len(item_ids))
        sampled_rows = []
        sampled_left = []
        sampled_right = []
        for sample_index, item_id in enumerate(selected):
            synthetic_id = f'{sample_index}:{item_id}'
            sampled_rows.extend(replace(row, item_id=synthetic_id) for row in by_id[item_id])
            sampled_left.extend((synthetic_id, aspect, polarity)
                                for _, aspect, polarity in contender_by_id[item_id])
            sampled_right.extend((synthetic_id, aspect, polarity)
                                 for _, aspect, polarity in baseline_by_id[item_id])
        deltas.append(pair_micro_f1(sampled_rows, sampled_left) - pair_micro_f1(sampled_rows, sampled_right))
    ordered = sorted(deltas)
    return {
        'samples': samples,
        'mean': statistics.fmean(deltas),
        'lower_95': ordered[int(0.025 * (samples - 1))],
        'upper_95': ordered[int(0.975 * (samples - 1))],
    }


def paired_polarity_bootstrap(
    contender: Sequence[Sequence[str]], baseline: Sequence[Sequence[str]],
    samples: int, seed: int,
) -> dict[str, float | int]:
    left = {(value[0], value[1]): (value[2], value[3]) for value in contender}
    right = {(value[0], value[1]): (value[2], value[3]) for value in baseline}
    if set(left) != set(right) or not left:
        raise ValueError('Paired polarity predictions must cover the same nonempty annotations')
    by_id: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item_id, aspect in left:
        by_id[item_id].append((item_id, aspect))
    item_ids = sorted(by_id)
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        selected = rng.choices(item_ids, k=len(item_ids))
        left_gold = []
        left_predicted = []
        right_gold = []
        right_predicted = []
        for item_id in selected:
            for key in by_id[item_id]:
                gold, prediction = left[key]
                other_gold, other_prediction = right[key]
                left_gold.append(POLARITIES.index(gold))
                left_predicted.append(POLARITIES.index(prediction))
                right_gold.append(POLARITIES.index(other_gold))
                right_predicted.append(POLARITIES.index(other_prediction))
        deltas.append(
            float(classification_metrics(left_gold, left_predicted)['macro_f1'])
            - float(classification_metrics(right_gold, right_predicted)['macro_f1'])
        )
    ordered = sorted(deltas)
    return {
        'samples': samples, 'mean': statistics.fmean(deltas),
        'lower_95': ordered[int(0.025 * (samples - 1))],
        'upper_95': ordered[int(0.975 * (samples - 1))],
    }


def curve_diagnosis(delta: float, interval: dict[str, float | int]) -> str:
    if delta >= PRIMARY_GAIN and float(interval['lower_95']) > 0:
        return 'data-limited'
    if float(interval['upper_95']) < PRIMARY_GAIN:
        return 'plateaued'
    return 'inconclusive'


def _serialized_config(config: LearningCurveConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()}


def _training_config(
    config: LearningCurveConfig, model: str, run_dir: Path, seed: int,
) -> TrainingConfig:
    return TrainingConfig(
        input=config.input, split_dir=config.input.parent, run_dir=run_dir,
        model=model, epochs=config.epochs,
        evaluation_interval=config.evaluation_interval,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        seed=seed, max_length=config.max_length,
        learning_rate=config.learning_rate, train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        checkpoint_steps=config.checkpoint_steps, log_steps=config.log_steps,
        mixed_precision=config.mixed_precision,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        mlflow_experiment=config.mlflow_experiment,
        mlflow_run_name=f'learning-curve-aspect-{seed}',
    )


def _polarity_config(
    config: LearningCurveConfig, model: str, run_dir: Path,
) -> JointExperimentConfig:
    return JointExperimentConfig(
        input=config.input, split_dir=config.input.parent, run_dir=run_dir,
        evaluator=config.evaluator, model=model, epochs=config.epochs,
        evaluation_interval=config.evaluation_interval,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        max_length=config.max_length, learning_rate=config.learning_rate,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        log_steps=config.log_steps, mixed_precision=config.mixed_precision,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        mlflow_experiment=config.mlflow_experiment,
        bootstrap_samples=config.bootstrap_samples,
    )


def _aspect_probabilities(
    torch: Any, transformers: Any, tokenizer: Any, config: LearningCurveConfig,
    checkpoint: Path, rows: Sequence[LabeledRow], label: str,
) -> list[list[float]]:
    inference = AspectExperimentConfig(
        input=config.input, split_dir=config.input.parent,
        model=config.roberta_model, seeds=(config.seed,),
        max_length=config.max_length, eval_batch_size=config.eval_batch_size,
    )
    return _predict_checkpoint_probabilities(
        torch, transformers, tokenizer, inference, checkpoint, rows, label,
    )


def _candidate_indices(logits: Any, rows: Sequence[LabeledRow]) -> list[list[int]]:
    return _reshape(logits.argmax(dim=-1).tolist(), len(group_aspect_targets(rows)))


def _stage_manifest(
    config: LearningCurveConfig, fold: int, fraction: int,
    train_ids: set[str], selection_ids: set[str], heldout_ids: set[str],
) -> dict[str, Any]:
    return {
        'config': _serialized_config(config), 'input_sha256': file_sha256(config.input),
        'fold': fold, 'fraction': fraction, 'seed': config.seed + fold,
        'train_ids': sorted(train_ids), 'selection_ids': sorted(selection_ids),
        'heldout_ids': sorted(heldout_ids),
        'models': {
            'aspect': config.roberta_model,
            'polarity': [config.roberta_model, config.deberta_model],
        },
    }


def _write_or_validate(path: Path, value: dict[str, Any]) -> None:
    if path.is_file() and json.loads(path.read_text(encoding='utf-8')) != value:
        raise RuntimeError(f'Existing manifest differs from requested run: {path}')
    atomic_write_json(path, value)


def _cleanup_fraction_checkpoints(directory: Path) -> None:
    for relative in (
        'aspect/best', 'aspect/checkpoints',
        'polarity/roberta/checkpoints', 'polarity/deberta/checkpoints',
    ):
        shutil.rmtree(directory / relative, ignore_errors=True)
    for path in (directory / 'polarity/roberta/best.pt', directory / 'polarity/deberta/best.pt'):
        path.unlink(missing_ok=True)


def _run_fraction(
    torch: Any, transformers: Any, mlflow: Any,
    roberta_tokenizer: Any, deberta_tokenizer: Any,
    config: LearningCurveConfig, all_rows: list[LabeledRow], fold: dict[str, Any], fraction: int,
) -> dict[str, Any]:
    directory = config.run_dir / f"fold-{fold['fold']}" / f'fraction-{fraction}'
    result_path = directory / 'result.json'
    manifest = _stage_manifest(
        config, fold['fold'], fraction, fold['subsets'][fraction],
        fold['selection_ids'], fold['heldout_ids'],
    )
    directory.mkdir(parents=True, exist_ok=True)
    _write_or_validate(directory / 'manifest.json', manifest)
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result.get('finished'):
            LOGGER.info('Reusing fold %d fraction %d', fold['fold'], fraction)
            return result

    seed = config.seed + fold['fold']
    train_rows = rows_for_ids(all_rows, fold['subsets'][fraction])
    selection_rows = rows_for_ids(all_rows, fold['selection_ids'])
    heldout_rows = rows_for_ids(all_rows, fold['heldout_ids'])

    aspect_dir = directory / 'aspect'
    aspect_result_path = aspect_dir / 'result.json'
    if aspect_result_path.is_file():
        aspect_training = json.loads(aspect_result_path.read_text())
    else:
        _seed_everything(torch, seed)
        with mlflow.start_run(run_name=f'lc-aspect-f{fold["fold"]}-{fraction}'):
            aspect_training = train_stage(
                torch, transformers, mlflow, roberta_tokenizer,
                _training_config(config, config.roberta_model, aspect_dir, seed),
                'aspect', train_rows, selection_rows,
                checkpoint_on_evaluation_only=True,
            )
        atomic_write_json(aspect_result_path, aspect_training)
    aspect_checkpoint = aspect_dir / 'best/aspect.pt'
    selection_aspects = _aspect_probabilities(
        torch, transformers, roberta_tokenizer, config, aspect_checkpoint,
        selection_rows, f'fold {fold["fold"]} {fraction}% selection aspects',
    )
    heldout_aspects = _aspect_probabilities(
        torch, transformers, roberta_tokenizer, config, aspect_checkpoint,
        heldout_rows, f'fold {fold["fold"]} {fraction}% heldout aspects',
    )
    aspect_thresholds = tune_thresholds(
        selection_aspects,
        [[int(index in review.aspects) for index in range(len(ASPECTS))]
         for review in group_aspect_targets(selection_rows)],
    )
    heldout_aspect_predictions = aspect_predictions(
        heldout_rows, heldout_aspects, aspect_thresholds,
    )

    polarity_logits = {}
    polarity_training = {}
    for name, model, tokenizer in (
        ('roberta', config.roberta_model, roberta_tokenizer),
        ('deberta', config.deberta_model, deberta_tokenizer),
    ):
        polarity_dir = directory / 'polarity' / name
        polarity_config = _polarity_config(config, model, polarity_dir)
        polarity_training[name] = train_separate_polarity(
            torch, transformers, mlflow, tokenizer, polarity_config,
            polarity_dir, LossSpec('weighted-ce'), seed,
            train_rows, selection_rows, selection_aspects,
            checkpoint_on_evaluation_only=True,
        )
        checkpoint = polarity_dir / 'best.pt'
        polarity_logits[name] = {
            'selection': _separate_logits(
                torch, transformers, tokenizer, polarity_config, checkpoint,
                selection_rows, f'fold {fold["fold"]} {fraction}% {name} selection polarity',
            ),
            'heldout': _separate_logits(
                torch, transformers, tokenizer, polarity_config, checkpoint,
                heldout_rows, f'fold {fold["fold"]} {fraction}% {name} heldout polarity',
            ),
        }

    mixed_selection = average_backbone_logits(
        torch, polarity_logits['roberta']['selection'], polarity_logits['deberta']['selection'],
    )
    mixed_heldout = average_backbone_logits(
        torch, polarity_logits['roberta']['heldout'], polarity_logits['deberta']['heldout'],
    )
    selection_polarities = _candidate_indices(mixed_selection, selection_rows)
    pair_thresholds = tune_pair_thresholds(
        selection_rows, selection_aspects, selection_polarities,
    )
    heldout_polarities = _candidate_indices(mixed_heldout, heldout_rows)
    pair_predictions = decode_pairs(
        group_aspect_targets(heldout_rows), heldout_aspects,
        heldout_polarities, pair_thresholds,
    )
    pair_metrics = calculate_metrics(heldout_rows, pair_predictions)
    pair_metrics['polarity_pair_classes'] = _pair_class_metrics(heldout_rows, pair_predictions)
    pair_metrics['review_subsets'] = _group_accuracy(heldout_rows, pair_predictions)
    result = {
        'finished': True, 'fold': fold['fold'], 'fraction': fraction,
        'seed': seed, 'counts': {
            'train_ids': len(fold['subsets'][fraction]), 'train_rows': len(train_rows),
            'selection_ids': len(fold['selection_ids']), 'selection_rows': len(selection_rows),
            'heldout_ids': len(fold['heldout_ids']), 'heldout_rows': len(heldout_rows),
        },
        'training': {'aspect': aspect_training, 'polarity': polarity_training},
        'thresholds': {'aspect': aspect_thresholds, 'pair': pair_thresholds},
        'metrics': {
            'aspect': calculate_aspect_metrics(heldout_rows, heldout_aspect_predictions),
            'polarity': {
                name: gold_aspect_polarity_metrics(
                    heldout_rows, _candidate_indices(values['heldout'], heldout_rows),
                ) for name, values in polarity_logits.items()
            } | {'mixed': gold_aspect_polarity_metrics(heldout_rows, heldout_polarities)},
            'pair': pair_metrics,
        },
        'predictions': {
            'aspect': [list(value) for value in heldout_aspect_predictions],
            'pair': [list(value) for value in pair_predictions],
            'polarity': {
                name: gold_polarity_rows(
                    heldout_rows, _candidate_indices(values['heldout'], heldout_rows),
                ) for name, values in polarity_logits.items()
            } | {'mixed': gold_polarity_rows(heldout_rows, heldout_polarities)},
        },
    }
    atomic_write_json(result_path, result)
    _cleanup_fraction_checkpoints(directory)
    return result


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    return {
        'mean': statistics.fmean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _fraction_summary(
    rows: list[LabeledRow], results: Sequence[dict[str, Any]], fraction: int,
) -> dict[str, Any]:
    selected = [value for value in results if value['fraction'] == fraction]
    pair_predictions = [tuple(item) for value in selected for item in value['predictions']['pair']]
    aspect_predictions_values = [tuple(item) for value in selected for item in value['predictions']['aspect']]
    pair_folds = [float(value['metrics']['pair']['source_of_truth_metrics']['overall']['micro']['f1'])
                  for value in selected]
    aspect_folds = [float(value['metrics']['aspect']['aspect']['micro']['f1']) for value in selected]
    polarity_folds = [float(value['metrics']['polarity']['mixed']['macro_f1']) for value in selected]
    pooled_polarity = {}
    for name in ('roberta', 'deberta', 'mixed'):
        values = [item for result in selected for item in result['predictions']['polarity'][name]]
        pooled_polarity[name] = classification_metrics(
            [POLARITIES.index(item[2]) for item in values],
            [POLARITIES.index(item[3]) for item in values],
        )
    pooled_pair = calculate_metrics(rows, pair_predictions)
    pooled_pair['polarity_pair_classes'] = _pair_class_metrics(rows, pair_predictions)
    pooled_pair['review_subsets'] = _group_accuracy(rows, pair_predictions)
    polarity_predictions = {
        name: [item for result in selected for item in result['predictions']['polarity'][name]]
        for name in ('roberta', 'deberta', 'mixed')
    }
    return {
        'fraction': fraction,
        'fold_mean_std': {
            'pair_micro_f1': _mean_std(pair_folds),
            'aspect_micro_f1': _mean_std(aspect_folds),
            'polarity_macro_f1': _mean_std(polarity_folds),
        },
        'pooled': {
            'pair': pooled_pair,
            'aspect': calculate_aspect_metrics(rows, aspect_predictions_values),
            'polarity': pooled_polarity,
        },
        'predictions': {
            'pair': [list(value) for value in pair_predictions],
            'aspect': [list(value) for value in aspect_predictions_values],
            'polarity': polarity_predictions,
        },
    }


def build_summary(
    rows: list[LabeledRow], results: Sequence[dict[str, Any]],
    bootstrap_samples: int, seed: int,
) -> dict[str, Any]:
    fractions = {fraction: _fraction_summary(rows, results, fraction) for fraction in FRACTIONS}
    comparisons = {}
    for left, right in zip(FRACTIONS, FRACTIONS[1:]):
        baseline = [tuple(value) for value in fractions[left]['predictions']['pair']]
        contender = [tuple(value) for value in fractions[right]['predictions']['pair']]
        interval = paired_pair_bootstrap(
            rows, contender, baseline, bootstrap_samples, seed + right,
        )
        left_score = float(fractions[left]['pooled']['pair']['source_of_truth_metrics']['overall']['micro']['f1'])
        right_score = float(fractions[right]['pooled']['pair']['source_of_truth_metrics']['overall']['micro']['f1'])
        delta = right_score - left_score
        comparisons[f'{left}-{right}'] = {
            'delta': delta, 'bootstrap': interval,
            'diagnosis': curve_diagnosis(delta, interval) if (left, right) == (75, 100) else None,
        }
    lower = fractions[75]
    upper = fractions[100]
    aspect_interval = paired_bootstrap_delta(
        rows,
        [tuple(value) for value in upper['predictions']['aspect']],
        [tuple(value) for value in lower['predictions']['aspect']],
        bootstrap_samples, seed + 175,
    )
    aspect_lower = float(lower['pooled']['aspect']['aspect']['micro']['f1'])
    aspect_upper = float(upper['pooled']['aspect']['aspect']['micro']['f1'])
    polarity_interval = paired_polarity_bootstrap(
        upper['predictions']['polarity']['mixed'],
        lower['predictions']['polarity']['mixed'],
        bootstrap_samples, seed + 275,
    )
    polarity_lower = float(lower['pooled']['polarity']['mixed']['macro_f1'])
    polarity_upper = float(upper['pooled']['polarity']['mixed']['macro_f1'])
    component_diagnoses = {
        'aspect': {
            'metric': 'micro_f1', 'delta': aspect_upper - aspect_lower,
            'bootstrap': aspect_interval,
            'diagnosis': curve_diagnosis(aspect_upper - aspect_lower, aspect_interval),
        },
        'polarity': {
            'metric': 'macro_f1', 'delta': polarity_upper - polarity_lower,
            'bootstrap': polarity_interval,
            'diagnosis': curve_diagnosis(polarity_upper - polarity_lower, polarity_interval),
        },
        'pair': comparisons['75-100'],
    }
    return {
        'fractions': {str(key): value for key, value in fractions.items()},
        'comparisons': comparisons,
        'primary_diagnosis': comparisons['75-100']['diagnosis'],
        'component_diagnoses': component_diagnoses,
        'decision_rule': {
            'meaningful_gain': PRIMARY_GAIN,
            'data_limited': 'delta >= 0.01 and lower_95 > 0',
            'plateaued': 'upper_95 < 0.01',
            'otherwise': 'inconclusive',
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    rows = []
    for fraction in FRACTIONS:
        value = summary['fractions'][str(fraction)]
        means = value['fold_mean_std']
        pooled = value['pooled']
        rows.append(
            f"| {fraction}% | {means['aspect_micro_f1']['mean']:.4f} +/- {means['aspect_micro_f1']['std']:.4f} | "
            f"{means['polarity_macro_f1']['mean']:.4f} +/- {means['polarity_macro_f1']['std']:.4f} | "
            f"{means['pair_micro_f1']['mean']:.4f} +/- {means['pair_micro_f1']['std']:.4f} | "
            f"{pooled['pair']['source_of_truth_metrics']['overall']['micro']['f1']:.4f} |"
        )
    comparison_rows = []
    for name, value in summary['comparisons'].items():
        interval = value['bootstrap']
        comparison_rows.append(
            f"| {name.replace('-', '% to ')}% | {value['delta']:+.4f} | "
            f"[{interval['lower_95']:+.4f}, {interval['upper_95']:+.4f}] |"
        )
    aspect_rows = []
    polarity_rows = []
    pair_rows = []
    for fraction in FRACTIONS:
        pooled = summary['fractions'][str(fraction)]['pooled']
        aspect = pooled['aspect']
        aspect_rows.append(
            f"| {fraction}% | {aspect['aspect']['macro']['f1']:.4f} | "
            f"{aspect['exact_set_accuracy']:.4f} | {aspect['subsets']['multi_aspect']['exact_set_accuracy']:.4f} | "
            f"{aspect['aspect']['classes']['anecdotes/miscellaneous']['f1']:.4f} | "
            f"{aspect['cardinality']['gold_mean']:.3f} | {aspect['cardinality']['predicted_mean']:.3f} |"
        )
        polarity = pooled['polarity']['mixed']
        polarity_rows.append(
            f"| {fraction}% | {polarity['accuracy']:.4f} | {polarity['macro_f1']:.4f} | "
            + ' | '.join(f"{polarity['classes'][label]['f1']:.4f}" for label in POLARITIES) + ' |'
        )
        pair = pooled['pair']
        pair_rows.append(
            f"| {fraction}% | {pair['supplemental_exact_match_accuracy']['overall']:.4f} | "
            f"{pair['review_subsets']['single']['exact_set_accuracy']:.4f} | "
            f"{pair['review_subsets']['multi']['exact_set_accuracy']:.4f} | "
            + ' | '.join(f"{pair['polarity_pair_classes'][label]['f1']:.4f}" for label in POLARITIES) + ' |'
        )
    diagnosis_rows = []
    for name in ('aspect', 'polarity', 'pair'):
        value = summary['component_diagnoses'][name]
        interval = value['bootstrap']
        diagnosis_rows.append(
            f"| {name} | {value['metric'] if 'metric' in value else 'micro_f1'} | "
            f"{value['delta']:+.4f} | [{interval['lower_95']:+.4f}, {interval['upper_95']:+.4f}] | "
            f"{value['diagnosis']} |"
        )
    return f'''# ABSA learning-curve diagnostic

This is five-fold grouped out-of-fold evaluation on the original training partition.
The historical validation and test partitions were not used.

| Training data | Aspect micro-F1 | Polarity macro-F1 | Pair micro-F1 | Pooled pair F1 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Aspect detail

| Data | Macro-F1 | Exact set | Multi exact | Misc F1 | Gold cardinality | Predicted cardinality |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(aspect_rows)}

## Gold-aspect polarity detail

| Data | Accuracy | Macro-F1 | Positive F1 | Negative F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(polarity_rows)}

## Composed-pair detail

| Data | Exact set | Single exact | Multi exact | Positive F1 | Negative F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(pair_rows)}

| Comparison | Pair-F1 delta | Paired review bootstrap 95% CI |
|---|---:|---:|
{chr(10).join(comparison_rows)}

## 75% to 100% diagnosis

| Component | Metric | Delta | Paired bootstrap 95% CI | Diagnosis |
|---|---|---:|---:|---|
{chr(10).join(diagnosis_rows)}

Primary 75% to 100% diagnosis: **{summary['primary_diagnosis']}**.

Data-limited requires a gain of at least 0.01 with the interval excluding zero.
Plateaued means the interval's upper bound is below 0.01. Otherwise the result is inconclusive.
'''


def render_curve_csv(summary: dict[str, Any]) -> str:
    lines = ['fraction,aspect_micro_f1,polarity_macro_f1,pair_micro_f1']
    for fraction in FRACTIONS:
        pooled = summary['fractions'][str(fraction)]['pooled']
        lines.append(','.join((
            str(fraction),
            f"{pooled['aspect']['aspect']['micro']['f1']:.8f}",
            f"{pooled['polarity']['mixed']['macro_f1']:.8f}",
            f"{pooled['pair']['source_of_truth_metrics']['overall']['micro']['f1']:.8f}",
        )))
    return '\n'.join(lines) + '\n'


def render_curve_svg(summary: dict[str, Any]) -> str:
    width, height = 720, 430
    left, top, plot_width, plot_height = 70, 35, 610, 320
    colors = {'aspect': '#2563eb', 'polarity': '#d97706', 'pair': '#059669'}
    series = {}
    for name in colors:
        values = []
        for fraction in FRACTIONS:
            pooled = summary['fractions'][str(fraction)]['pooled']
            if name == 'aspect':
                score = pooled['aspect']['aspect']['micro']['f1']
            elif name == 'polarity':
                score = pooled['polarity']['mixed']['macro_f1']
            else:
                score = pooled['pair']['source_of_truth_metrics']['overall']['micro']['f1']
            x = left + (fraction - 25) / 75 * plot_width
            y = top + (1 - float(score)) * plot_height
            values.append((x, y))
        series[name] = values
    grid = []
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + (1 - value) * plot_height
        grid.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_width}" y2="{y}" stroke="#e5e7eb"/>')
        grid.append(f'<text x="{left - 12}" y="{y + 4}" text-anchor="end">{value:.2f}</text>')
    curves = []
    for name, points in series.items():
        encoded = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
        curves.append(f'<polyline points="{encoded}" fill="none" stroke="{colors[name]}" stroke-width="3"/>')
        curves.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[name]}"/>' for x, y in points)
    xlabels = [f'<text x="{left + (value - 25) / 75 * plot_width:.1f}" y="{top + plot_height + 28}" text-anchor="middle">{value}%</text>' for value in FRACTIONS]
    legend = [f'<rect x="{left + index * 160}" y="390" width="16" height="4" fill="{colors[name]}"/><text x="{left + 23 + index * 160}" y="396">{name}</text>' for index, name in enumerate(colors)]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="12" fill="#111827">
<text x="{width / 2}" y="20" text-anchor="middle" font-size="16" font-weight="bold">ABSA learning curves</text>
{''.join(grid)}<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>
{''.join(curves)}{''.join(xlabels)}{''.join(legend)}</g></svg>\n'''


def generate_report(config: LearningCurveConfig) -> dict[str, Any]:
    rows = read_labeled_csv(config.input)
    results = []
    missing = []
    for fold in range(1, config.folds + 1):
        for fraction in FRACTIONS:
            path = config.run_dir / f'fold-{fold}' / f'fraction-{fraction}' / 'result.json'
            if not path.is_file():
                missing.append(str(path))
            else:
                value = json.loads(path.read_text(encoding='utf-8'))
                if not value.get('finished'):
                    missing.append(str(path))
                results.append(value)
    if missing:
        raise RuntimeError(f'{len(missing)} fold/fraction results are missing or incomplete')
    summary = build_summary(rows, results, config.bootstrap_samples, config.seed)
    atomic_write_json(config.run_dir / 'summary.json', summary)
    atomic_write_text(config.run_dir / 'report.md', render_report(summary))
    atomic_write_text(config.run_dir / 'learning_curve.csv', render_curve_csv(summary))
    atomic_write_text(config.run_dir / 'learning_curve.svg', render_curve_svg(summary))
    return summary


def run_experiment(config: LearningCurveConfig) -> dict[str, Any]:
    import mlflow
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError('The full learning curve requires CUDA')
    rows = read_labeled_csv(config.input)
    plan = learning_curve_splits(rows, config.folds, config.selection_folds, config.seed)
    validate_split_plan(rows, plan)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'config': _serialized_config(config), 'input_sha256': file_sha256(config.input),
        'fractions': list(FRACTIONS), 'training_count': config.folds * len(FRACTIONS) * 3,
    }
    _write_or_validate(config.run_dir / 'manifest.json', manifest)
    _write_or_validate(config.run_dir / 'splits.json', {'folds': serialized_split_plan(plan)})
    with RunLock(config.run_dir / '.experiment.lock'):
        _prepare_mlflow_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        roberta_tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.roberta_model, use_fast=True,
        )
        deberta_tokenizer = load_tokenizer(transformers, config.deberta_model)
        for fold in plan:
            for fraction in FRACTIONS:
                _run_fraction(
                    torch, transformers, mlflow, roberta_tokenizer, deberta_tokenizer,
                    config, rows, fold, fraction,
                )
    summary = generate_report(config)
    atomic_write_text(
        config.run_dir / 'completed_at.txt',
        datetime.now(timezone.utc).isoformat() + '\n',
    )
    return summary


def _benchmark_stage(
    torch: Any, transformers: Any, tokenizer: Any, model_name: str,
    dataset: Any, multilabel: bool, seconds: float, config: LearningCurveConfig,
) -> dict[str, float | int | str]:
    task = 'aspect' if multilabel else 'polarity'
    LOGGER.info('Loading %s for %s benchmark', model_name, task)
    device = torch.device('cuda')
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(ASPECTS) if multilabel else len(POLARITIES),
    ).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    use_fp16, autocast_dtype = _precision_settings(torch, config.mixed_precision, device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True,
        collate_fn=transformers.DataCollatorWithPadding(tokenizer=tokenizer),
    )
    started = time.monotonic()
    examples = batches = 0
    progress = tqdm(total=seconds, desc=f'Benchmark {task} {model_name}', unit='s', dynamic_ncols=True)
    last_elapsed = 0.0
    try:
        while time.monotonic() - started < seconds:
            for batch in loader:
                targets = batch.pop('labels').to(device)
                with torch.autocast(
                    device_type='cuda', dtype=autocast_dtype,
                    enabled=autocast_dtype is not None,
                ):
                    logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
                    loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, targets.float(),
                    ) if multilabel else torch.nn.functional.cross_entropy(logits, targets))
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                examples += int(targets.shape[0]); batches += 1
                elapsed = min(time.monotonic() - started, seconds)
                progress.update(elapsed - last_elapsed)
                progress.set_postfix(examples=examples, batches=batches)
                last_elapsed = elapsed
                if time.monotonic() - started >= seconds:
                    break
    finally:
        progress.close()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    peak = torch.cuda.max_memory_reserved() / 1024 ** 3
    del model, optimizer
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return {
        'model': model_name, 'task': task,
        'elapsed_seconds': elapsed, 'examples': examples, 'batches': batches,
        'examples_per_second': examples / elapsed, 'peak_reserved_gib': peak,
    }


def run_benchmark(config: LearningCurveConfig) -> dict[str, Any]:
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError('The throughput benchmark requires CUDA')
    LOGGER.info('Starting %.1f-minute three-stage GPU calibration', config.benchmark_minutes)
    rows = read_labeled_csv(config.input)
    plan = learning_curve_splits(rows, config.folds, config.selection_folds, config.seed)
    fit_rows = rows_for_ids(rows, plan[0]['fit_ids'])
    roberta_tokenizer = transformers.AutoTokenizer.from_pretrained(config.roberta_model, use_fast=True)
    deberta_tokenizer = load_tokenizer(transformers, config.deberta_model)
    aspect_dataset = _attach_tokenizer(
        AspectDataset(group_aspect_targets(fit_rows), roberta_tokenizer, config.max_length),
        roberta_tokenizer,
    )
    roberta_polarity = _attach_tokenizer(
        PolarityDataset(fit_rows, roberta_tokenizer, config.max_length), roberta_tokenizer,
    )
    deberta_polarity = _attach_tokenizer(
        PolarityDataset(fit_rows, deberta_tokenizer, config.max_length), deberta_tokenizer,
    )
    seconds = config.benchmark_minutes * 60 / 3
    stages = [
        _benchmark_stage(torch, transformers, roberta_tokenizer, config.roberta_model,
                         aspect_dataset, True, seconds, config),
        _benchmark_stage(torch, transformers, roberta_tokenizer, config.roberta_model,
                         roberta_polarity, False, seconds, config),
        _benchmark_stage(torch, transformers, deberta_tokenizer, config.deberta_model,
                         deberta_polarity, False, seconds, config),
    ]
    per_task = {(value['task'], value['model']): float(value['examples_per_second']) for value in stages}
    projected = 0.0
    for fold in plan:
        for fraction in FRACTIONS:
            subset_rows = rows_for_ids(rows, fold['subsets'][fraction])
            review_count = len(group_aspect_targets(subset_rows))
            projected += config.epochs * review_count / per_task[('aspect', config.roberta_model)]
            projected += config.epochs * len(subset_rows) / per_task[('polarity', config.roberta_model)]
            projected += config.epochs * len(subset_rows) / per_task[('polarity', config.deberta_model)]
    result = {
        'benchmark_minutes': config.benchmark_minutes, 'stages': stages,
        'projected_max_epoch_training_seconds': projected,
        'projected_max_epoch_training_hours': projected / 3600,
        'note': 'Projection excludes validation/inference and assumes no early stopping.',
    }
    config.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config.run_dir / 'benchmark.json', result)
    LOGGER.info('Benchmark complete; projected maximum training time %.2f hours', projected / 3600)
    return result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    mode, config = config_from_args(argv)
    if mode == 'run':
        run_experiment(config)
    elif mode == 'benchmark':
        run_benchmark(config)
    else:
        generate_report(config)
    return 0
