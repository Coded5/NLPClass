from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .absa_training import (
    AspectDataset,
    ExperimentConfig as TrainingConfig,
    _attach_tokenizer,
    _atomic_torch_save,
    _predict_logits,
    decode_aspects,
    group_aspect_targets,
    train_stage,
    tune_thresholds,
)
from .checkpoint import RunLock
from .data import LabeledRow, read_labeled_csv
from .evaluation import read_predictions
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .labels import ASPECTS
from .roberta_training import (
    _prepare_mlflow_tracking_uri,
    _seed_everything,
    _write_labeled_csv,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AspectExperimentConfig:
    input: Path = Path('data/contest2_train.csv')
    split_dir: Path = Path('artifacts/training/roberta-aspect-exp1/splits')
    run_dir: Path = Path('artifacts/experiments/multilabel-aspect-v1')
    existing_multilabel_dir: Path = Path(
        'artifacts/experiments/multilabel-conditioned-v1'
    )
    model: str = 'FacebookAI/roberta-base'
    seeds: tuple[int, ...] = (42, 43, 44)
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
    mlflow_tracking_uri: str = 'sqlite:///artifacts/mlflow.db'
    mlflow_experiment: str = 'contest2-roberta-multilabel-aspect-v1'
    bootstrap_samples: int = 10_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train and evaluate a three-seed multilabel aspect ensemble.'
    )
    for name, field in AspectExperimentConfig.__dataclass_fields__.items():
        default = field.default
        option = f'--{name.replace("_", "-")}'
        if name == 'seeds':
            parser.add_argument(option, type=int, nargs='+', default=list(default))
        elif isinstance(default, Path):
            parser.add_argument(option, type=Path, default=default)
        elif isinstance(default, str):
            parser.add_argument(option, default=default)
        else:
            parser.add_argument(option, type=type(default), default=default)
    return parser


def config_from_args(argv: list[str] | None = None) -> AspectExperimentConfig:
    values = vars(build_parser().parse_args(argv))
    values['seeds'] = tuple(values['seeds'])
    config = AspectExperimentConfig(**values)
    validate_config(config)
    return config


def validate_config(config: AspectExperimentConfig) -> None:
    positive = {
        'epochs': config.epochs,
        'evaluation-interval': config.evaluation_interval,
        'early-stopping-patience': config.early_stopping_patience,
        'max-length': config.max_length,
        'train-batch-size': config.train_batch_size,
        'eval-batch-size': config.eval_batch_size,
        'gradient-accumulation-steps': config.gradient_accumulation_steps,
        'log-steps': config.log_steps,
        'bootstrap-samples': config.bootstrap_samples,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f'{", ".join(invalid)} must be at least 1')
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError('seeds must contain at least one unique value')
    if config.learning_rate <= 0 or config.early_stopping_min_delta < 0:
        raise ValueError('learning-rate must be positive and early-stopping-min-delta nonnegative')
    if config.checkpoint_steps < 0 or config.weight_decay < 0:
        raise ValueError('checkpoint-steps and weight-decay must be nonnegative')
    if not 0 <= config.warmup_ratio < 1:
        raise ValueError('warmup-ratio must be in [0, 1)')


def mean_probabilities(
    values: Sequence[Sequence[Sequence[float]]],
) -> list[list[float]]:
    if not values:
        raise ValueError('At least one probability matrix is required')
    row_count = len(values[0])
    if any(len(matrix) != row_count for matrix in values):
        raise ValueError('Probability matrices have different row counts')
    result: list[list[float]] = []
    for row_index in range(row_count):
        label_count = len(values[0][row_index])
        if any(len(matrix[row_index]) != label_count for matrix in values):
            raise ValueError('Probability matrices have different label counts')
        result.append([
            sum(matrix[row_index][label] for matrix in values) / len(values)
            for label in range(label_count)
        ])
    return result


def aspect_predictions(
    rows: Sequence[LabeledRow],
    probabilities: Sequence[Sequence[float]],
    thresholds: Sequence[float],
) -> list[tuple[str, str]]:
    reviews = group_aspect_targets(rows)
    if len(reviews) != len(probabilities):
        raise ValueError('Probability rows do not match grouped reviews')
    decoded = decode_aspects(probabilities, thresholds)
    return [
        (review.item_id, ASPECTS[aspect])
        for review, aspects in zip(reviews, decoded)
        for aspect in aspects
    ]


def calculate_aspect_metrics(
    gold_rows: Iterable[LabeledRow],
    predictions: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    gold_rows = list(gold_rows)
    gold = {(row.item_id, row.aspect) for row in gold_rows}
    predicted = set(predictions)
    class_reports: dict[str, dict[str, float | int]] = {}
    for label in ASPECTS:
        class_gold = {value for value in gold if value[1] == label}
        class_predicted = {value for value in predicted if value[1] == label}
        report = _prf(class_gold, class_predicted)
        report['support'] = len(class_gold)
        class_reports[label] = report
    macro = {
        metric: sum(float(report[metric]) for report in class_reports.values()) / len(ASPECTS)
        for metric in ('precision', 'recall', 'f1')
    }
    macro['support'] = sum(int(report['support']) for report in class_reports.values())
    micro = _prf(gold, predicted)
    micro['support'] = len(gold)

    gold_by_id = _aspects_by_id(gold)
    predicted_by_id = _aspects_by_id(predicted)
    expected_ids = set(gold_by_id)
    predicted_ids = set(predicted_by_id)
    all_ids = expected_ids | predicted_ids
    exact = _exact_accuracy(gold_by_id, predicted_by_id, all_ids)
    single_ids = {item_id for item_id, labels in gold_by_id.items() if len(labels) == 1}
    multi_ids = {item_id for item_id, labels in gold_by_id.items() if len(labels) > 1}
    return {
        'aspect': {'classes': class_reports, 'macro': macro, 'micro': micro},
        'exact_set_accuracy': exact,
        'subsets': {
            'single_aspect': {
                'ids': len(single_ids),
                'exact_set_accuracy': _exact_accuracy(
                    gold_by_id, predicted_by_id, single_ids
                ),
            },
            'multi_aspect': {
                'ids': len(multi_ids),
                'exact_set_accuracy': _exact_accuracy(
                    gold_by_id, predicted_by_id, multi_ids
                ),
            },
        },
        'coverage': {
            'expected_ids': len(expected_ids),
            'predicted_ids': len(predicted_ids),
            'covered_expected_ids': len(expected_ids & predicted_ids),
            'missing_ids': len(expected_ids - predicted_ids),
            'unexpected_ids': len(predicted_ids - expected_ids),
            'ratio': len(expected_ids & predicted_ids) / len(expected_ids)
            if expected_ids else 0.0,
        },
        'cardinality': {
            'gold_mean': len(gold) / len(expected_ids) if expected_ids else 0.0,
            'predicted_mean': len(predicted) / len(predicted_ids) if predicted_ids else 0.0,
        },
    }


def paired_bootstrap_delta(
    gold_rows: Sequence[LabeledRow],
    contender: Sequence[tuple[str, str]],
    baseline: Sequence[tuple[str, str]],
    samples: int,
    seed: int = 20260910,
) -> dict[str, float | int]:
    gold = _aspects_by_id((row.item_id, row.aspect) for row in gold_rows)
    left = _aspects_by_id(contender)
    right = _aspects_by_id(baseline)
    item_ids = sorted(gold)
    if not item_ids:
        raise ValueError('Cannot bootstrap an empty gold set')
    left_counts = {
        item_id: _counts(gold[item_id], left.get(item_id, set()))
        for item_id in item_ids
    }
    right_counts = {
        item_id: _counts(gold[item_id], right.get(item_id, set()))
        for item_id in item_ids
    }
    randomizer = random.Random(seed)
    deltas = []
    for _ in range(samples):
        selected = randomizer.choices(item_ids, k=len(item_ids))
        deltas.append(
            _f1_from_counts(left_counts[item_id] for item_id in selected)
            - _f1_from_counts(right_counts[item_id] for item_id in selected)
        )
    ordered = sorted(deltas)
    return {
        'samples': samples,
        'mean': statistics.fmean(deltas),
        'lower_95': ordered[int(0.025 * (samples - 1))],
        'upper_95': ordered[int(0.975 * (samples - 1))],
    }


def _prf(gold: set[tuple[str, str]], predicted: set[tuple[str, str]]) -> dict[str, float]:
    correct = len(gold & predicted)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def _aspects_by_id(
    values: Iterable[tuple[str, str]],
) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for item_id, aspect in values:
        grouped[item_id].add(aspect)
    return grouped


def _exact_accuracy(
    gold: dict[str, set[str]],
    predicted: dict[str, set[str]],
    item_ids: set[str],
) -> float:
    if not item_ids:
        return 0.0
    return sum(
        gold.get(item_id, set()) == predicted.get(item_id, set())
        for item_id in item_ids
    ) / len(item_ids)


def _counts(gold: set[str], predicted: set[str]) -> tuple[int, int, int]:
    return len(gold & predicted), len(predicted - gold), len(gold - predicted)


def _f1_from_counts(values: Iterable[tuple[int, int, int]]) -> float:
    true_positive = false_positive = false_negative = 0
    for correct, extra, missing in values:
        true_positive += correct
        false_positive += extra
        false_negative += missing
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def _write_predictions(path: Path, predictions: Iterable[tuple[str, str]]) -> None:
    rows = ['id,aspectCategory']
    rows.extend(
        f'{_csv_value(item_id)},{_csv_value(aspect)}'
        for item_id, aspect in predictions
    )
    atomic_write_text(path, '\n'.join(rows) + '\n')


def _csv_value(value: str) -> str:
    from io import StringIO

    output = StringIO()
    csv.writer(output, lineterminator='').writerow([value])
    return output.getvalue()


def _load_reference_predictions(path: Path) -> list[tuple[str, str]]:
    return [(item_id, aspect) for item_id, aspect, _ in read_predictions(path)]


def _training_config(
    config: AspectExperimentConfig, seed: int, run_dir: Path
) -> TrainingConfig:
    return TrainingConfig(
        input=config.input,
        split_dir=config.split_dir,
        run_dir=run_dir,
        model=config.model,
        epochs=config.epochs,
        evaluation_interval=config.evaluation_interval,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        seed=seed,
        max_length=config.max_length,
        learning_rate=config.learning_rate,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        checkpoint_steps=config.checkpoint_steps,
        log_steps=config.log_steps,
        mixed_precision=config.mixed_precision,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        mlflow_experiment=config.mlflow_experiment,
        mlflow_run_name=f'multilabel-aspect-seed-{seed}',
    )


def _scratch_seed_dir(config: AspectExperimentConfig, seed: int) -> Path:
    return Path('/tmp') / f'contest2-{config.run_dir.name}' / 'runs' / f'seed-{seed}'


def _export_compact_checkpoint(
    torch: Any,
    source: Path,
    destination: Path,
) -> None:
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    compact_model = {
        name: value.to(dtype=torch.float16) if value.is_floating_point() else value
        for name, value in checkpoint['model'].items()
    }
    _atomic_torch_save(torch, {
        **{key: value for key, value in checkpoint.items() if key != 'model'},
        'model': compact_model,
        'storage_dtype': 'float16',
    }, destination)


def _predict_checkpoint_probabilities(
    torch: Any,
    transformers: Any,
    tokenizer: Any,
    config: AspectExperimentConfig,
    checkpoint_path: Path,
    rows: Sequence[LabeledRow],
    label: str,
) -> list[list[float]]:
    reviews = group_aspect_targets(rows)
    dataset = _attach_tokenizer(
        AspectDataset(reviews, tokenizer, config.max_length), tokenizer
    )
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    label2id = {name: index for index, name in enumerate(ASPECTS)}
    id2label = {index: name for name, index in label2id.items()}
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        config.model,
        num_labels=len(ASPECTS),
        label2id=label2id,
        id2label=id2label,
    )
    model.load_state_dict(checkpoint['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logits = _predict_logits(
        torch, transformers, model, dataset, config.eval_batch_size, label
    )
    probabilities = torch.sigmoid(logits).tolist()
    del model, logits, checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probabilities


def _ensemble_probabilities(
    torch: Any,
    transformers: Any,
    tokenizer: Any,
    config: AspectExperimentConfig,
    checkpoints: Sequence[Path],
    rows: Sequence[LabeledRow],
    split_name: str,
) -> list[list[float]]:
    matrices = [
        _predict_checkpoint_probabilities(
            torch, transformers, tokenizer, config, checkpoint, rows,
            f'{split_name} seed {seed}',
        )
        for seed, checkpoint in zip(config.seeds, checkpoints)
    ]
    return mean_probabilities(matrices)


def _target_vectors(rows: Sequence[LabeledRow]) -> list[list[int]]:
    return [
        [int(index in review.aspects) for index in range(len(ASPECTS))]
        for review in group_aspect_targets(rows)
    ]


def _serialized_config(config: AspectExperimentConfig) -> dict[str, Any]:
    values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    values['seeds'] = list(config.seeds)
    return values


def _manifest(config: AspectExperimentConfig) -> dict[str, Any]:
    return {
        **_serialized_config(config),
        'input_sha256': file_sha256(config.input),
        'split_sha256': {
            name: file_sha256(config.split_dir / f'{name}.csv')
            for name in ('train', 'eval', 'test')
        },
        'labels': list(ASPECTS),
    }


def _write_or_validate_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing != manifest:
            raise RuntimeError('Run manifest differs from the requested experiment configuration')
    else:
        atomic_write_json(path, manifest)


def _validate_training_splits(
    config: AspectExperimentConfig,
) -> tuple[list[LabeledRow], list[LabeledRow]]:
    train_rows = read_labeled_csv(config.split_dir / 'train.csv')
    validation_rows = read_labeled_csv(config.split_dir / 'eval.csv')
    train_ids = {row.item_id for row in train_rows}
    validation_ids = {row.item_id for row in validation_rows}
    if train_ids & validation_ids:
        raise RuntimeError('Train and validation IDs overlap')
    return train_rows, validation_rows


def _load_and_validate_test_split(
    config: AspectExperimentConfig,
    train_rows: Sequence[LabeledRow],
    validation_rows: Sequence[LabeledRow],
) -> list[LabeledRow]:
    test_rows = read_labeled_csv(config.split_dir / 'test.csv')
    source_rows = read_labeled_csv(config.input)
    id_sets = [
        {row.item_id for row in values}
        for values in (train_rows, validation_rows, test_rows)
    ]
    if id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2]:
        raise RuntimeError('Test IDs overlap train or validation IDs')
    if set().union(*id_sets) != {row.item_id for row in source_rows}:
        raise RuntimeError('Persisted splits do not cover the source dataset')
    return test_rows


def _write_evaluation(
    output_dir: Path,
    rows: Sequence[LabeledRow],
    predictions: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    _write_labeled_csv(output_dir / 'gold.csv', rows)
    _write_predictions(output_dir / 'predictions.csv', predictions)
    metrics = calculate_aspect_metrics(rows, predictions)
    atomic_write_json(output_dir / 'metrics.json', metrics)
    return metrics


def _write_report(
    config: AspectExperimentConfig,
    selections: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
    validation: dict[str, Any],
    systems: dict[str, dict[str, Any]],
    bootstrap: dict[str, Any],
    runtime_seconds: float,
) -> None:
    seed_rows = '\n'.join(
        f"| {seed} | {selection['epoch']} | {selection['score']:.6f} |"
        for seed, selection in zip(config.seeds, selections)
    )
    system_rows = '\n'.join(
        '| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |'.format(
            name,
            metrics['aspect']['micro']['f1'],
            metrics['aspect']['macro']['f1'],
            metrics['exact_set_accuracy'],
            metrics['subsets']['multi_aspect']['exact_set_accuracy'],
        )
        for name, metrics in systems.items()
    )
    report = f'''# Multi-label aspect experiment

## Validation selection

| Seed | Best epoch | Validation micro-F1 |
|---:|---:|---:|
{seed_rows}

Three-seed mean: {statistics.fmean(float(value['score']) for value in selections):.6f}.
Three-seed sample standard deviation: {statistics.stdev(float(value['score']) for value in selections) if len(selections) > 1 else 0.0:.6f}.
Ensemble validation micro-F1: {validation['aspect']['micro']['f1']:.6f}.
Locked ensemble thresholds: `{json.dumps(list(thresholds))}`.

## Held-out test comparison

| System | Aspect micro-F1 | Aspect macro-F1 | Exact-set accuracy | Multi-aspect exact accuracy |
|---|---:|---:|---:|---:|
{system_rows}

Ensemble minus legacy baseline paired-bootstrap delta: {bootstrap['mean']:+.6f}
(95% CI {bootstrap['lower_95']:+.6f} to {bootstrap['upper_95']:+.6f};
{bootstrap['samples']} ID-grouped samples).

Checkpoint and threshold selection used validation data only. The held-out test split
has been evaluated by earlier repository experiments, so it is not described as globally untouched.

Runtime: {runtime_seconds / 60:.1f} minutes.
'''
    atomic_write_text(config.run_dir / 'report.md', report)


def run_experiment(config: AspectExperimentConfig) -> None:
    import mlflow
    import torch
    import transformers

    started = time.monotonic()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(config.run_dir / '.experiment.lock'):
        manifest = _manifest(config)
        _write_or_validate_manifest(config.run_dir / 'manifest.json', manifest)
        atomic_write_json(config.run_dir / 'config.json', _serialized_config(config))
        completed_path = config.run_dir / 'results.json'
        if completed_path.is_file():
            LOGGER.info('Experiment is already complete: %s', completed_path)
            return

        train_rows, validation_rows = _validate_training_splits(config)
        _prepare_mlflow_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, use_fast=True)

        selections = []
        checkpoint_paths = []
        for seed in config.seeds:
            artifact_seed_dir = config.run_dir / 'runs' / f'seed-{seed}'
            scratch_seed_dir = _scratch_seed_dir(config, seed)
            compact_checkpoint = artifact_seed_dir / 'best' / 'aspect.pt'
            selection_path = artifact_seed_dir / 'selection.json'
            if compact_checkpoint.is_file() and selection_path.is_file():
                LOGGER.info('Seed %d is already complete', seed)
                selections.append(json.loads(selection_path.read_text(encoding='utf-8')))
                checkpoint_paths.append(compact_checkpoint)
                continue

            training_config = _training_config(config, seed, scratch_seed_dir)
            _seed_everything(torch, seed)
            run_id_path = artifact_seed_dir / 'mlflow_run.json'
            run_data = json.loads(run_id_path.read_text()) if run_id_path.is_file() else {}
            start_kwargs = (
                {'run_id': run_data['run_id']}
                if run_data else {'run_name': training_config.mlflow_run_name}
            )
            with mlflow.start_run(**start_kwargs) as active_run:
                if not run_data:
                    atomic_write_json(run_id_path, {'run_id': active_run.info.run_id})
                    mlflow.log_params({
                        **_serialized_config(config),
                        'seed': seed,
                    })
                LOGGER.info('Starting multi-label aspect seed %d', seed)
                selection = train_stage(
                    torch, transformers, mlflow, tokenizer, training_config,
                    'aspect', train_rows, validation_rows,
                )
            _export_compact_checkpoint(
                torch, scratch_seed_dir / 'best' / 'aspect.pt', compact_checkpoint
            )
            atomic_write_json(selection_path, selection)
            shutil.rmtree(scratch_seed_dir)
            selections.append(selection)
            checkpoint_paths.append(compact_checkpoint)

        validation_probabilities = _ensemble_probabilities(
            torch, transformers, tokenizer, config, checkpoint_paths,
            validation_rows, 'Validate ensemble',
        )
        thresholds = tune_thresholds(
            validation_probabilities, _target_vectors(validation_rows)
        )
        validation_predictions = aspect_predictions(
            validation_rows, validation_probabilities, thresholds
        )
        validation_metrics = _write_evaluation(
            config.run_dir / 'ensemble' / 'validation',
            validation_rows, validation_predictions,
        )
        locked = {
            'seeds': list(config.seeds),
            'checkpoints': [str(path) for path in checkpoint_paths],
            'seed_selections': selections,
            'thresholds': thresholds,
            'ensemble_validation_micro_f1': validation_metrics['aspect']['micro']['f1'],
            'test_accessed': False,
        }
        atomic_write_json(config.run_dir / 'locked_selection.json', locked)

        # Test labels are loaded only after checkpoints and thresholds are locked.
        test_rows = _load_and_validate_test_split(
            config, train_rows, validation_rows
        )
        test_probabilities = _ensemble_probabilities(
            torch, transformers, tokenizer, config, checkpoint_paths,
            test_rows, 'Test ensemble',
        )
        ensemble_predictions = aspect_predictions(
            test_rows, test_probabilities, thresholds
        )
        ensemble_metrics = _write_evaluation(
            config.run_dir / 'ensemble' / 'test', test_rows, ensemble_predictions
        )

        legacy_predictions = _load_reference_predictions(
            config.existing_multilabel_dir / 'baseline-test' / 'predictions.csv'
        )
        prior_predictions = _load_reference_predictions(
            config.existing_multilabel_dir / 'test' / 'predictions.csv'
        )
        systems = {
            'Three-seed ensemble': ensemble_metrics,
            'Prior one-seed multi-label': calculate_aspect_metrics(
                test_rows, prior_predictions
            ),
            'Legacy single-label': calculate_aspect_metrics(
                test_rows, legacy_predictions
            ),
        }
        bootstrap = paired_bootstrap_delta(
            test_rows, ensemble_predictions, legacy_predictions,
            config.bootstrap_samples,
        )
        runtime_seconds = time.monotonic() - started
        results = {
            'validation': validation_metrics,
            'test_systems': systems,
            'bootstrap_ensemble_minus_legacy': bootstrap,
            'runtime_seconds': runtime_seconds,
        }
        atomic_write_json(config.run_dir / 'results.json', results)
        _write_report(
            config, selections, thresholds, validation_metrics, systems,
            bootstrap, runtime_seconds,
        )
        locked['test_accessed'] = True
        atomic_write_json(config.run_dir / 'locked_selection.json', locked)

        summary_path = config.run_dir / 'ensemble_mlflow_run.json'
        summary_data = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        start_kwargs = (
            {'run_id': summary_data['run_id']}
            if summary_data else {'run_name': 'multilabel-aspect-three-seed-ensemble'}
        )
        with mlflow.start_run(**start_kwargs) as active_run:
            if not summary_data:
                atomic_write_json(summary_path, {'run_id': active_run.info.run_id})
            mlflow.log_metric(
                'validation/aspect_micro_f1',
                validation_metrics['aspect']['micro']['f1'],
            )
            mlflow.log_metric(
                'test/aspect_micro_f1', ensemble_metrics['aspect']['micro']['f1']
            )
            mlflow.log_metric(
                'test/aspect_exact_set_accuracy',
                ensemble_metrics['exact_set_accuracy'],
            )
            mlflow.log_artifact(str(config.run_dir / 'report.md'))
            mlflow.log_artifact(str(config.run_dir / 'results.json'))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    try:
        run_experiment(config_from_args(argv))
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted; rerun the same command to resume')
        return 130
    except Exception:
        LOGGER.exception('Experiment failed')
        return 1
    return 0
