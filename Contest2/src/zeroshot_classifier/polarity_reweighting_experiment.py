from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .absa_composition_experiment import _write_evaluation
from .absa_training import group_aspect_targets
from .checkpoint import RunLock
from .data import LabeledRow, read_labeled_csv
from .imbalance_losses import LossSpec, loss_weights, polarity_counts
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .joint_absa_experiment import (
    CONFIRMATION_SEEDS,
    JointExperimentConfig,
    _mean_logits,
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
    _ensemble_probabilities,
)


LOGGER = logging.getLogger(__name__)
WEIGHT_GRID = (
    (1.25, 1.5),
    (1.5, 2.0),
    (2.0, 2.0),
    (1.5, 3.0),
    (2.0, 3.0),
)


@dataclass(frozen=True)
class ReweightingConfig:
    input: Path = Path('data/contest2_train.csv')
    split_dir: Path = Path('artifacts/training/roberta-aspect-exp1/splits')
    aspect_run_dir: Path = Path('artifacts/experiments/multilabel-aspect-v1')
    baseline_run_dir: Path = Path('artifacts/experiments/absa-imbalance-joint-v1')
    run_dir: Path = Path('artifacts/experiments/polarity-reweighting-v1')
    optimizer_dir: Path = Path('/tmp/contest2-polarity-reweighting-v1')
    evaluator: Path = Path('scripts/evaluate.py')
    model: str = 'FacebookAI/roberta-base'
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
    log_steps: int = 10
    mixed_precision: str = 'auto'
    pair_f1_tolerance: float = 0.005
    mlflow_tracking_uri: str = (
        'sqlite:////home/kami/Projects/NLP/Contest2/artifacts/mlflow.db'
    )
    mlflow_experiment: str = 'contest2-polarity-reweighting-v1'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Screen and confirm extra neutral/conflict polarity weights.'
    )
    for name, field in ReweightingConfig.__dataclass_fields__.items():
        default = field.default
        option = f'--{name.replace("_", "-")}'
        if isinstance(default, Path):
            parser.add_argument(option, type=Path, default=default)
        elif isinstance(default, str):
            parser.add_argument(option, default=default)
        else:
            parser.add_argument(option, type=type(default), default=default)
    return parser


def config_from_args(argv: list[str] | None = None) -> ReweightingConfig:
    config = ReweightingConfig(**vars(build_parser().parse_args(argv)))
    positive = {
        'epochs': config.epochs,
        'evaluation-interval': config.evaluation_interval,
        'early-stopping-patience': config.early_stopping_patience,
        'max-length': config.max_length,
        'train-batch-size': config.train_batch_size,
        'eval-batch-size': config.eval_batch_size,
        'gradient-accumulation-steps': config.gradient_accumulation_steps,
        'log-steps': config.log_steps,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f'{", ".join(invalid)} must be at least 1')
    if not 0 <= config.pair_f1_tolerance < 1:
        raise ValueError('pair-f1-tolerance must be in [0, 1)')
    return config


def _training_config(config: ReweightingConfig) -> JointExperimentConfig:
    return JointExperimentConfig(
        input=config.input,
        split_dir=config.split_dir,
        run_dir=config.run_dir,
        evaluator=config.evaluator,
        model=config.model,
        epochs=config.epochs,
        evaluation_interval=config.evaluation_interval,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        max_length=config.max_length,
        learning_rate=config.learning_rate,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        log_steps=config.log_steps,
        mixed_precision=config.mixed_precision,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        mlflow_experiment=config.mlflow_experiment,
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'Required artifact not found: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _source_paths(
    config: ReweightingConfig,
) -> tuple[list[Path], list[Path], list[int]]:
    aspect = _load_json(config.aspect_run_dir / 'locked_selection.json')
    baseline = _load_json(config.baseline_run_dir / 'locked_selection.json')
    aspect_paths = [Path(value) for value in aspect['checkpoints']]
    finalist = baseline['finalists']['separate']
    baseline_paths = [Path(value) for value in finalist['checkpoints']]
    seeds = [int(value) for value in finalist['seeds']]
    for path in [*aspect_paths, *baseline_paths]:
        if not path.is_file():
            raise FileNotFoundError(f'Required checkpoint not found: {path}')
    if seeds != list(CONFIRMATION_SEEDS):
        raise RuntimeError('Baseline polarity seeds differ from confirmation seeds')
    return aspect_paths, baseline_paths, seeds


def _serialized_config(config: ReweightingConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _manifest(
    config: ReweightingConfig,
    aspect_paths: Sequence[Path],
    baseline_paths: Sequence[Path],
) -> dict[str, Any]:
    return {
        **_serialized_config(config),
        'weight_grid': [list(value) for value in WEIGHT_GRID],
        'confirmation_seeds': list(CONFIRMATION_SEEDS),
        'aspect_checkpoints': [str(value) for value in aspect_paths],
        'baseline_checkpoints': [str(value) for value in baseline_paths],
        'input_sha256': file_sha256(config.input),
        'split_sha256': {
            name: file_sha256(config.split_dir / f'{name}.csv')
            for name in ('train', 'eval', 'test')
        },
        'labels': {'aspect': list(ASPECTS), 'polarity': list(POLARITIES)},
    }


def _write_or_validate(path: Path, value: dict[str, Any]) -> None:
    if path.is_file() and _load_json(path) != value:
        raise RuntimeError(f'Existing manifest differs from requested run: {path}')
    atomic_write_json(path, value)


def _aspect_probabilities(
    torch: Any,
    transformers: Any,
    tokenizer: Any,
    config: ReweightingConfig,
    checkpoints: Sequence[Path],
    rows: Sequence[LabeledRow],
    label: str,
) -> list[list[float]]:
    aspect_config = AspectExperimentConfig(
        model=config.model,
        seeds=(42, 43, 44),
        max_length=config.max_length,
        eval_batch_size=config.eval_batch_size,
    )
    return _ensemble_probabilities(
        torch, transformers, tokenizer, aspect_config,
        checkpoints, rows, label,
    )


def _evaluate_ensemble(
    torch: Any,
    transformers: Any,
    tokenizer: Any,
    config: ReweightingConfig,
    checkpoints: Sequence[Path],
    rows: list[LabeledRow],
    aspect_probabilities: list[list[float]],
    label: str,
    thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    logits = [
        _separate_logits(
            torch, transformers, tokenizer, _training_config(config),
            checkpoint, rows, f'{label} seed {index + 1}/{len(checkpoints)}',
        )
        for index, checkpoint in enumerate(checkpoints)
    ]
    averaged = _mean_logits(torch, logits)
    reviews = group_aspect_targets(rows)
    polarity_indices = _reshape(averaged.argmax(dim=-1).tolist(), len(reviews))
    selected_thresholds = list(thresholds) if thresholds is not None else (
        tune_pair_thresholds(rows, aspect_probabilities, polarity_indices)
    )
    predictions = decode_pairs(
        reviews, aspect_probabilities, polarity_indices, selected_thresholds
    )

    review_indices = {review.item_id: index for index, review in enumerate(reviews)}
    probabilities = torch.softmax(averaged, dim=-1).tolist()
    gold_predictions = []
    gold_probabilities: dict[str, list[list[float]]] = {
        polarity: [] for polarity in POLARITIES
    }
    for row in rows:
        flat_index = (
            review_indices[row.item_id] * len(ASPECTS) + ASPECTS.index(row.aspect)
        )
        scores = probabilities[flat_index]
        prediction = POLARITIES[max(range(len(scores)), key=scores.__getitem__)]
        gold_predictions.append((row.item_id, row.aspect, prediction))
        gold_probabilities[row.polarity].append(scores)
    classes = _pair_class_metrics(rows, gold_predictions)
    minority = (
        float(classes['neutral']['f1']) + float(classes['conflict']['f1'])
    ) / 2

    from .evaluation import calculate_metrics
    metrics = calculate_metrics(rows, predictions)
    source = metrics['source_of_truth_metrics']
    probability_summary = {}
    for polarity in ('neutral', 'conflict'):
        values = gold_probabilities[polarity]
        probability_summary[polarity] = {
            predicted: sum(value[index] for value in values) / len(values)
            if values else 0.0
            for index, predicted in enumerate(POLARITIES)
        }
    return {
        'thresholds': selected_thresholds,
        'predictions': predictions,
        'pair_micro_f1': float(source['overall']['micro']['f1']),
        'polarity_micro_f1': float(source['polarity']['micro']['f1']),
        'exact_set_accuracy': float(
            metrics['supplemental_exact_match_accuracy']['overall']
        ),
        'gold_aspect_polarity_classes': classes,
        'minority_f1': minority,
        'probability_summary': probability_summary,
    }


def select_screening_candidate(
    candidates: dict[str, dict[str, Any]],
) -> str:
    eligible = [name for name, value in candidates.items() if value['eligible']]
    names = eligible or list(candidates)
    return max(
        names,
        key=lambda name: (
            float(candidates[name]['validation_minority_f1'])
            if eligible else float(candidates[name]['validation_pair_micro_f1']),
            float(candidates[name]['validation_pair_micro_f1']),
            -sum(float(value) for value in candidates[name]['multipliers']),
            name,
        ),
    )


def select_final_winner(
    baseline: dict[str, Any], candidate: dict[str, Any], tolerance: float,
) -> str:
    eligible = (
        candidate['pair_micro_f1'] >= baseline['pair_micro_f1'] - tolerance
        and candidate['minority_f1'] > baseline['minority_f1']
    )
    return 'reweighted' if eligible else 'baseline'


def _train_candidate(
    torch: Any,
    transformers: Any,
    mlflow: Any,
    tokenizer: Any,
    config: ReweightingConfig,
    train_rows: list[LabeledRow],
    validation_rows: list[LabeledRow],
    validation_aspects: list[list[float]],
    neutral: float,
    conflict: float,
    seed: int,
    minimum_pair_f1: float,
) -> dict[str, Any]:
    spec = LossSpec(
        'weighted-ce', neutral_multiplier=neutral,
        conflict_multiplier=conflict,
    )
    run_dir = config.run_dir / 'runs' / spec.slug / f'seed-{seed}'
    optimizer_checkpoint = (
        config.optimizer_dir / spec.slug / f'seed-{seed}' / 'latest.pt'
    )
    result = train_separate_polarity(
        torch, transformers, mlflow, tokenizer, _training_config(config),
        run_dir, spec, seed, train_rows, validation_rows,
        validation_aspects, minimum_pair_f1=minimum_pair_f1,
        optimizer_checkpoint=optimizer_checkpoint,
    )
    optimizer_checkpoint.unlink(missing_ok=True)
    effective_weights = loss_weights(torch, train_rows, spec).tolist()
    return {
        **result,
        'multipliers': [neutral, conflict],
        'effective_weights': dict(zip(POLARITIES, effective_weights)),
        'checkpoint': str(run_dir / 'best.pt'),
    }


def _public_result(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != 'predictions'}


def _write_report(
    config: ReweightingConfig,
    baseline_validation: dict[str, Any],
    screening: dict[str, dict[str, Any]],
    selected: str,
    final_validation: dict[str, dict[str, Any]],
    winner: str,
    test: dict[str, dict[str, Any]],
    runtime: float,
) -> None:
    screening_rows = '\n'.join(
        f"| {name} | {value['multipliers'][0]:g} | "
        f"{value['multipliers'][1]:g} | "
        f"{value['validation_pair_micro_f1']:.6f} | "
        f"{value['validation_minority_f1']:.6f} | {value['eligible']} |"
        for name, value in screening.items()
    )
    validation_rows = '\n'.join(
        f"| {name} | {value['pair_micro_f1']:.6f} | "
        f"{value['minority_f1']:.6f} | "
        f"{value['gold_aspect_polarity_classes']['neutral']['f1']:.6f} | "
        f"{value['gold_aspect_polarity_classes']['conflict']['f1']:.6f} |"
        for name, value in final_validation.items()
    )
    test_rows = '\n'.join(
        f"| {name} | {value['pair_micro_f1']:.6f} | "
        f"{value['polarity_micro_f1']:.6f} | "
        f"{value['exact_set_accuracy']:.6f} | "
        f"{value['minority_f1']:.6f} | "
        f"{value['gold_aspect_polarity_classes']['neutral']['f1']:.6f} | "
        f"{value['gold_aspect_polarity_classes']['conflict']['f1']:.6f} |"
        for name, value in test.items()
    )
    text = f'''# Neutral and conflict polarity reweighting

## Weighting

The existing inverse-frequency weights were multiplied only for neutral and
conflict. The validation pair-F1 tolerance was `{config.pair_f1_tolerance:.3f}`.

Baseline validation pair micro-F1: {baseline_validation['pair_micro_f1']:.6f}.
Baseline validation minority F1: {baseline_validation['minority_f1']:.6f}.

| Candidate | Neutral multiplier | Conflict multiplier | Pair F1 | Minority F1 | Eligible |
|---|---:|---:|---:|---:|---|
{screening_rows}

Selected screening candidate: `{selected}`.

## Three-seed validation

| System | Pair F1 | Minority F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|
{validation_rows}

Locked winner: `{winner}`.

## Held-out test

| System | Pair F1 | Polarity F1 | Exact set | Minority F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
{test_rows}

Selection and threshold tuning used validation only. This test split has been
evaluated by earlier experiments, so it is not globally untouched.

Runtime: {runtime / 60:.1f} minutes.
'''
    atomic_write_text(config.run_dir / 'report.md', text)


def run_experiment(config: ReweightingConfig) -> None:
    import mlflow
    import torch
    import transformers

    started = time.monotonic()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(config.run_dir / '.experiment.lock'):
        if (config.run_dir / 'results.json').is_file():
            LOGGER.info('Experiment already complete: %s', config.run_dir)
            return
        aspect_paths, baseline_paths, seeds = _source_paths(config)
        _write_or_validate(
            config.run_dir / 'manifest.json',
            _manifest(config, aspect_paths, baseline_paths),
        )
        atomic_write_json(config.run_dir / 'config.json', _serialized_config(config))

        train_rows = read_labeled_csv(config.split_dir / 'train.csv')
        validation_rows = read_labeled_csv(config.split_dir / 'eval.csv')
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.model, use_fast=True
        )
        validation_aspects = _aspect_probabilities(
            torch, transformers, tokenizer, config, aspect_paths,
            validation_rows, 'Validation aspects',
        )

        baseline_by_seed = {}
        for seed, checkpoint in zip(seeds, baseline_paths):
            baseline_by_seed[seed] = _evaluate_ensemble(
                torch, transformers, tokenizer, config, [checkpoint],
                validation_rows, validation_aspects,
                f'Validation baseline seed {seed}',
            )
        baseline_validation = _evaluate_ensemble(
            torch, transformers, tokenizer, config, baseline_paths,
            validation_rows, validation_aspects, 'Validation baseline ensemble',
        )

        base_spec = LossSpec('weighted-ce')
        counts = polarity_counts(train_rows)
        base_weights = loss_weights(torch, train_rows, base_spec).tolist()
        weighting = {
            'counts': dict(zip(POLARITIES, counts)),
            'base_inverse_frequency_weights': dict(zip(POLARITIES, base_weights)),
        }
        atomic_write_json(config.run_dir / 'weights.json', weighting)

        screening = {}
        baseline_seed_42 = baseline_by_seed[42]
        minimum_seed_42 = (
            baseline_seed_42['pair_micro_f1'] - config.pair_f1_tolerance
        )
        for neutral, conflict in WEIGHT_GRID:
            value = _train_candidate(
                torch, transformers, mlflow, tokenizer, config,
                train_rows, validation_rows, validation_aspects,
                neutral, conflict, 42, minimum_seed_42,
            )
            screening[LossSpec(
                'weighted-ce', neutral_multiplier=neutral,
                conflict_multiplier=conflict,
            ).slug] = value
            atomic_write_json(config.run_dir / 'screening.json', screening)

        selected = select_screening_candidate(screening)
        multipliers = screening[selected]['multipliers']
        confirmation = {42: screening[selected]}
        for seed, checkpoint in zip(seeds, baseline_paths):
            if seed == 42:
                continue
            minimum = (
                baseline_by_seed[seed]['pair_micro_f1']
                - config.pair_f1_tolerance
            )
            confirmation[seed] = _train_candidate(
                torch, transformers, mlflow, tokenizer, config,
                train_rows, validation_rows, validation_aspects,
                float(multipliers[0]), float(multipliers[1]), seed, minimum,
            )
        candidate_paths = [
            Path(confirmation[seed]['checkpoint']) for seed in seeds
        ]
        candidate_validation = _evaluate_ensemble(
            torch, transformers, tokenizer, config, candidate_paths,
            validation_rows, validation_aspects, 'Validation reweighted ensemble',
        )
        final_validation = {
            'baseline': baseline_validation,
            'reweighted': candidate_validation,
        }
        winner = select_final_winner(
            baseline_validation, candidate_validation,
            config.pair_f1_tolerance,
        )
        locked = {
            'winner': winner,
            'selected_candidate': selected,
            'multipliers': multipliers,
            'aspect_checkpoints': [str(path) for path in aspect_paths],
            'baseline_checkpoints': [str(path) for path in baseline_paths],
            'reweighted_checkpoints': [str(path) for path in candidate_paths],
            'thresholds': final_validation[winner]['thresholds'],
            'validation': {
                name: _public_result(value)
                for name, value in final_validation.items()
            },
            'test_accessed': False,
        }
        atomic_write_json(config.run_dir / 'locked_selection.json', locked)

        test_rows = read_labeled_csv(config.split_dir / 'test.csv')
        test_aspects = _aspect_probabilities(
            torch, transformers, tokenizer, config, aspect_paths,
            test_rows, 'Test aspects',
        )
        test = {}
        for name, checkpoints in (
            ('baseline', baseline_paths), ('reweighted', candidate_paths)
        ):
            test[name] = _evaluate_ensemble(
                torch, transformers, tokenizer, config, checkpoints,
                test_rows, test_aspects, f'Test {name}',
                thresholds=final_validation[name]['thresholds'],
            )
            _write_evaluation(
                _training_config(config), config.run_dir / 'test' / name,
                test_rows, test[name]['predictions'],
            )

        runtime = time.monotonic() - started
        results = {
            'weights': weighting,
            'baseline_by_seed': {
                str(seed): _public_result(value)
                for seed, value in baseline_by_seed.items()
            },
            'screening': screening,
            'selected_candidate': selected,
            'confirmation': confirmation,
            'validation': {
                name: _public_result(value)
                for name, value in final_validation.items()
            },
            'winner': winner,
            'test': {name: _public_result(value) for name, value in test.items()},
            'runtime_seconds': runtime,
        }
        atomic_write_json(config.run_dir / 'results.json', results)
        _write_report(
            config, baseline_validation, screening, selected,
            final_validation, winner, test, runtime,
        )
        locked['test_accessed'] = True
        atomic_write_json(config.run_dir / 'locked_selection.json', locked)


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
        LOGGER.exception('Polarity reweighting experiment failed')
        return 1
    return 0
