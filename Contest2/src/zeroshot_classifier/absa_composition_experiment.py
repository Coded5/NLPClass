from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .absa_training import group_aspect_targets
from .checkpoint import RunLock
from .data import LabeledRow, read_labeled_csv
from .evaluation import calculate_metrics, evaluate, read_predictions
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .joint_absa_experiment import (
    JointExperimentConfig,
    _bootstrap_delta,
    _group_accuracy,
    _mean_logits,
    _pair_class_metrics,
    _reshape,
    _separate_logits,
    decode_pairs,
    tune_pair_thresholds,
)
from .labels import ASPECTS, POLARITIES
from .multilabel_aspect_experiment import (
    AspectExperimentConfig,
    _ensemble_probabilities,
)
from .roberta_training import _prepare_mlflow_tracking_uri, _write_labeled_csv, _write_predictions


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompositionConfig:
    input: Path = Path('data/contest2_train.csv')
    split_dir: Path = Path('artifacts/training/roberta-aspect-exp1/splits')
    aspect_run_dir: Path = Path('artifacts/experiments/multilabel-aspect-v1')
    conditioned_run_dir: Path = Path(
        'artifacts/experiments/multilabel-conditioned-v1'
    )
    joint_run_dir: Path = Path(
        'artifacts/experiments/absa-imbalance-joint-v1'
    )
    run_dir: Path = Path(
        'artifacts/experiments/multilabel-aspect-old-polarity-v1'
    )
    evaluator: Path = Path('scripts/evaluate.py')
    model: str = 'FacebookAI/roberta-base'
    max_length: int = 256
    eval_batch_size: int = 32
    bootstrap_samples: int = 10_000
    mlflow_tracking_uri: str = 'sqlite:///artifacts/mlflow.db'
    mlflow_experiment: str = 'contest2-multilabel-aspect-old-polarity-v1'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Compose the three-seed aspect ensemble with existing '
            'aspect-conditioned polarity models.'
        )
    )
    for name, field in CompositionConfig.__dataclass_fields__.items():
        default = field.default
        option = f'--{name.replace("_", "-")}'
        if isinstance(default, Path):
            parser.add_argument(option, type=Path, default=default)
        elif isinstance(default, str):
            parser.add_argument(option, default=default)
        else:
            parser.add_argument(option, type=type(default), default=default)
    return parser


def config_from_args(argv: list[str] | None = None) -> CompositionConfig:
    config = CompositionConfig(**vars(build_parser().parse_args(argv)))
    if config.max_length < 1 or config.eval_batch_size < 1:
        raise ValueError('max-length and eval-batch-size must be at least 1')
    if config.bootstrap_samples < 1:
        raise ValueError('bootstrap-samples must be at least 1')
    return config


def select_candidate(candidates: dict[str, dict[str, Any]]) -> str:
    if not candidates:
        raise ValueError('At least one polarity candidate is required')
    return max(
        candidates,
        key=lambda name: (
            float(candidates[name]['metrics']['source_of_truth_metrics']['overall']['micro']['f1']),
            float(candidates[name]['metrics']['supplemental_exact_match_accuracy']['overall']),
            name == 'one_seed',
        ),
    )


def oracle_aspect_predictions(
    rows: Sequence[LabeledRow],
    polarity_predictions: Sequence[Sequence[int]],
) -> list[tuple[str, str, str]]:
    reviews = group_aspect_targets(rows)
    if len(reviews) != len(polarity_predictions):
        raise ValueError('Polarity rows do not match grouped reviews')
    return [
        (review.item_id, ASPECTS[aspect], POLARITIES[polarities[aspect]])
        for review, polarities in zip(reviews, polarity_predictions)
        for aspect in review.aspects
    ]


def _serialized_config(config: CompositionConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'Required experiment artifact not found: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _checkpoint_paths(
    config: CompositionConfig,
) -> tuple[list[Path], Path, list[Path], dict[str, Any]]:
    aspect_selection = _load_json(config.aspect_run_dir / 'locked_selection.json')
    aspect_checkpoints = [Path(value) for value in aspect_selection['checkpoints']]
    one_seed = config.conditioned_run_dir / 'best' / 'polarity.pt'
    joint_selection = _load_json(config.joint_run_dir / 'locked_selection.json')
    separate = joint_selection['finalists']['separate']
    ensemble = [Path(value) for value in separate['checkpoints']]
    for path in [*aspect_checkpoints, one_seed, *ensemble]:
        if not path.is_file():
            raise FileNotFoundError(f'Required checkpoint not found: {path}')
    return aspect_checkpoints, one_seed, ensemble, aspect_selection


def _expected_hashes(config: CompositionConfig) -> tuple[str, dict[str, str]]:
    return (
        file_sha256(config.input),
        {
            name: file_sha256(config.split_dir / f'{name}.csv')
            for name in ('train', 'eval', 'test')
        },
    )


def _validate_manifest_values(
    manifest: dict[str, Any],
    config: CompositionConfig,
    input_hash: str,
    split_hashes: dict[str, str],
    label_kind: str,
) -> None:
    expected = {
        'model': config.model,
        'max_length': config.max_length,
        'input_sha256': input_hash,
        'split_sha256': split_hashes,
    }
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise RuntimeError(
            f'{label_kind} manifest mismatch: {", ".join(sorted(mismatches))}'
        )


def _validate_sources(
    config: CompositionConfig,
    aspect_selection: dict[str, Any],
    polarity_ensemble: Sequence[Path],
) -> dict[str, Any]:
    input_hash, split_hashes = _expected_hashes(config)
    aspect_manifest = _load_json(config.aspect_run_dir / 'manifest.json')
    conditioned_manifest = _load_json(config.conditioned_run_dir / 'manifest.json')
    _validate_manifest_values(
        aspect_manifest, config, input_hash, split_hashes, 'Aspect'
    )
    _validate_manifest_values(
        conditioned_manifest, config, input_hash, split_hashes, 'One-seed polarity'
    )
    if aspect_manifest.get('labels') != list(ASPECTS):
        raise RuntimeError('Aspect label order differs from the repository label order')
    labels = conditioned_manifest.get('labels', {})
    if labels.get('aspect') != list(ASPECTS) or labels.get('polarity') != list(POLARITIES):
        raise RuntimeError('Conditioned model label order differs from repository labels')
    if not aspect_selection.get('test_accessed'):
        raise RuntimeError('Aspect ensemble is not marked complete')

    ensemble_manifests = []
    for checkpoint in polarity_ensemble:
        manifest = _load_json(checkpoint.parent / 'manifest.json')
        _validate_manifest_values(
            manifest, config, input_hash, split_hashes, 'Three-seed polarity'
        )
        if manifest.get('architecture') != 'separate':
            raise RuntimeError('Polarity ensemble contains a non-separate checkpoint')
        loss = manifest.get('loss', {})
        if loss.get('name') != 'weighted-ce':
            raise RuntimeError('Polarity ensemble contains a non-weighted-CE checkpoint')
        ensemble_manifests.append(manifest)
    return {
        'input_sha256': input_hash,
        'split_sha256': split_hashes,
        'aspect_seeds': aspect_selection['seeds'],
        'polarity_ensemble_seeds': [value['seed'] for value in ensemble_manifests],
        'labels': {'aspect': list(ASPECTS), 'polarity': list(POLARITIES)},
    }


def _write_or_validate_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.is_file():
        if _load_json(path) != manifest:
            raise RuntimeError('Run manifest differs from the requested composition')
    else:
        atomic_write_json(path, manifest)


def _aspect_probabilities(
    torch: Any,
    transformers: Any,
    tokenizer: Any,
    config: CompositionConfig,
    checkpoints: Sequence[Path],
    seeds: Sequence[int],
    rows: Sequence[LabeledRow],
    split_name: str,
) -> list[list[float]]:
    aspect_config = AspectExperimentConfig(
        model=config.model,
        seeds=tuple(seeds),
        max_length=config.max_length,
        eval_batch_size=config.eval_batch_size,
    )
    return _ensemble_probabilities(
        torch, transformers, tokenizer, aspect_config,
        checkpoints, rows, f'{split_name} aspects',
    )


def _polarity_indices(
    torch: Any,
    transformers: Any,
    tokenizer: Any,
    config: CompositionConfig,
    checkpoints: Sequence[Path],
    rows: list[LabeledRow],
    candidate_name: str,
) -> list[list[int]]:
    inference_config = JointExperimentConfig(
        model=config.model,
        max_length=config.max_length,
        eval_batch_size=config.eval_batch_size,
    )
    logits = [
        _separate_logits(
            torch, transformers, tokenizer, inference_config, checkpoint,
            rows, f'{candidate_name} polarity {index + 1}/{len(checkpoints)}',
        )
        for index, checkpoint in enumerate(checkpoints)
    ]
    averaged = _mean_logits(torch, logits)
    return _reshape(
        averaged.argmax(dim=-1).tolist(), len(group_aspect_targets(rows))
    )


def _assert_full_coverage(
    rows: Sequence[LabeledRow], predictions: Sequence[tuple[str, str, str]]
) -> None:
    expected = {row.item_id for row in rows}
    actual = {item_id for item_id, _, _ in predictions}
    if actual != expected:
        raise RuntimeError(
            f'Prediction coverage mismatch: missing={len(expected - actual)} '
            f'unexpected={len(actual - expected)}'
        )


def _write_evaluation(
    config: CompositionConfig,
    output_dir: Path,
    rows: list[LabeledRow],
    predictions: list[tuple[str, str, str]],
) -> dict[str, Any]:
    _assert_full_coverage(rows, predictions)
    gold_path = output_dir / 'gold.csv'
    prediction_path = output_dir / 'predictions.csv'
    _write_labeled_csv(gold_path, rows)
    _write_predictions(prediction_path, predictions)
    return evaluate(
        gold_path, prediction_path, config.evaluator,
        output_dir / 'metrics.json', output_dir / 'official_evaluation.txt',
    )


def _system_summary(
    rows: list[LabeledRow], predictions: list[tuple[str, str, str]]
) -> dict[str, Any]:
    metrics = calculate_metrics(rows, predictions)
    groups = _group_accuracy(rows, predictions)
    source = metrics['source_of_truth_metrics']
    return {
        'aspect_micro_f1': source['aspect']['micro']['f1'],
        'polarity_micro_f1': source['polarity']['micro']['f1'],
        'overall_pair_micro_f1': source['overall']['micro']['f1'],
        'exact_set_accuracy': metrics['supplemental_exact_match_accuracy']['overall'],
        'single_exact_accuracy': groups['single']['exact_set_accuracy'],
        'multi_exact_accuracy': groups['multi']['exact_set_accuracy'],
        'pair_polarity_classes': _pair_class_metrics(rows, predictions),
    }


def _reference_systems(
    config: CompositionConfig, test_rows: list[LabeledRow]
) -> tuple[dict[str, dict[str, Any]], dict[str, list[tuple[str, str, str]]]]:
    paths = {
        'Previous one-seed pipeline': config.conditioned_run_dir / 'test' / 'predictions.csv',
        'Previous separate ensemble': config.joint_run_dir / 'ensembles' / 'separate' / 'test' / 'predictions.csv',
        'Previous joint ensemble': config.joint_run_dir / 'ensembles' / 'joint' / 'test' / 'predictions.csv',
        'Legacy single-label': config.conditioned_run_dir / 'baseline-test' / 'predictions.csv',
    }
    predictions = {name: read_predictions(path) for name, path in paths.items()}
    for values in predictions.values():
        _assert_full_coverage(test_rows, values)
    return (
        {name: _system_summary(test_rows, values) for name, values in predictions.items()},
        predictions,
    )


def _write_report(
    config: CompositionConfig,
    candidates: dict[str, dict[str, Any]],
    winner: str,
    systems: dict[str, dict[str, Any]],
    oracle_metrics: dict[str, Any],
    bootstrap: dict[str, dict[str, float]],
    runtime_seconds: float,
) -> None:
    candidate_rows = '\n'.join(
        '| {} | {:.6f} | {:.6f} | `{}` |'.format(
            name,
            value['metrics']['source_of_truth_metrics']['overall']['micro']['f1'],
            value['metrics']['supplemental_exact_match_accuracy']['overall'],
            json.dumps(value['thresholds']),
        )
        for name, value in candidates.items()
    )
    system_rows = '\n'.join(
        '| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |'.format(
            name,
            value['aspect_micro_f1'], value['polarity_micro_f1'],
            value['overall_pair_micro_f1'], value['exact_set_accuracy'],
            value['multi_exact_accuracy'],
        )
        for name, value in systems.items()
    )
    polarity_rows = '\n'.join(
        '| {} | {} | {:.6f} | {:.6f} | {:.6f} | {} |'.format(
            name, polarity, values['precision'], values['recall'],
            values['f1'], values['support'],
        )
        for name, system in systems.items()
        for polarity, values in system['pair_polarity_classes'].items()
    )
    text = f'''# Multi-label aspect plus existing polarity experiment

## Validation selection

| Candidate | Pair micro-F1 | Exact-set accuracy | Pair-tuned thresholds |
|---|---:|---:|---|
{candidate_rows}

Locked polarity candidate: `{winner}`.

## Held-out test

| System | Aspect micro-F1 | Polarity micro-F1 | Pair micro-F1 | Exact-set accuracy | Multi-aspect exact |
|---|---:|---:|---:|---:|---:|
{system_rows}

Oracle-aspect polarity pair micro-F1: {oracle_metrics['source_of_truth_metrics']['overall']['micro']['f1']:.6f}.

## Pair-level polarity breakdown

| System | Polarity | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
{polarity_rows}

Winner minus previous one-seed pipeline: {bootstrap['winner_vs_one_seed']['mean']:+.6f}
(95% CI {bootstrap['winner_vs_one_seed']['lower_95']:+.6f} to {bootstrap['winner_vs_one_seed']['upper_95']:+.6f}).

Winner minus legacy baseline: {bootstrap['winner_vs_legacy']['mean']:+.6f}
(95% CI {bootstrap['winner_vs_legacy']['lower_95']:+.6f} to {bootstrap['winner_vs_legacy']['upper_95']:+.6f}).

Polarity candidate selection and pair-threshold tuning used validation data only.
This test split has been evaluated by previous experiments and is therefore held-out
for this run, but not globally untouched.

Runtime: {runtime_seconds / 60:.1f} minutes.
'''
    atomic_write_text(config.run_dir / 'report.md', text)


def run_experiment(config: CompositionConfig) -> None:
    import mlflow
    import torch
    import transformers

    started = time.monotonic()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(config.run_dir / '.experiment.lock'):
        completed_path = config.run_dir / 'results.json'
        if completed_path.is_file() and (config.run_dir / 'mlflow_run.json').is_file():
            LOGGER.info('Composition experiment is already complete: %s', completed_path)
            return

        aspect_checkpoints, one_seed, polarity_ensemble, aspect_selection = (
            _checkpoint_paths(config)
        )
        source_manifest = _validate_sources(
            config, aspect_selection, polarity_ensemble
        )
        manifest = {
            **_serialized_config(config),
            **source_manifest,
            'aspect_checkpoints': [str(path) for path in aspect_checkpoints],
            'polarity_candidates': {
                'one_seed': [str(one_seed)],
                'three_seed': [str(path) for path in polarity_ensemble],
            },
        }
        _write_or_validate_manifest(config.run_dir / 'manifest.json', manifest)
        atomic_write_json(config.run_dir / 'config.json', _serialized_config(config))

        validation_rows = read_labeled_csv(config.split_dir / 'eval.csv')
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, use_fast=True)
        validation_aspects = _aspect_probabilities(
            torch, transformers, tokenizer, config, aspect_checkpoints,
            aspect_selection['seeds'], validation_rows, 'Validation',
        )
        candidates: dict[str, dict[str, Any]] = {}
        candidate_checkpoints = {
            'one_seed': [one_seed],
            'three_seed': polarity_ensemble,
        }
        for name, checkpoints in candidate_checkpoints.items():
            polarities = _polarity_indices(
                torch, transformers, tokenizer, config, checkpoints,
                validation_rows, f'Validation {name}',
            )
            thresholds = tune_pair_thresholds(
                validation_rows, validation_aspects, polarities
            )
            predictions = decode_pairs(
                group_aspect_targets(validation_rows), validation_aspects,
                polarities, thresholds,
            )
            metrics = _write_evaluation(
                config, config.run_dir / 'validation' / name,
                validation_rows, predictions,
            )
            candidates[name] = {
                'checkpoints': [str(path) for path in checkpoints],
                'thresholds': thresholds,
                'metrics': metrics,
            }

        winner = select_candidate(candidates)
        locked = {
            'winner': winner,
            'aspect_checkpoints': [str(path) for path in aspect_checkpoints],
            'polarity_checkpoints': candidates[winner]['checkpoints'],
            'thresholds': candidates[winner]['thresholds'],
            'validation_pair_micro_f1': candidates[winner]['metrics']['source_of_truth_metrics']['overall']['micro']['f1'],
            'test_accessed': False,
        }
        atomic_write_json(config.run_dir / 'locked_selection.json', locked)

        # Test labels and reference test predictions are loaded only after selection is locked.
        test_rows = read_labeled_csv(config.split_dir / 'test.csv')
        train_rows = read_labeled_csv(config.split_dir / 'train.csv')
        source_rows = read_labeled_csv(config.input)
        split_ids = [
            {row.item_id for row in rows}
            for rows in (train_rows, validation_rows, test_rows)
        ]
        if any(split_ids[left] & split_ids[right] for left, right in ((0, 1), (0, 2), (1, 2))):
            raise RuntimeError('Persisted splits contain ID leakage')
        if set().union(*split_ids) != {row.item_id for row in source_rows}:
            raise RuntimeError('Persisted splits do not cover the source dataset')

        test_aspects = _aspect_probabilities(
            torch, transformers, tokenizer, config, aspect_checkpoints,
            aspect_selection['seeds'], test_rows, 'Test',
        )
        winner_polarities = _polarity_indices(
            torch, transformers, tokenizer, config,
            [Path(value) for value in candidates[winner]['checkpoints']],
            test_rows, f'Test {winner}',
        )
        winner_predictions = decode_pairs(
            group_aspect_targets(test_rows), test_aspects, winner_polarities,
            candidates[winner]['thresholds'],
        )
        winner_metrics = _write_evaluation(
            config, config.run_dir / 'test' / 'winner',
            test_rows, winner_predictions,
        )
        oracle_predictions = oracle_aspect_predictions(
            test_rows, winner_polarities
        )
        oracle_metrics = _write_evaluation(
            config, config.run_dir / 'test' / 'oracle-aspects',
            test_rows, oracle_predictions,
        )

        reference_summaries, reference_predictions = _reference_systems(
            config, test_rows
        )
        systems = {
            'New composed winner': _system_summary(test_rows, winner_predictions),
            **reference_summaries,
        }
        bootstrap = {
            'samples': config.bootstrap_samples,
            'winner_vs_one_seed': _bootstrap_delta(
                test_rows, winner_predictions,
                reference_predictions['Previous one-seed pipeline'],
                config.bootstrap_samples, seed=20260910,
            ),
            'winner_vs_legacy': _bootstrap_delta(
                test_rows, winner_predictions,
                reference_predictions['Legacy single-label'],
                config.bootstrap_samples, seed=20260911,
            ),
        }
        runtime_seconds = time.monotonic() - started
        results = {
            'winner': winner,
            'validation_candidates': candidates,
            'test_winner_metrics': winner_metrics,
            'oracle_aspect_polarity_metrics': oracle_metrics,
            'test_systems': systems,
            'bootstrap': bootstrap,
            'runtime_seconds': runtime_seconds,
        }
        atomic_write_json(config.run_dir / 'results.json', results)
        _write_report(
            config, candidates, winner, systems, oracle_metrics,
            bootstrap, runtime_seconds,
        )
        locked['test_accessed'] = True
        atomic_write_json(config.run_dir / 'locked_selection.json', locked)

        _prepare_mlflow_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        with mlflow.start_run(run_name='composed-absa-evaluation') as active_run:
            atomic_write_json(
                config.run_dir / 'mlflow_run.json', {'run_id': active_run.info.run_id}
            )
            source = winner_metrics['source_of_truth_metrics']
            mlflow.log_params({'winner': winner, 'model': config.model})
            mlflow.log_metric('test/aspect_micro_f1', source['aspect']['micro']['f1'])
            mlflow.log_metric('test/polarity_micro_f1', source['polarity']['micro']['f1'])
            mlflow.log_metric('test/overall_pair_micro_f1', source['overall']['micro']['f1'])
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
        LOGGER.warning('Interrupted; rerun the same command to restart inference')
        return 130
    except Exception:
        LOGGER.exception('Composition experiment failed')
        return 1
    return 0
