from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .absa_training import ExperimentConfig as TrainingConfig, group_aspect_targets, train_stage
from .checkpoint import RunLock
from .data import LabeledRow, read_labeled_csv
from .deberta_experiment import load_tokenizer
from .imbalance_losses import LossSpec
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .joint_absa_experiment import (
    JointExperimentConfig,
    _mean_logits,
    _reshape,
    _separate_logits,
    _write_evaluation,
    decode_pairs,
    train_separate_polarity,
    tune_pair_thresholds,
)
from .labels import ASPECTS
from .multilabel_aspect_experiment import (
    AspectExperimentConfig,
    _predict_checkpoint_probabilities,
    mean_probabilities,
)
from .roberta_training import _prepare_mlflow_tracking_uri, _seed_everything


LOGGER = logging.getLogger(__name__)
DEBERTA_ASPECT_SEEDS = (42, 43, 44)
POLARITY_SEEDS = (17, 42, 73)
SYSTEMS = (
    'deberta-only',
    'roberta-aspect-deberta-polarity',
    'roberta-aspect-mixed-polarity',
)


@dataclass(frozen=True)
class ThreeExperimentConfig:
    split_dir: Path = Path('artifacts/training/roberta-aspect-exp1/splits')
    run_dir: Path = Path('artifacts/experiments/deberta-v3-three-systems-v1')
    pilot_dir: Path = Path('artifacts/experiments/deberta-v3-current-best-v1')
    roberta_aspect_dir: Path = Path('artifacts/experiments/multilabel-aspect-v1')
    roberta_polarity_dir: Path = Path('artifacts/experiments/absa-imbalance-joint-v1')
    deberta_model: str = 'microsoft/deberta-v3-base'
    roberta_model: str = 'FacebookAI/roberta-base'
    epochs: int = 50
    evaluation_interval: int = 5
    early_stopping_patience: int = 3
    max_length: int = 256
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    mixed_precision: str = 'auto'
    mlflow_tracking_uri: str = 'sqlite:////home/kami/Projects/NLP/Contest2/artifacts/mlflow.db'
    mlflow_experiment: str = 'contest2-deberta-v3-three-systems-v1'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train DeBERTa ensembles and evaluate three locked ABSA systems.'
    )
    for name, field in ThreeExperimentConfig.__dataclass_fields__.items():
        default = field.default
        option = f'--{name.replace("_", "-")}'
        parser.add_argument(option, type=Path if isinstance(default, Path) else type(default), default=default)
    return parser


def config_from_args(argv: list[str] | None = None) -> ThreeExperimentConfig:
    config = ThreeExperimentConfig(**vars(build_parser().parse_args(argv)))
    positive = (
        config.epochs, config.evaluation_interval, config.early_stopping_patience,
        config.max_length, config.train_batch_size, config.eval_batch_size,
        config.gradient_accumulation_steps,
    )
    if any(value <= 0 for value in positive):
        raise ValueError('Epoch, interval, patience, length, and batch values must be positive')
    return config


def average_backbone_logits(torch: Any, left: Any, right: Any) -> Any:
    if left.shape != right.shape:
        raise ValueError('Backbone polarity logits must have the same shape')
    return (left + right) / 2


def system_components(name: str) -> tuple[str, str]:
    mapping = {
        'deberta-only': ('deberta', 'deberta'),
        'roberta-aspect-deberta-polarity': ('roberta', 'deberta'),
        'roberta-aspect-mixed-polarity': ('roberta', 'mixed'),
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f'Unknown system: {name}') from error


def _serialized(config: ThreeExperimentConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _training_config(config: ThreeExperimentConfig, seed: int, run_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        split_dir=config.split_dir, run_dir=run_dir, model=config.deberta_model,
        epochs=config.epochs, evaluation_interval=config.evaluation_interval,
        early_stopping_patience=config.early_stopping_patience, seed=seed,
        max_length=config.max_length, learning_rate=config.learning_rate,
        train_batch_size=config.train_batch_size, eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        mixed_precision=config.mixed_precision,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        mlflow_experiment=config.mlflow_experiment,
        mlflow_run_name=f'deberta-aspect-seed-{seed}',
    )


def _joint_config(
    config: ThreeExperimentConfig, model: str, run_dir: Path,
) -> JointExperimentConfig:
    return JointExperimentConfig(
        split_dir=config.split_dir, run_dir=run_dir, model=model,
        epochs=config.epochs, evaluation_interval=config.evaluation_interval,
        early_stopping_patience=config.early_stopping_patience,
        max_length=config.max_length, learning_rate=config.learning_rate,
        train_batch_size=config.train_batch_size, eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        mixed_precision=config.mixed_precision,
        mlflow_tracking_uri=config.mlflow_tracking_uri,
        mlflow_experiment=config.mlflow_experiment,
    )


def _aspect_inference_config(
    config: ThreeExperimentConfig, model: str,
) -> AspectExperimentConfig:
    return replace(
        AspectExperimentConfig(), model=model, split_dir=config.split_dir,
        eval_batch_size=config.eval_batch_size,
    )


def _deberta_aspect_paths(config: ThreeExperimentConfig) -> list[Path]:
    return [
        config.pilot_dir / 'aspect/best/aspect.pt',
        config.run_dir / 'models/aspect/seed-43/best/aspect.pt',
        config.run_dir / 'models/aspect/seed-44/best/aspect.pt',
    ]


def _roberta_aspect_paths(config: ThreeExperimentConfig) -> list[Path]:
    return [
        config.roberta_aspect_dir / f'runs/seed-{seed}/best/aspect.pt'
        for seed in DEBERTA_ASPECT_SEEDS
    ]


def _deberta_polarity_paths(config: ThreeExperimentConfig) -> list[Path]:
    return [
        config.run_dir / f'models/polarity/seed-{seed}/best.pt'
        for seed in POLARITY_SEEDS
    ]


def _roberta_polarity_paths(config: ThreeExperimentConfig) -> list[Path]:
    base = config.roberta_polarity_dir
    return [
        base / 'confirmation/separate/weighted-ce/seed-17/best.pt',
        base / 'screening/loss/weighted-ce/seed-42/best.pt',
        base / 'confirmation/separate/weighted-ce/seed-73/best.pt',
    ]


def _aspect_probabilities(
    torch: Any, transformers: Any, tokenizer: Any,
    inference_config: AspectExperimentConfig, checkpoints: Sequence[Path],
    rows: Sequence[LabeledRow], label: str,
) -> list[list[float]]:
    return mean_probabilities([
        _predict_checkpoint_probabilities(
            torch, transformers, tokenizer, inference_config, checkpoint,
            rows, f'{label} seed {index + 1}/{len(checkpoints)}',
        )
        for index, checkpoint in enumerate(checkpoints)
    ])


def _polarity_logits(
    torch: Any, transformers: Any, tokenizer: Any,
    inference_config: JointExperimentConfig, checkpoints: Sequence[Path],
    rows: list[LabeledRow], label: str,
) -> Any:
    return _mean_logits(torch, [
        _separate_logits(
            torch, transformers, tokenizer, inference_config, checkpoint,
            rows, f'{label} seed {index + 1}/{len(checkpoints)}',
        )
        for index, checkpoint in enumerate(checkpoints)
    ])


def _decode(
    rows: list[LabeledRow], aspect_probabilities: list[list[float]],
    polarity_logits: Any, thresholds: Sequence[float] | None = None,
) -> tuple[list[tuple[str, str, str]], list[float]]:
    reviews = group_aspect_targets(rows)
    polarities = _reshape(polarity_logits.argmax(dim=-1).tolist(), len(reviews))
    selected = list(thresholds) if thresholds is not None else tune_pair_thresholds(
        rows, aspect_probabilities, polarities
    )
    return decode_pairs(reviews, aspect_probabilities, polarities, selected), selected


def _train_models(
    torch: Any, transformers: Any, mlflow: Any, tokenizer: Any,
    config: ThreeExperimentConfig, train_rows: list[LabeledRow],
    validation_rows: list[LabeledRow],
) -> tuple[list[Path], list[Path], list[list[float]]]:
    aspect_paths = _deberta_aspect_paths(config)
    if not aspect_paths[0].is_file():
        raise FileNotFoundError(f'Pilot aspect checkpoint is missing: {aspect_paths[0]}')
    for seed, checkpoint in zip(DEBERTA_ASPECT_SEEDS[1:], aspect_paths[1:]):
        if checkpoint.is_file():
            LOGGER.info('Reusing completed DeBERTa aspect seed %d', seed)
            continue
        run_dir = checkpoint.parents[1]
        training = _training_config(config, seed, run_dir)
        _seed_everything(torch, seed)
        with mlflow.start_run(run_name=training.mlflow_run_name):
            train_stage(
                torch, transformers, mlflow, tokenizer, training,
                'aspect', train_rows, validation_rows,
            )

    aspect_probabilities = _aspect_probabilities(
        torch, transformers, tokenizer,
        _aspect_inference_config(config, config.deberta_model),
        aspect_paths, validation_rows, 'Validation DeBERTa aspects',
    )
    polarity_paths = _deberta_polarity_paths(config)
    polarity_config = _joint_config(
        config, config.deberta_model, config.run_dir / 'models/polarity'
    )
    for seed, checkpoint in zip(POLARITY_SEEDS, polarity_paths):
        if checkpoint.is_file() and (checkpoint.parent / 'result.json').is_file():
            LOGGER.info('Reusing completed DeBERTa polarity seed %d', seed)
            continue
        run_dir = checkpoint.parent
        train_separate_polarity(
            torch, transformers, mlflow, tokenizer,
            replace(polarity_config, run_dir=run_dir), run_dir,
            LossSpec('weighted-ce'), seed, train_rows, validation_rows,
            aspect_probabilities,
        )
    return aspect_paths, polarity_paths, aspect_probabilities


def _write_system_report(
    directory: Path, name: str, validation: dict[str, Any],
    test: dict[str, Any], runtime_seconds: float,
) -> None:
    def values(metrics: dict[str, Any]) -> tuple[float, float, float]:
        truth = metrics['source_of_truth_metrics']
        return (
            float(truth['overall']['micro']['f1']),
            float(truth['polarity']['classes']['neutral']['f1']),
            float(truth['polarity']['classes']['conflict']['f1']),
        )
    validation_values = values(validation); test_values = values(test)
    atomic_write_text(directory / 'report.md', f'''# {name}

| Split | Pair micro-F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|
| Validation | {validation_values[0]:.6f} | {validation_values[1]:.6f} | {validation_values[2]:.6f} |
| Test | {test_values[0]:.6f} | {test_values[1]:.6f} | {test_values[2]:.6f} |

Thresholds and model checkpoints were locked on validation before test labels were loaded.
Shared orchestration runtime: {runtime_seconds / 60:.1f} minutes.
''')


def run_experiments(config: ThreeExperimentConfig) -> None:
    import mlflow
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError('The overnight experiment requires CUDA')
    config.run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(config.run_dir / '.experiment.lock'):
        started = time.monotonic()
        manifest = {
            **_serialized(config),
            'deberta_aspect_seeds': list(DEBERTA_ASPECT_SEEDS),
            'polarity_seeds': list(POLARITY_SEEDS),
            'systems': list(SYSTEMS),
            'split_sha256': {
                name: file_sha256(config.split_dir / f'{name}.csv')
                for name in ('train', 'eval', 'test')
            },
        }
        manifest_path = config.run_dir / 'manifest.json'
        if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Overnight experiment manifest mismatch')
        atomic_write_json(manifest_path, manifest)
        train_rows = read_labeled_csv(config.split_dir / 'train.csv')
        validation_rows = read_labeled_csv(config.split_dir / 'eval.csv')
        train_ids = {row.item_id for row in train_rows}
        validation_ids = {row.item_id for row in validation_rows}
        if train_ids & validation_ids:
            raise RuntimeError('Train and validation IDs overlap')
        _prepare_mlflow_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        deberta_tokenizer = load_tokenizer(transformers, config.deberta_model)
        roberta_tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.roberta_model, use_fast=True
        )
        aspect_paths, polarity_paths, deberta_validation_aspects = _train_models(
            torch, transformers, mlflow, deberta_tokenizer, config,
            train_rows, validation_rows,
        )
        roberta_validation_aspects = _aspect_probabilities(
            torch, transformers, roberta_tokenizer,
            _aspect_inference_config(config, config.roberta_model),
            _roberta_aspect_paths(config), validation_rows,
            'Validation RoBERTa aspects',
        )
        deberta_polarity_config = _joint_config(config, config.deberta_model, config.run_dir)
        roberta_polarity_config = _joint_config(config, config.roberta_model, config.run_dir)
        deberta_validation_logits = _polarity_logits(
            torch, transformers, deberta_tokenizer, deberta_polarity_config,
            polarity_paths, validation_rows, 'Validation DeBERTa polarity',
        )
        roberta_validation_logits = _polarity_logits(
            torch, transformers, roberta_tokenizer, roberta_polarity_config,
            _roberta_polarity_paths(config), validation_rows, 'Validation RoBERTa polarity',
        )
        validation_inputs = {
            'deberta-only': (deberta_validation_aspects, deberta_validation_logits),
            'roberta-aspect-deberta-polarity': (
                roberta_validation_aspects, deberta_validation_logits,
            ),
            'roberta-aspect-mixed-polarity': (
                roberta_validation_aspects,
                average_backbone_logits(
                    torch, roberta_validation_logits, deberta_validation_logits
                ),
            ),
        }
        locked: dict[str, dict[str, Any]] = {}
        validation_metrics: dict[str, dict[str, Any]] = {}
        for name in SYSTEMS:
            system_dir = config.run_dir / 'systems' / name
            predictions, thresholds = _decode(
                validation_rows, *validation_inputs[name]
            )
            validation_metrics[name] = _write_evaluation(
                deberta_polarity_config, system_dir / 'validation',
                validation_rows, predictions,
            )
            aspect_backbone, polarity_backbone = system_components(name)
            locked[name] = {
                'system': name, 'aspect_backbone': aspect_backbone,
                'polarity_backbone': polarity_backbone,
                'thresholds': thresholds,
                'deberta_aspect_checkpoints': [str(path) for path in aspect_paths]
                    if aspect_backbone == 'deberta' else [],
                'roberta_aspect_checkpoints': [str(path) for path in _roberta_aspect_paths(config)]
                    if aspect_backbone == 'roberta' else [],
                'deberta_polarity_checkpoints': [str(path) for path in polarity_paths],
                'roberta_polarity_checkpoints': [str(path) for path in _roberta_polarity_paths(config)]
                    if polarity_backbone == 'mixed' else [],
                'validation_pair_micro_f1': validation_metrics[name][
                    'source_of_truth_metrics'
                ]['overall']['micro']['f1'],
                'test_accessed': False,
            }
            atomic_write_json(system_dir / 'locked_selection.json', locked[name])

        # Test is loaded only after every system has a persisted validation lock.
        test_rows = read_labeled_csv(config.split_dir / 'test.csv')
        test_ids = {row.item_id for row in test_rows}
        if test_ids & (train_ids | validation_ids):
            raise RuntimeError('Test IDs overlap train or validation IDs')
        deberta_test_aspects = _aspect_probabilities(
            torch, transformers, deberta_tokenizer,
            _aspect_inference_config(config, config.deberta_model),
            aspect_paths, test_rows, 'Test DeBERTa aspects',
        )
        roberta_test_aspects = _aspect_probabilities(
            torch, transformers, roberta_tokenizer,
            _aspect_inference_config(config, config.roberta_model),
            _roberta_aspect_paths(config), test_rows, 'Test RoBERTa aspects',
        )
        deberta_test_logits = _polarity_logits(
            torch, transformers, deberta_tokenizer, deberta_polarity_config,
            polarity_paths, test_rows, 'Test DeBERTa polarity',
        )
        roberta_test_logits = _polarity_logits(
            torch, transformers, roberta_tokenizer, roberta_polarity_config,
            _roberta_polarity_paths(config), test_rows, 'Test RoBERTa polarity',
        )
        test_inputs = {
            'deberta-only': (deberta_test_aspects, deberta_test_logits),
            'roberta-aspect-deberta-polarity': (roberta_test_aspects, deberta_test_logits),
            'roberta-aspect-mixed-polarity': (
                roberta_test_aspects,
                average_backbone_logits(torch, roberta_test_logits, deberta_test_logits),
            ),
        }
        test_metrics: dict[str, dict[str, Any]] = {}
        for name in SYSTEMS:
            system_dir = config.run_dir / 'systems' / name
            predictions, _ = _decode(
                test_rows, *test_inputs[name], locked[name]['thresholds']
            )
            test_metrics[name] = _write_evaluation(
                deberta_polarity_config, system_dir / 'test', test_rows, predictions
            )
            locked[name]['test_accessed'] = True
            atomic_write_json(system_dir / 'locked_selection.json', locked[name])
        runtime = time.monotonic() - started
        for name in SYSTEMS:
            system_dir = config.run_dir / 'systems' / name
            atomic_write_json(system_dir / 'results.json', {
                'validation': validation_metrics[name], 'test': test_metrics[name],
                'runtime_seconds': runtime,
            })
            _write_system_report(
                system_dir, name, validation_metrics[name], test_metrics[name], runtime
            )
        atomic_write_json(config.run_dir / 'results.json', {
            'validation': validation_metrics, 'test': test_metrics,
            'runtime_seconds': runtime,
        })
        atomic_write_text(config.run_dir / 'DONE', '0\n')


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    try:
        run_experiments(config_from_args(argv))
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted; rerun the same command to resume completed epochs')
        return 130
    except Exception:
        LOGGER.exception('Three-system DeBERTa experiment failed')
        return 1
    return 0
