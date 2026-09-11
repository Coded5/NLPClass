from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from .absa_training import (
    AspectDataset,
    ExperimentConfig as TrainingConfig,
    PolarityDataset,
    _aspect_positive_weights,
    _attach_tokenizer,
    _polarity_weights,
    _predict_logits,
    group_aspect_targets,
    train_stage,
)
from .checkpoint import RunLock
from .data import LabeledRow, read_labeled_csv
from .evaluation import calculate_metrics
from .imbalance_losses import LossSpec
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .joint_absa_experiment import (
    JointExperimentConfig,
    _ensemble_predictions,
    _pair_class_metrics,
    _separate_logits,
    all_candidate_rows,
    train_separate_polarity,
)
from .labels import ASPECTS, POLARITIES
from .multilabel_aspect_experiment import (
    AspectExperimentConfig,
    _predict_checkpoint_probabilities,
    aspect_predictions,
    calculate_aspect_metrics,
    tune_thresholds,
)
from .roberta_training import (
    _precision_settings,
    _prepare_mlflow_tracking_uri,
    _seed_everything,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DebertaExperimentConfig:
    split_dir: Path = Path('artifacts/training/roberta-aspect-exp1/splits')
    run_dir: Path = Path('artifacts/experiments/deberta-v3-current-best-v1')
    model: str = 'microsoft/deberta-v3-base'
    seed: int = 42
    epochs: int = 50
    evaluation_interval: int = 5
    early_stopping_patience: int = 3
    max_length: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    candidate_batch_sizes: tuple[int, ...] = (8, 4, 2)
    effective_batch_size: int = 16
    eval_batch_size: int = 16
    benchmark_seconds_per_stage: int = 300
    maximum_projected_minutes: float = 120.0
    mixed_precision: str = 'auto'
    mlflow_tracking_uri: str = 'sqlite:////home/kami/Projects/NLP/Contest2/artifacts/mlflow.db'
    mlflow_experiment: str = 'contest2-deberta-v3-current-best-v1'


def load_tokenizer(transformers: Any, model: str) -> Any:
    return transformers.AutoTokenizer.from_pretrained(
        model, use_fast=True, fix_mistral_regex=True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Benchmark and train one validation-only DeBERTa-v3 ABSA seed.'
    )
    for name, field in DebertaExperimentConfig.__dataclass_fields__.items():
        default = field.default
        option = f'--{name.replace("_", "-")}'
        if name == 'candidate_batch_sizes':
            parser.add_argument(option, type=int, nargs='+', default=list(default))
        elif isinstance(default, Path):
            parser.add_argument(option, type=Path, default=default)
        else:
            parser.add_argument(option, type=type(default), default=default)
    return parser


def config_from_args(argv: list[str] | None = None) -> DebertaExperimentConfig:
    values = vars(build_parser().parse_args(argv))
    values['candidate_batch_sizes'] = tuple(values['candidate_batch_sizes'])
    config = DebertaExperimentConfig(**values)
    positive = (
        config.seed, config.epochs, config.evaluation_interval,
        config.early_stopping_patience, config.max_length,
        config.effective_batch_size, config.eval_batch_size,
        config.benchmark_seconds_per_stage, config.maximum_projected_minutes,
    )
    if any(value <= 0 for value in positive):
        raise ValueError('Seed, training, benchmark, and time-limit values must be positive')
    if not config.candidate_batch_sizes or any(value <= 0 for value in config.candidate_batch_sizes):
        raise ValueError('Candidate batch sizes must be positive')
    if any(config.effective_batch_size % value for value in config.candidate_batch_sizes):
        raise ValueError('Every candidate batch size must divide the effective batch size')
    return config


def gradient_accumulation(effective_batch_size: int, physical_batch_size: int) -> int:
    if physical_batch_size <= 0 or effective_batch_size % physical_batch_size:
        raise ValueError('Physical batch size must divide the effective batch size')
    return effective_batch_size // physical_batch_size


def projected_minutes(
    examples_per_second: float, aspect_examples: int, polarity_examples: int,
    epochs: int,
) -> float:
    if examples_per_second <= 0:
        raise ValueError('Throughput must be positive')
    return epochs * (aspect_examples + polarity_examples) / examples_per_second / 60


def should_continue(benchmark: dict[str, Any], maximum_minutes: float) -> bool:
    stages = benchmark.get('stages', {})
    return (
        set(stages) == {'aspect', 'polarity'}
        and all(value.get('finite_loss') and value.get('optimizer_steps', 0) > 0 for value in stages.values())
        and float(benchmark.get('projected_one_seed_max_minutes', math.inf)) <= maximum_minutes
    )


def _datasets(
    rows: Sequence[LabeledRow], tokenizer: Any, max_length: int,
) -> dict[str, Any]:
    return {
        'aspect': _attach_tokenizer(
            AspectDataset(group_aspect_targets(rows), tokenizer, max_length), tokenizer
        ),
        'polarity': _attach_tokenizer(
            PolarityDataset(rows, tokenizer, max_length), tokenizer
        ),
    }


def _benchmark_stage(
    torch: Any, transformers: Any, tokenizer: Any, config: DebertaExperimentConfig,
    stage: str, dataset: Any,
) -> dict[str, Any]:
    labels = ASPECTS if stage == 'aspect' else POLARITIES
    device = torch.device('cuda')
    last_error = ''
    for checkpointing in (False, True):
        for batch_size in config.candidate_batch_sizes:
            model = None
            try:
                torch.cuda.empty_cache()
                model = transformers.AutoModelForSequenceClassification.from_pretrained(
                    config.model, num_labels=len(labels)
                ).float().to(device)
                if checkpointing:
                    model.gradient_checkpointing_enable()
                accumulation = gradient_accumulation(config.effective_batch_size, batch_size)
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True,
                    collate_fn=transformers.DataCollatorWithPadding(tokenizer=tokenizer),
                )
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                loss_fn = (
                    torch.nn.BCEWithLogitsLoss(
                        pos_weight=_aspect_positive_weights(
                            torch, group_aspect_targets(read_labeled_csv(config.split_dir / 'train.csv'))
                        ).to(device)
                    ) if stage == 'aspect' else torch.nn.CrossEntropyLoss(
                        weight=_polarity_weights(
                            torch, read_labeled_csv(config.split_dir / 'train.csv')
                        ).to(device)
                    )
                )
                use_fp16, dtype = _precision_settings(torch, config.mixed_precision, device)
                scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
                model.train(); optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic(); examples = optimizer_steps = batches = 0
                running_loss = 0.0
                progress = tqdm(
                    total=config.benchmark_seconds_per_stage,
                    desc=f'Benchmark {stage} b={batch_size}', unit='s', dynamic_ncols=True,
                )
                while time.monotonic() - started < config.benchmark_seconds_per_stage:
                    for batch in loader:
                        targets = batch.pop('labels').to(device)
                        inputs = {key: value.to(device) for key, value in batch.items()}
                        with torch.autocast('cuda', dtype=dtype, enabled=dtype is not None):
                            loss = loss_fn(model(**inputs).logits, targets) / accumulation
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError('Non-finite benchmark loss')
                        scaler.scale(loss).backward()
                        running_loss += float(loss.detach().cpu()) * accumulation
                        batches += 1; examples += int(targets.shape[0])
                        if batches % accumulation == 0:
                            scaler.step(optimizer); scaler.update()
                            optimizer.zero_grad(set_to_none=True); optimizer_steps += 1
                        elapsed = time.monotonic() - started
                        progress.n = min(int(elapsed), config.benchmark_seconds_per_stage)
                        progress.set_postfix(examples=examples, steps=optimizer_steps)
                        progress.refresh()
                        if elapsed >= config.benchmark_seconds_per_stage:
                            break
                progress.close(); torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                return {
                    'physical_batch_size': batch_size,
                    'gradient_accumulation_steps': accumulation,
                    'gradient_checkpointing': checkpointing,
                    'elapsed_seconds': elapsed,
                    'examples': examples,
                    'examples_per_second': examples / elapsed,
                    'optimizer_steps': optimizer_steps,
                    'optimizer_steps_per_second': optimizer_steps / elapsed,
                    'mean_batch_loss': running_loss / batches,
                    'finite_loss': True,
                    'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
                    'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2 ** 20,
                }
            except torch.OutOfMemoryError as error:
                last_error = str(error)
                LOGGER.warning(
                    'OOM benchmarking %s at batch %d checkpointing=%s',
                    stage, batch_size, checkpointing,
                )
            finally:
                if model is not None:
                    del model
                gc.collect(); torch.cuda.empty_cache()
    raise RuntimeError(f'DeBERTa {stage} does not fit at batch size 2: {last_error}')


def _training_configs(
    config: DebertaExperimentConfig, batch_size: int,
) -> tuple[TrainingConfig, JointExperimentConfig]:
    accumulation = gradient_accumulation(config.effective_batch_size, batch_size)
    common = {
        'split_dir': config.split_dir, 'model': config.model,
        'epochs': config.epochs, 'evaluation_interval': config.evaluation_interval,
        'early_stopping_patience': config.early_stopping_patience,
        'max_length': config.max_length, 'learning_rate': config.learning_rate,
        'train_batch_size': batch_size, 'eval_batch_size': config.eval_batch_size,
        'gradient_accumulation_steps': accumulation,
        'weight_decay': config.weight_decay, 'warmup_ratio': config.warmup_ratio,
        'mixed_precision': config.mixed_precision,
        'mlflow_tracking_uri': config.mlflow_tracking_uri,
        'mlflow_experiment': config.mlflow_experiment,
    }
    aspect = TrainingConfig(
        **common, run_dir=config.run_dir / 'aspect', seed=config.seed,
        mlflow_run_name='deberta-v3-aspect-seed-42',
    )
    polarity = JointExperimentConfig(
        **common, run_dir=config.run_dir / 'polarity',
    )
    return aspect, polarity


def _run_one_seed(
    torch: Any, transformers: Any, mlflow: Any, tokenizer: Any,
    config: DebertaExperimentConfig, train_rows: list[LabeledRow],
    validation_rows: list[LabeledRow], benchmark: dict[str, Any],
) -> dict[str, Any]:
    batch_size = min(
        int(benchmark['stages']['aspect']['physical_batch_size']),
        int(benchmark['stages']['polarity']['physical_batch_size']),
    )
    aspect_config, polarity_config = _training_configs(config, batch_size)
    _seed_everything(torch, config.seed)
    with mlflow.start_run(run_name='deberta-v3-aspect-seed-42'):
        aspect_selection = train_stage(
            torch, transformers, mlflow, tokenizer, aspect_config,
            'aspect', train_rows, validation_rows,
        )
    aspect_checkpoint = aspect_config.run_dir / 'best/aspect.pt'
    aspect_probabilities = _predict_checkpoint_probabilities(
        torch, transformers, tokenizer,
        replace(
            AspectExperimentConfig(),
            model=config.model, split_dir=config.split_dir,
            run_dir=config.run_dir / 'aspect', seeds=(config.seed,),
            eval_batch_size=config.eval_batch_size,
        ),
        aspect_checkpoint, validation_rows, 'Validate DeBERTa aspect',
    )
    aspect_thresholds = tune_thresholds(
        aspect_probabilities,
        [[int(index in review.aspects) for index in range(len(ASPECTS))]
         for review in group_aspect_targets(validation_rows)],
    )
    aspect_metrics = calculate_aspect_metrics(
        validation_rows,
        aspect_predictions(validation_rows, aspect_probabilities, aspect_thresholds),
    )
    polarity_result = train_separate_polarity(
        torch, transformers, mlflow, tokenizer, polarity_config,
        polarity_config.run_dir, LossSpec('weighted-ce'), config.seed,
        train_rows, validation_rows, aspect_probabilities,
    )
    predictions, pair_thresholds = _ensemble_predictions(
        torch, transformers, tokenizer, polarity_config, 'separate',
        [polarity_config.run_dir / 'best.pt'], validation_rows,
        aspect_probabilities, polarity_result['thresholds'],
    )
    metrics = calculate_metrics(validation_rows, predictions)
    return {
        'seed': config.seed, 'test_accessed': False,
        'batch_size': batch_size,
        'gradient_accumulation_steps': gradient_accumulation(
            config.effective_batch_size, batch_size
        ),
        'aspect_selection': aspect_selection,
        'aspect_thresholds': aspect_thresholds,
        'aspect_validation': aspect_metrics,
        'polarity_selection': polarity_result,
        'pair_thresholds': pair_thresholds,
        'composed_validation': metrics,
    }


def run_experiment(config: DebertaExperimentConfig) -> None:
    import mlflow
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError('The DeBERTa feasibility experiment requires CUDA')
    config.run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(config.run_dir / '.experiment.lock'):
        train_rows = read_labeled_csv(config.split_dir / 'train.csv')
        validation_rows = read_labeled_csv(config.split_dir / 'eval.csv')
        if {row.item_id for row in train_rows} & {row.item_id for row in validation_rows}:
            raise RuntimeError('Train and validation IDs overlap')
        manifest = {
            **{key: str(value) if isinstance(value, Path) else value
               for key, value in asdict(config).items()},
            'candidate_batch_sizes': list(config.candidate_batch_sizes),
            'split_sha256': {
                name: file_sha256(config.split_dir / f'{name}.csv')
                for name in ('train', 'eval')
            },
            'test_accessed': False,
        }
        manifest_path = config.run_dir / 'manifest.json'
        if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Experiment manifest mismatch')
        atomic_write_json(manifest_path, manifest)
        tokenizer = load_tokenizer(transformers, config.model)
        _prepare_mlflow_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        datasets = _datasets(train_rows, tokenizer, config.max_length)
        benchmark_path = config.run_dir / 'benchmark.json'
        if benchmark_path.is_file():
            benchmark = json.loads(benchmark_path.read_text())
        else:
            stages = {
                stage: _benchmark_stage(
                    torch, transformers, tokenizer, config, stage, dataset
                ) for stage, dataset in datasets.items()
            }
            conservative_throughput = min(
                float(value['examples_per_second']) for value in stages.values()
            )
            benchmark = {
                'stages': stages,
                'projected_one_seed_max_minutes': projected_minutes(
                    conservative_throughput, len(datasets['aspect']),
                    len(datasets['polarity']), config.epochs,
                ),
            }
            benchmark['feasible'] = should_continue(
                benchmark, config.maximum_projected_minutes
            )
            atomic_write_json(benchmark_path, benchmark)
        if not should_continue(benchmark, config.maximum_projected_minutes):
            atomic_write_text(config.run_dir / 'STOPPED', 'Benchmark did not meet feasibility criteria.\n')
            LOGGER.warning('Benchmark did not meet feasibility criteria; stopping')
            return
        result_path = config.run_dir / 'results.json'
        if not result_path.is_file():
            started = time.monotonic()
            result = _run_one_seed(
                torch, transformers, mlflow, tokenizer, config,
                train_rows, validation_rows, benchmark,
            )
            result['runtime_seconds'] = time.monotonic() - started
            atomic_write_json(result_path, result)
            overall = result['composed_validation']['source_of_truth_metrics']['overall']['micro']['f1']
            polarity_classes = result['polarity_selection']['validation_polarity_classes']
            polarity_macro = sum(
                float(polarity_classes[label]['f1']) for label in POLARITIES
            ) / len(POLARITIES)
            atomic_write_text(config.run_dir / 'report.md', f'''# DeBERTa-v3 current-best pilot

Model: `{config.model}`. Seed: `{config.seed}`. Test accessed: `false`.

Benchmark projected one-seed maximum: {benchmark['projected_one_seed_max_minutes']:.1f} minutes.
Aspect validation micro-F1: {result['aspect_validation']['aspect']['micro']['f1']:.6f}.
Composed validation pair micro-F1: {overall:.6f}.
Polarity oracle-aspect macro-F1: {polarity_macro:.6f}.
Neutral F1: {polarity_classes['neutral']['f1']:.6f}.
Conflict F1: {polarity_classes['conflict']['f1']:.6f}.

The experiment stops after one validation-only seed. A three-seed run requires a separate decision.
''')
        atomic_write_text(config.run_dir / 'DONE', '0\n')


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    try:
        run_experiment(config_from_args(argv))
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted; rerun the same command to resume')
        return 130
    except Exception:
        LOGGER.exception('DeBERTa experiment failed')
        return 1
    return 0
