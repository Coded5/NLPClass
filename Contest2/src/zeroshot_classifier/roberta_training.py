from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from .checkpoint import ResumeMismatchError, RunLock
from .data import LabeledRow, read_input_items, read_labeled_csv, stratified_group_split
from .evaluation import evaluate
from .io_utils import atomic_write_json, file_sha256
from .labels import ASPECTS, POLARITIES


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    input: Path = Path('data/contest2_train.csv')
    contest_input: Path = Path('data/contest2_test.csv')
    run_dir: Path = Path('artifacts/training/roberta-two-model')
    evaluator: Path = Path('scripts/evaluate.py')
    model: str = 'FacebookAI/roberta-base'
    task: str = 'both'
    train_ratio: float = 0.8
    eval_ratio: float = 0.1
    test_ratio: float = 0.1
    epochs: int = 4
    evaluation_interval: int = 10
    seed: int = 42
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
    mlflow_experiment: str = 'contest2-roberta-two-model'
    mlflow_run_name: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train resumable sequential RoBERTa aspect and polarity classifiers.',
        allow_abbrev=False,
    )
    parser.add_argument('--input', type=Path, default=TrainingConfig.input)
    parser.add_argument('--contest-input', type=Path, default=TrainingConfig.contest_input)
    parser.add_argument('--run-dir', type=Path, default=TrainingConfig.run_dir)
    parser.add_argument('--evaluator', type=Path, default=TrainingConfig.evaluator)
    parser.add_argument('--model', default=TrainingConfig.model)
    parser.add_argument(
        '--task', choices=('aspect', 'polarity', 'both'), default=TrainingConfig.task
    )
    parser.add_argument('--train-ratio', type=float, default=TrainingConfig.train_ratio)
    parser.add_argument('--eval-ratio', type=float, default=TrainingConfig.eval_ratio)
    parser.add_argument('--test-ratio', type=float, default=TrainingConfig.test_ratio)
    parser.add_argument('--epochs', '--epoch', dest='epochs', type=int, default=TrainingConfig.epochs)
    parser.add_argument(
        '--evaluation-interval', type=int, default=TrainingConfig.evaluation_interval
    )
    parser.add_argument('--seed', type=int, default=TrainingConfig.seed)
    parser.add_argument('--max-length', type=int, default=TrainingConfig.max_length)
    parser.add_argument('--learning-rate', type=float, default=TrainingConfig.learning_rate)
    parser.add_argument('--train-batch-size', type=int, default=TrainingConfig.train_batch_size)
    parser.add_argument('--eval-batch-size', type=int, default=TrainingConfig.eval_batch_size)
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=TrainingConfig.gradient_accumulation_steps,
    )
    parser.add_argument('--weight-decay', type=float, default=TrainingConfig.weight_decay)
    parser.add_argument('--warmup-ratio', type=float, default=TrainingConfig.warmup_ratio)
    parser.add_argument('--checkpoint-steps', type=int, default=TrainingConfig.checkpoint_steps)
    parser.add_argument('--log-steps', type=int, default=TrainingConfig.log_steps)
    parser.add_argument(
        '--mixed-precision',
        choices=('auto', 'no', 'fp16', 'bf16'),
        default=TrainingConfig.mixed_precision,
    )
    parser.add_argument('--mlflow-tracking-uri', default=TrainingConfig.mlflow_tracking_uri)
    parser.add_argument('--mlflow-experiment', default=TrainingConfig.mlflow_experiment)
    parser.add_argument('--mlflow-run-name')
    return parser


def config_from_args(argv: list[str] | None = None) -> TrainingConfig:
    args = build_parser().parse_args(argv)
    config = TrainingConfig(**vars(args))
    validate_config(config)
    return config


def validate_config(config: TrainingConfig) -> None:
    ratios = (config.train_ratio, config.eval_ratio, config.test_ratio)
    if any(ratio <= 0 or ratio >= 1 for ratio in ratios):
        raise ValueError('Train, eval, and test ratios must each be between 0 and 1')
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError('Train, eval, and test ratios must sum to 1')
    positive = {
        'epochs': config.epochs,
        'evaluation-interval': config.evaluation_interval,
        'max-length': config.max_length,
        'train-batch-size': config.train_batch_size,
        'eval-batch-size': config.eval_batch_size,
        'gradient-accumulation-steps': config.gradient_accumulation_steps,
        'log-steps': config.log_steps,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError(f'{", ".join(invalid)} must be at least 1')
    if config.learning_rate <= 0:
        raise ValueError('learning-rate must be greater than 0')
    if not 0 <= config.warmup_ratio < 1:
        raise ValueError('warmup-ratio must be in [0, 1)')
    if config.weight_decay < 0 or config.checkpoint_steps < 0:
        raise ValueError('weight-decay and checkpoint-steps cannot be negative')


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    try:
        config = config_from_args(argv)
        run_training(config)
    except KeyboardInterrupt:
        print('\nInterrupted; resume will use the latest periodic checkpoint.')
        return 130
    except Exception as error:
        print(f'Error: {error}')
        return 1
    return 0


def run_training(config: TrainingConfig) -> None:
    LOGGER.info(
        'Starting training: task=%s epochs=%d evaluation_interval=%d run_dir=%s',
        config.task,
        config.epochs,
        config.evaluation_interval,
        config.run_dir,
    )
    torch, mlflow, transformers = _load_runtime()
    _seed_everything(torch, config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(config.run_dir / '.training.lock'):
        splits, manifest = _prepare_splits(config)
        state_path = config.run_dir / 'state.json'
        state = _read_json(state_path, {'completed_epoch': 0, 'best_epoch': None, 'best_score': -1.0})
        if state['completed_epoch']:
            LOGGER.info('Resuming after epoch %d', state['completed_epoch'])
        _prepare_mlflow_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        run_id_path = config.run_dir / 'mlflow_run.json'
        run_id_data = _read_json(run_id_path, {})
        start_kwargs = {'run_id': run_id_data['run_id']} if run_id_data else {
            'run_name': config.mlflow_run_name
        }
        with mlflow.start_run(**start_kwargs) as active_run:
            LOGGER.info('MLflow run: %s', active_run.info.run_id)
            if not run_id_data:
                atomic_write_json(run_id_path, {'run_id': active_run.info.run_id})
                mlflow.log_params(_mlflow_params(config, manifest))
                mlflow.log_artifact(str(config.run_dir / 'manifest.json'), artifact_path='configuration')
            tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, use_fast=True)
            task_runtimes: dict[str, dict[str, Any]] = {}
            for epoch in range(int(state['completed_epoch']) + 1, config.epochs + 1):
                evaluate_epoch = (
                    epoch % config.evaluation_interval == 0 or epoch == config.epochs
                )
                LOGGER.info(
                    'Epoch %d/%d started%s', epoch, config.epochs,
                    ' with evaluation' if evaluate_epoch else '',
                )
                task_predictions: dict[str, list[int]] = {}
                losses: dict[str, float] = {}
                for task, labels in _selected_tasks(config.task):
                    loss, predictions, task_runtimes[task] = _train_task_epoch(
                        torch, transformers, mlflow, tokenizer, config, task, labels,
                        splits['train'], splits['eval'], epoch, task_runtimes.get(task),
                        evaluate=evaluate_epoch,
                    )
                    losses[task] = loss
                    if predictions is not None:
                        task_predictions[task] = predictions
                for task, loss in losses.items():
                    mlflow.log_metric(f'train/{task}_epoch_loss', loss, step=epoch)
                if evaluate_epoch:
                    evaluation_dir = config.run_dir / 'evaluations' / f'epoch-{epoch:03d}'
                    metrics = _evaluate_predictions(
                        config,
                        splits['eval'],
                        task_predictions.get('aspect'),
                        task_predictions.get('polarity'),
                        evaluation_dir,
                    )
                    score_target = 'overall' if config.task == 'both' else config.task
                    score = float(
                        metrics['source_of_truth_metrics'][score_target]['micro']['f1']
                    )
                    _log_epoch_metrics(mlflow, epoch, {}, metrics)
                    LOGGER.info(
                        'Epoch %d evaluation %s micro-F1: %.6f',
                        epoch,
                        score_target,
                        score,
                    )
                    if score > float(state['best_score']):
                        _save_best_checkpoints(config, epoch)
                        state['best_epoch'] = epoch
                        state['best_score'] = score
                        mlflow.set_tag('best_epoch', str(epoch))
                        LOGGER.info('New best checkpoint: epoch=%d score=%.6f', epoch, score)
                    mlflow.log_artifacts(
                        str(evaluation_dir), artifact_path=f'evaluations/epoch-{epoch:03d}'
                    )
                state['completed_epoch'] = epoch
                atomic_write_json(state_path, state)
                LOGGER.info('Epoch %d/%d completed', epoch, config.epochs)

            task_runtimes.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not state.get('final_complete'):
                LOGGER.info('Starting final CPU evaluation from epoch %s', state['best_epoch'])
                _run_final_evaluation(
                    torch, transformers, mlflow, tokenizer, config, splits, state
                )
                state['final_complete'] = True
                atomic_write_json(state_path, state)
                LOGGER.info('Training and final evaluation completed')


def _prepare_splits(config: TrainingConfig) -> tuple[dict[str, list[LabeledRow]], dict[str, Any]]:
    rows = read_labeled_csv(config.input)
    manifest_path = config.run_dir / 'manifest.json'
    requested = {
        'format_version': 1,
        'input': str(config.input.resolve()),
        'input_sha256': file_sha256(config.input),
        'model': config.model,
        'task': config.task,
        'ratios': {
            'train': config.train_ratio,
            'eval': config.eval_ratio,
            'test': config.test_ratio,
        },
        'epochs': config.epochs,
        'seed': config.seed,
        'max_length': config.max_length,
        'learning_rate': config.learning_rate,
        'train_batch_size': config.train_batch_size,
        'eval_batch_size': config.eval_batch_size,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'weight_decay': config.weight_decay,
        'warmup_ratio': config.warmup_ratio,
        'mixed_precision': config.mixed_precision,
        'labels': {'aspect': list(ASPECTS), 'polarity': list(POLARITIES)},
    }
    if manifest_path.exists():
        existing = _read_json(manifest_path, {})
        # Runs created before --task existed always trained both models.
        existing.setdefault('task', 'both')
        comparable = {key: existing.get(key) for key in requested}
        if comparable != requested:
            raise ResumeMismatchError(
                'Run configuration or input changed; choose a new --run-dir'
            )
        return {
            name: read_labeled_csv(config.run_dir / 'splits' / f'{name}.csv')
            for name in ('train', 'eval', 'test')
        }, existing

    outer = stratified_group_split(rows, test_ratio=config.test_ratio, seed=config.seed)
    eval_fraction_of_remainder = config.eval_ratio / (config.train_ratio + config.eval_ratio)
    inner = stratified_group_split(
        outer.train_rows, test_ratio=eval_fraction_of_remainder, seed=config.seed + 1
    )
    splits = {
        'train': list(inner.train_rows),
        'eval': list(inner.test_rows),
        'test': list(outer.test_rows),
    }
    split_ids = {name: {row.item_id for row in part} for name, part in splits.items()}
    if split_ids['train'] & split_ids['eval'] or split_ids['train'] & split_ids['test'] or split_ids['eval'] & split_ids['test']:
        raise RuntimeError('Review ID leakage detected while creating splits')
    split_dir = config.run_dir / 'splits'
    for name, part in splits.items():
        _write_labeled_csv(split_dir / f'{name}.csv', part)
    manifest = requested | {
        'rows': {name: len(part) for name, part in splits.items()},
        'ids': {name: len(split_ids[name]) for name in splits},
    }
    atomic_write_json(manifest_path, manifest)
    return splits, manifest


class _TextDataset:
    def __init__(self, rows: Sequence[Any], tokenizer: Any, max_length: int, task: str | None):
        self.items = []
        label_order = ASPECTS if task == 'aspect' else POLARITIES
        label2id = {label: index for index, label in enumerate(label_order)}
        for row in rows:
            encoded = tokenizer(row.text, truncation=True, max_length=max_length)
            if task is not None:
                value = row.aspect if task == 'aspect' else row.polarity
                encoded['labels'] = label2id[value]
            self.items.append(encoded)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def _train_task_epoch(
    torch: Any,
    transformers: Any,
    mlflow: Any,
    tokenizer: Any,
    config: TrainingConfig,
    task: str,
    labels: tuple[str, ...],
    train_rows: list[LabeledRow],
    eval_rows: list[LabeledRow],
    epoch: int,
    runtime: dict[str, Any] | None = None,
    evaluate: bool = True,
) -> tuple[float, list[int] | None, dict[str, Any]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = config.run_dir / 'checkpoints' / task / 'latest.pt'
    start_batch = 0
    loss_sum = 0.0
    loss_count = 0
    if runtime is None:
        label2id = {label: index for index, label in enumerate(labels)}
        id2label = {index: label for label, index in label2id.items()}
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            config.model, num_labels=len(labels), label2id=label2id, id2label=id2label
        )
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        train_dataset = _TextDataset(train_rows, tokenizer, config.max_length, task)
        batches_per_epoch = math.ceil(len(train_dataset) / config.train_batch_size)
        updates_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=round(updates_per_epoch * config.epochs * config.warmup_ratio),
            num_training_steps=updates_per_epoch * config.epochs,
        )
        use_fp16, autocast_dtype = _precision_settings(torch, config.mixed_precision, device)
        scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
        runtime = {
            'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
            'scaler': scaler, 'train_dataset': train_dataset,
            'batches_per_epoch': batches_per_epoch, 'autocast_dtype': autocast_dtype,
            'completed_epoch': 0, 'global_step': 0, 'inference_model': None,
        }
        checkpoint = _load_checkpoint(torch, checkpoint_path, device)
        if checkpoint:
            LOGGER.info('Restoring %s checkpoint from %s', task, checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            _restore_optimizer_state(optimizer, model, checkpoint)
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            runtime['completed_epoch'] = int(checkpoint['completed_epoch'])
            runtime['global_step'] = int(checkpoint['global_step'])
            if int(checkpoint.get('active_epoch', 0)) == epoch:
                start_batch = int(checkpoint.get('next_batch', 0))
                loss_sum = float(checkpoint.get('loss_sum', 0.0))
                loss_count = int(checkpoint.get('loss_count', 0))
            elif runtime['completed_epoch'] >= epoch:
                loss_sum = float(checkpoint.get('epoch_loss', 0.0))
                loss_count = 1
            _restore_rng(torch, checkpoint.get('rng'))

    model = runtime['model']
    optimizer = runtime['optimizer']
    scheduler = runtime['scheduler']
    scaler = runtime['scaler']
    train_dataset = runtime['train_dataset']
    batches_per_epoch = runtime['batches_per_epoch']
    autocast_dtype = runtime['autocast_dtype']
    completed_epoch = int(runtime['completed_epoch'])
    global_step = int(runtime['global_step'])

    started = time.monotonic()
    if completed_epoch < epoch:
        model.train()
        order_generator = torch.Generator().manual_seed(config.seed + epoch * 100 + (0 if task == 'aspect' else 1))
        order = torch.randperm(len(train_dataset), generator=order_generator).tolist()
        remaining = order[start_batch * config.train_batch_size:]
        subset = torch.utils.data.Subset(train_dataset, remaining)
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=config.train_batch_size,
            shuffle=False,
            collate_fn=transformers.DataCollatorWithPadding(tokenizer=tokenizer),
        )
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            loader,
            desc=f'Train {task} epoch {epoch}/{config.epochs}',
            initial=start_batch,
            total=batches_per_epoch,
            unit='batch',
            dynamic_ncols=True,
        )
        for relative_batch, batch in enumerate(progress):
            batch_index = start_batch + relative_batch
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                output = model(**batch)
                loss = output.loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            raw_loss = float(loss.detach().cpu()) * config.gradient_accumulation_steps
            loss_sum += raw_loss
            loss_count += 1
            progress.set_postfix(loss=f'{loss_sum / loss_count:.4f}')
            is_update = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == batches_per_epoch
            )
            if not is_update:
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if global_step % config.log_steps == 0:
                mlflow.log_metric(f'train/{task}_loss', raw_loss, step=global_step)
                mlflow.log_metric(
                    f'train/{task}_learning_rate', scheduler.get_last_lr()[0], step=global_step
                )
            if config.checkpoint_steps and global_step % config.checkpoint_steps == 0:
                _save_checkpoint(
                    torch, checkpoint_path, model, optimizer, scheduler, scaler,
                    completed_epoch=epoch - 1, active_epoch=epoch,
                    next_batch=batch_index + 1, global_step=global_step,
                    loss_sum=loss_sum, loss_count=loss_count,
                )
        completed_epoch = epoch
        epoch_loss = loss_sum / loss_count if loss_count else 0.0
        _save_checkpoint(
            torch, checkpoint_path, model, optimizer, scheduler, scaler,
            completed_epoch=epoch, active_epoch=0, next_batch=0, global_step=global_step,
            epoch_loss=epoch_loss,
        )
        LOGGER.info('Saved %s checkpoint for epoch %d', task, epoch)
        runtime['completed_epoch'] = completed_epoch
        runtime['global_step'] = global_step
    predictions = None
    if evaluate:
        inference_model = runtime['inference_model']
        if inference_model is None:
            LOGGER.info('Loading reusable CPU inference model for %s', task)
            inference_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                config.model, num_labels=len(labels), label2id=model.config.label2id,
                id2label=model.config.id2label,
            )
            runtime['inference_model'] = inference_model
        inference_model.load_state_dict(model.state_dict())
        predictions = _predict(
            torch, transformers, inference_model, tokenizer, eval_rows, config,
            torch.device('cpu'), f'Evaluate {task} epoch {epoch}/{config.epochs}',
        )
    duration = time.monotonic() - started
    mlflow.log_metric(f'train/{task}_epoch_seconds', duration, step=epoch)
    mean_loss = loss_sum / loss_count if loss_count else 0.0
    return mean_loss, predictions, runtime


def _predict(
    torch: Any,
    transformers: Any,
    model: Any,
    tokenizer: Any,
    rows: Sequence[Any],
    config: TrainingConfig,
    device: Any,
    progress_label: str = 'Inference',
) -> list[int]:
    dataset = _TextDataset(rows, tokenizer, config.max_length, task=None)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=transformers.DataCollatorWithPadding(tokenizer=tokenizer),
    )
    predictions: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            loader, desc=progress_label, unit='batch', dynamic_ncols=True
        ):
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return predictions


def _evaluate_predictions(
    config: TrainingConfig,
    gold_rows: Sequence[LabeledRow],
    aspect_predictions: Sequence[int] | None,
    polarity_predictions: Sequence[int] | None,
    output_dir: Path,
) -> dict[str, Any]:
    if aspect_predictions is not None and len(gold_rows) != len(aspect_predictions):
        raise RuntimeError('Aspect prediction count does not match gold row count')
    if polarity_predictions is not None and len(gold_rows) != len(polarity_predictions):
        raise RuntimeError('Polarity prediction count does not match gold row count')
    gold_path = output_dir / 'gold.csv'
    prediction_path = output_dir / 'predictions.csv'
    _write_labeled_csv(gold_path, gold_rows)
    records = []
    for index, row in enumerate(gold_rows):
        aspect = row.aspect if aspect_predictions is None else ASPECTS[aspect_predictions[index]]
        polarity = (
            row.polarity
            if polarity_predictions is None
            else POLARITIES[polarity_predictions[index]]
        )
        records.append((row.item_id, aspect, polarity))
    _write_predictions(prediction_path, records)
    return evaluate(
        gold_path,
        prediction_path,
        config.evaluator,
        output_dir / 'metrics.json',
        output_dir / 'official_evaluation.txt',
    )


def _run_final_evaluation(
    torch: Any,
    transformers: Any,
    mlflow: Any,
    tokenizer: Any,
    config: TrainingConfig,
    splits: dict[str, list[LabeledRow]],
    state: dict[str, Any],
) -> None:
    if state.get('best_epoch') is None:
        raise RuntimeError('No completed epoch is available for final evaluation')
    contest_items = read_input_items(config.contest_input)
    predictions: dict[str, dict[str, list[int]]] = {'test': {}, 'contest': {}}
    for task, labels in _selected_tasks(config.task):
        device = torch.device('cpu')
        label2id = {label: index for index, label in enumerate(labels)}
        id2label = {index: label for label, index in label2id.items()}
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            config.model, num_labels=len(labels), label2id=label2id, id2label=id2label
        )
        checkpoint = _load_checkpoint(
            torch, config.run_dir / 'best' / f'{task}.pt', device
        )
        if not checkpoint:
            raise RuntimeError(f'Best {task} checkpoint is missing')
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        predictions['test'][task] = _predict(
            torch, transformers, model, tokenizer, splits['test'], config, device,
            f'Final test inference ({task})',
        )
        predictions['contest'][task] = _predict(
            torch, transformers, model, tokenizer, contest_items, config, device,
            f'Contest inference ({task})',
        )
        del model, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_dir = config.run_dir / 'final'
    metrics = _evaluate_predictions(
        config,
        splits['test'],
        predictions['test'].get('aspect'),
        predictions['test'].get('polarity'),
        final_dir,
    )
    _write_contest_predictions(
        final_dir / 'contest_predictions.csv', contest_items, predictions['contest']
    )
    _log_metric_tree(mlflow, 'test', metrics['source_of_truth_metrics'], step=int(state['best_epoch']))
    mlflow.log_artifacts(str(final_dir), artifact_path='final')


def _save_checkpoint(
    torch: Any,
    path: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    **progress: int | float,
) -> None:
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_param_names': [name for name, _ in model.named_parameters()],
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'rng': _capture_rng(torch),
        **progress,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix='.checkpoint-', suffix='.pt')
    os.close(descriptor)
    try:
        torch.save(payload, temporary_name)
        with open(temporary_name, 'rb') as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_checkpoint(torch: Any, path: Path, device: Any) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return torch.load(path, map_location=device, weights_only=False)


def _restore_optimizer_state(
    optimizer: Any,
    model: Any,
    checkpoint: dict[str, Any],
) -> None:
    saved = checkpoint['optimizer']
    saved_groups = saved['param_groups']
    if len(saved_groups) != 1 or len(optimizer.param_groups) != 1:
        raise RuntimeError('Only one optimizer parameter group is supported')

    saved_ids = saved_groups[0]['params']
    saved_names = checkpoint.get('optimizer_param_names')
    if saved_names is None:
        # Checkpoints written before parameter names were persisted used the
        # model state-dict registration order. This migrates those checkpoints.
        saved_names = list(checkpoint['model'])
    if len(saved_names) != len(saved_ids):
        raise RuntimeError('Optimizer checkpoint parameter metadata is inconsistent')

    saved_state_by_name = {
        name: saved['state'].get(parameter_id, {})
        for name, parameter_id in zip(saved_names, saved_ids)
    }
    current_state = optimizer.state_dict()
    current_ids = current_state['param_groups'][0]['params']
    current_names = [name for name, _ in model.named_parameters()]
    if len(current_names) != len(current_ids):
        raise RuntimeError('Current optimizer parameter metadata is inconsistent')
    missing = set(current_names) - set(saved_state_by_name)
    if missing:
        raise RuntimeError(
            f'Optimizer checkpoint is missing parameters: {", ".join(sorted(missing)[:3])}'
        )

    remapped_group = dict(saved_groups[0])
    remapped_group['params'] = current_ids
    remapped = {
        'state': {
            parameter_id: saved_state_by_name[name]
            for name, parameter_id in zip(current_names, current_ids)
        },
        'param_groups': [remapped_group],
    }
    optimizer.load_state_dict(remapped)


def _save_best_checkpoints(config: TrainingConfig, epoch: int) -> None:
    best_dir = config.run_dir / 'best'
    best_dir.mkdir(parents=True, exist_ok=True)
    for task, _ in _selected_tasks(config.task):
        source = config.run_dir / 'checkpoints' / task / 'latest.pt'
        destination = best_dir / f'{task}.pt'
        temporary = best_dir / f'.{task}.epoch-{epoch}.tmp'
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    atomic_write_json(best_dir / 'manifest.json', {'epoch': epoch})


def _capture_rng(torch: Any) -> dict[str, Any]:
    state = {'python': random.getstate(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(torch: Any, state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].detach().cpu())
    if torch.cuda.is_available() and 'cuda' in state:
        # Checkpoints are loaded onto the training device so optimizer tensors
        # resume correctly, but CUDA's RNG API specifically requires CPU byte tensors.
        cuda_states = [item.detach().cpu() for item in state['cuda']]
        torch.cuda.set_rng_state_all(cuda_states)


def _precision_settings(torch: Any, setting: str, device: Any) -> tuple[bool, Any | None]:
    if device.type != 'cuda':
        if setting in ('fp16', 'bf16'):
            raise ValueError(f'{setting} mixed precision requires a CUDA device')
        return False, None
    chosen = 'fp16' if setting == 'auto' else setting
    if chosen == 'no':
        return False, None
    if chosen == 'bf16':
        if not torch.cuda.is_bf16_supported():
            raise ValueError('This CUDA device does not support bf16')
        return False, torch.bfloat16
    return True, torch.float16


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_labeled_csv(path: Path, rows: Sequence[LabeledRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=path.parent, delete=False) as handle:
        writer = csv.writer(handle)
        writer.writerow(('id', 'text', 'aspectCategory', 'polarity'))
        writer.writerows((row.item_id, row.text, row.aspect, row.polarity) for row in rows)
        temporary = handle.name
    os.replace(temporary, path)


def _write_predictions(
    path: Path,
    records: Sequence[tuple[str, str, str]],
    include_text: Sequence[Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=path.parent, delete=False) as handle:
        writer = csv.writer(handle)
        if include_text is None:
            writer.writerow(('id', 'aspectCategory', 'polarity'))
            writer.writerows(records)
        else:
            writer.writerow(('id', 'text', 'aspectCategory', 'polarity'))
            writer.writerows(
                (item_id, item.text, aspect, polarity)
                for (item_id, aspect, polarity), item in zip(records, include_text)
            )
        temporary = handle.name
    os.replace(temporary, path)


def _write_contest_predictions(
    path: Path,
    items: Sequence[Any],
    predictions: dict[str, list[int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['id', 'text']
    if 'aspect' in predictions:
        columns.append('aspectCategory')
    if 'polarity' in predictions:
        columns.append('polarity')
    with tempfile.NamedTemporaryFile(
        'w', encoding='utf-8', newline='', dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, item in enumerate(items):
            row = {'id': item.item_id, 'text': item.text}
            if 'aspect' in predictions:
                row['aspectCategory'] = ASPECTS[predictions['aspect'][index]]
            if 'polarity' in predictions:
                row['polarity'] = POLARITIES[predictions['polarity'][index]]
            writer.writerow(row)
        temporary = handle.name
    os.replace(temporary, path)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return dict(default)
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _mlflow_params(config: TrainingConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    params = asdict(config)
    params.pop('mlflow_tracking_uri', None)
    params.pop('mlflow_run_name', None)
    params.update({f'{name}_rows': count for name, count in manifest['rows'].items()})
    return {key: str(value) if isinstance(value, Path) else value for key, value in params.items()}


def _log_epoch_metrics(
    mlflow: Any,
    epoch: int,
    losses: dict[str, float],
    metrics: dict[str, Any],
) -> None:
    for task, loss in losses.items():
        mlflow.log_metric(f'train/{task}_epoch_loss', loss, step=epoch)
    _log_metric_tree(mlflow, 'eval', metrics['source_of_truth_metrics'], step=epoch)
    for name, value in metrics['supplemental_exact_match_accuracy'].items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(f'eval/exact_match_{name}', value, step=epoch)


def _log_metric_tree(mlflow: Any, prefix: str, tree: dict[str, Any], step: int) -> None:
    for target, report in tree.items():
        if target == 'overall':
            mlflow.log_metric(f'{prefix}/overall_micro_f1', report['micro']['f1'], step=step)
            continue
        for average in ('micro', 'macro'):
            for metric in ('precision', 'recall', 'f1'):
                mlflow.log_metric(
                    f'{prefix}/{target}_{average}_{metric}',
                    report[average][metric],
                    step=step,
                )


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        import mlflow
        import torch
        import transformers
    except ImportError as error:
        raise RuntimeError(
            'Training dependencies are missing; run `python -m pip install -e ".[training]"`'
        ) from error
    return torch, mlflow, transformers


def _selected_tasks(task: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if task == 'aspect':
        return (('aspect', ASPECTS),)
    if task == 'polarity':
        return (('polarity', POLARITIES),)
    return (('aspect', ASPECTS), ('polarity', POLARITIES))


def _prepare_mlflow_tracking_uri(tracking_uri: str) -> None:
    prefix = 'sqlite:///'
    if not tracking_uri.startswith(prefix):
        return
    database_path = tracking_uri[len(prefix):]
    if not database_path or database_path == ':memory:':
        return
    Path(database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
