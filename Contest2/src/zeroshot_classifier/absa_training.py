from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from .checkpoint import RunLock
from .data import LabeledRow, read_labeled_csv
from .evaluation import evaluate
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .labels import ASPECTS, POLARITIES
from .roberta_training import (
    _prepare_mlflow_tracking_uri,
    _precision_settings,
    _seed_everything,
    _write_labeled_csv,
    _write_predictions,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    input: Path = Path('data/contest2_train.csv')
    split_dir: Path = Path('artifacts/training/roberta-aspect-exp1/splits')
    run_dir: Path = Path('artifacts/experiments/multilabel-conditioned-v1')
    baseline_aspect_run: Path = Path('artifacts/training/roberta-aspect-exp1')
    baseline_polarity_run: Path = Path('artifacts/training/roberta-polarity-exp2')
    evaluator: Path = Path('scripts/evaluate.py')
    model: str = 'FacebookAI/roberta-base'
    epochs: int = 50
    evaluation_interval: int = 5
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
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
    mlflow_experiment: str = 'contest2-roberta-multilabel-conditioned'
    mlflow_run_name: str | None = 'multilabel-conditioned-v1'


@dataclass(frozen=True)
class ReviewTarget:
    item_id: str
    text: str
    aspects: tuple[int, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train multilabel aspect detection and aspect-conditioned polarity.'
    )
    for name, field in ExperimentConfig.__dataclass_fields__.items():
        default = field.default
        option = f'--{name.replace("_", "-")}'
        if isinstance(default, Path):
            parser.add_argument(option, type=Path, default=default)
        elif isinstance(default, bool):
            parser.add_argument(option, action='store_true', default=default)
        elif default is None or isinstance(default, str):
            parser.add_argument(option, default=default)
        else:
            parser.add_argument(option, type=type(default), default=default)
    return parser


def config_from_args(argv: list[str] | None = None) -> ExperimentConfig:
    config = ExperimentConfig(**vars(build_parser().parse_args(argv)))
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
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
    if config.learning_rate <= 0 or config.early_stopping_min_delta < 0:
        raise ValueError('learning-rate must be positive and early-stopping-min-delta nonnegative')
    if not 0 <= config.warmup_ratio < 1:
        raise ValueError('warmup-ratio must be in [0, 1)')
    if config.checkpoint_steps < 0 or config.weight_decay < 0:
        raise ValueError('checkpoint-steps and weight-decay must be nonnegative')


def group_aspect_targets(rows: Sequence[LabeledRow]) -> list[ReviewTarget]:
    grouped: dict[str, tuple[str, set[int]]] = {}
    for row in rows:
        text, aspects = grouped.setdefault(row.item_id, (row.text, set()))
        if text != row.text:
            raise ValueError(f'ID {row.item_id!r} maps to more than one text')
        aspects.add(ASPECTS.index(row.aspect))
    return [
        ReviewTarget(item_id, text, tuple(sorted(aspects)))
        for item_id, (text, aspects) in grouped.items()
    ]


def decode_aspects(probabilities: Sequence[Sequence[float]], thresholds: Sequence[float]) -> list[list[int]]:
    decoded = []
    for values in probabilities:
        selected = [index for index, value in enumerate(values) if value >= thresholds[index]]
        decoded.append(selected or [max(range(len(values)), key=lambda index: values[index])])
    return decoded


def tune_thresholds(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[Sequence[int]],
) -> list[float]:
    grid = [value / 100 for value in range(10, 91, 5)]
    thresholds = []
    for label_index in range(len(ASPECTS)):
        gold = {index for index, target in enumerate(targets) if target[label_index]}
        best = (0.0, 0.5)
        for threshold in grid:
            predicted = {
                index for index, values in enumerate(probabilities)
                if values[label_index] >= threshold
            }
            correct = len(gold & predicted)
            precision = correct / len(predicted) if predicted else 0.0
            recall = correct / len(gold) if gold else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if f1 > best[0]:
                best = (f1, threshold)
        thresholds.append(best[1])
    return thresholds


class AspectDataset:
    def __init__(self, reviews: Sequence[ReviewTarget], tokenizer: Any, max_length: int):
        self.items = []
        for review in reviews:
            encoded = tokenizer(review.text, truncation=True, max_length=max_length)
            target = [0.0] * len(ASPECTS)
            for index in review.aspects:
                target[index] = 1.0
            encoded['labels'] = target
            self.items.append(encoded)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class PolarityDataset:
    def __init__(self, rows: Sequence[LabeledRow], tokenizer: Any, max_length: int):
        self.items = []
        for row in rows:
            encoded = tokenizer(
                row.text, f'aspect: {row.aspect}', truncation=True, max_length=max_length
            )
            encoded['labels'] = POLARITIES.index(row.polarity)
            self.items.append(encoded)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def _aspect_positive_weights(torch: Any, reviews: Sequence[ReviewTarget]) -> Any:
    positives = [0] * len(ASPECTS)
    for review in reviews:
        for index in review.aspects:
            positives[index] += 1
    return torch.tensor([
        (len(reviews) - count) / count if count else 1.0 for count in positives
    ], dtype=torch.float32)


def _polarity_weights(torch: Any, rows: Sequence[LabeledRow]) -> Any:
    counts = Counter(row.polarity for row in rows)
    total = len(rows)
    return torch.tensor([
        total / (len(POLARITIES) * counts[label]) for label in POLARITIES
    ], dtype=torch.float32)


def _atomic_torch_save(torch: Any, value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.checkpoint-', suffix='.pt')
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _model_state_cpu(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _macro_f1(gold: Sequence[int], predicted: Sequence[int], class_count: int) -> float:
    scores = []
    for label in range(class_count):
        gold_set = {index for index, value in enumerate(gold) if value == label}
        predicted_set = {index for index, value in enumerate(predicted) if value == label}
        correct = len(gold_set & predicted_set)
        precision = correct / len(predicted_set) if predicted_set else 0.0
        recall = correct / len(gold_set) if gold_set else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def _aspect_micro_f1(targets: Sequence[Sequence[int]], predicted: Sequence[Sequence[int]]) -> float:
    gold = {(row, label) for row, values in enumerate(targets) for label, value in enumerate(values) if value}
    guessed = {(row, label) for row, values in enumerate(predicted) for label in values}
    correct = len(gold & guessed)
    precision = correct / len(guessed) if guessed else 0.0
    recall = correct / len(gold) if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _predict_logits(torch: Any, transformers: Any, model: Any, dataset: Any, batch_size: int, label: str) -> Any:
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=transformers.DataCollatorWithPadding(tokenizer=dataset.tokenizer),
    )
    logits = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=label, unit='batch', dynamic_ncols=True):
            batch.pop('labels', None)
            output = model(**{key: value.to(device) for key, value in batch.items()})
            logits.append(output.logits.cpu())
    return torch.cat(logits)


def _attach_tokenizer(dataset: Any, tokenizer: Any) -> Any:
    dataset.tokenizer = tokenizer
    return dataset


def _stage_paths(config: ExperimentConfig, stage: str) -> tuple[Path, Path]:
    return (
        config.run_dir / 'checkpoints' / stage / 'latest.pt',
        config.run_dir / 'best' / f'{stage}.pt',
    )


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    completed_epoch: int,
    active_epoch: int,
    next_batch: int,
    global_step: int,
    best_score: float,
    stale_checks: int,
    loss_sum: float = 0.0,
    loss_count: int = 0,
    finished: bool = False,
) -> dict[str, Any]:
    return {
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
        'completed_epoch': completed_epoch, 'active_epoch': active_epoch,
        'next_batch': next_batch, 'global_step': global_step,
        'best_score': best_score, 'stale_checks': stale_checks,
        'loss_sum': loss_sum, 'loss_count': loss_count, 'finished': finished,
    }


def train_stage(
    torch: Any,
    transformers: Any,
    mlflow: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    stage: str,
    train_rows: list[LabeledRow],
    validation_rows: list[LabeledRow],
) -> dict[str, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if stage == 'aspect':
        train_reviews = group_aspect_targets(train_rows)
        validation_reviews = group_aspect_targets(validation_rows)
        train_dataset = _attach_tokenizer(AspectDataset(train_reviews, tokenizer, config.max_length), tokenizer)
        validation_dataset = _attach_tokenizer(AspectDataset(validation_reviews, tokenizer, config.max_length), tokenizer)
        loss_function = torch.nn.BCEWithLogitsLoss(
            pos_weight=_aspect_positive_weights(torch, train_reviews).to(device)
        )
        labels = ASPECTS
    else:
        train_dataset = _attach_tokenizer(PolarityDataset(train_rows, tokenizer, config.max_length), tokenizer)
        validation_dataset = _attach_tokenizer(PolarityDataset(validation_rows, tokenizer, config.max_length), tokenizer)
        loss_function = torch.nn.CrossEntropyLoss(
            weight=_polarity_weights(torch, train_rows).to(device)
        )
        labels = POLARITIES
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        config.model, num_labels=len(labels), label2id=label2id, id2label=id2label
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    batches_per_epoch = math.ceil(len(train_dataset) / config.train_batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(updates_per_epoch * config.epochs * config.warmup_ratio),
        num_training_steps=updates_per_epoch * config.epochs,
    )
    use_fp16, autocast_dtype = _precision_settings(torch, config.mixed_precision, device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    latest_path, best_path = _stage_paths(config, stage)
    completed_epoch = global_step = stale_checks = 0
    best_score = -1.0
    resume_epoch = resume_batch = 0
    resume_loss_sum = 0.0
    resume_loss_count = 0
    checkpoint = torch.load(latest_path, map_location=device, weights_only=False) if latest_path.is_file() else None
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        completed_epoch = int(checkpoint['completed_epoch'])
        global_step = int(checkpoint['global_step'])
        best_score = float(checkpoint['best_score'])
        stale_checks = int(checkpoint['stale_checks'])
        resume_epoch = int(checkpoint.get('active_epoch', 0))
        resume_batch = int(checkpoint.get('next_batch', 0))
        resume_loss_sum = float(checkpoint.get('loss_sum', 0.0))
        resume_loss_count = int(checkpoint.get('loss_count', 0))
        LOGGER.info('Resuming %s after epoch %d', stage, completed_epoch)
        if checkpoint.get('finished'):
            result = torch.load(best_path, map_location='cpu', weights_only=False)
            del model, optimizer, scheduler, scaler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                'epoch': int(result['epoch']), 'score': float(result['score']),
                'thresholds': result.get('thresholds'),
            }
    for epoch in range(completed_epoch + 1, config.epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(config.seed + epoch + (0 if stage == 'aspect' else 10_000))
        order = torch.randperm(len(train_dataset), generator=generator).tolist()
        start_batch = resume_batch if resume_epoch == epoch else 0
        remaining = order[start_batch * config.train_batch_size:]
        subset = torch.utils.data.Subset(train_dataset, remaining)
        loader = torch.utils.data.DataLoader(
            subset, batch_size=config.train_batch_size, shuffle=False,
            collate_fn=transformers.DataCollatorWithPadding(tokenizer=tokenizer),
        )
        optimizer.zero_grad(set_to_none=True)
        running_loss = resume_loss_sum if resume_epoch == epoch else 0.0
        loss_count = resume_loss_count if resume_epoch == epoch else 0
        progress = tqdm(
            loader, desc=f'Train {stage} {epoch}/{config.epochs}', unit='batch',
            initial=start_batch, total=batches_per_epoch, dynamic_ncols=True,
        )
        for relative_batch, batch in enumerate(progress):
            batch_index = start_batch + relative_batch
            targets = batch.pop('labels').to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits = model(**batch).logits
                loss = loss_function(logits, targets) / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            raw_loss = float(loss.detach().cpu()) * config.gradient_accumulation_steps
            running_loss += raw_loss
            loss_count += 1
            progress.set_postfix(loss=f'{running_loss / loss_count:.4f}')
            update = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == batches_per_epoch
            )
            if not update:
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if global_step % config.log_steps == 0:
                mlflow.log_metric(f'train/{stage}_loss', raw_loss, step=global_step)
            if config.checkpoint_steps and global_step % config.checkpoint_steps == 0:
                _atomic_torch_save(torch, _checkpoint_payload(
                    model, optimizer, scheduler, scaler, epoch - 1, epoch,
                    batch_index + 1, global_step, best_score, stale_checks,
                    running_loss, loss_count,
                ), latest_path)
        epoch_loss = running_loss / loss_count
        mlflow.log_metric(f'train/{stage}_epoch_loss', epoch_loss, step=epoch)
        evaluate_now = epoch % config.evaluation_interval == 0 or epoch == config.epochs
        thresholds = None
        score = None
        if evaluate_now:
            cpu_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                config.model, num_labels=len(labels), label2id=label2id, id2label=id2label
            )
            cpu_model.load_state_dict(_model_state_cpu(model))
            logits = _predict_logits(
                torch, transformers, cpu_model, validation_dataset,
                config.eval_batch_size, f'Validate {stage} {epoch}',
            )
            if stage == 'aspect':
                probabilities = torch.sigmoid(logits).tolist()
                targets = [item['labels'] for item in validation_dataset.items]
                thresholds = tune_thresholds(probabilities, targets)
                score = _aspect_micro_f1(targets, decode_aspects(probabilities, thresholds))
                for index, threshold in enumerate(thresholds):
                    mlflow.log_metric(f'validation/threshold_{ASPECTS[index]}', threshold, step=epoch)
            else:
                predicted = logits.argmax(dim=-1).tolist()
                gold = [int(item['labels']) for item in validation_dataset.items]
                score = _macro_f1(gold, predicted, len(POLARITIES))
            mlflow.log_metric(f'validation/{stage}_selection_f1', score, step=epoch)
            improved = score > best_score + config.early_stopping_min_delta
            if improved:
                best_score = score
                stale_checks = 0
                _atomic_torch_save(torch, {
                    'model': _model_state_cpu(model), 'epoch': epoch,
                    'score': score, 'thresholds': thresholds,
                    'model_name': config.model,
                }, best_path)
                LOGGER.info('New best %s checkpoint: epoch=%d score=%.6f', stage, epoch, score)
            else:
                stale_checks += 1
            del cpu_model, logits
            gc.collect()
        finished = epoch == config.epochs or (
            evaluate_now and stale_checks >= config.early_stopping_patience
        )
        _atomic_torch_save(torch, _checkpoint_payload(
            model, optimizer, scheduler, scaler, epoch, 0, 0, global_step,
            best_score, stale_checks, finished=finished,
        ), latest_path)
        atomic_write_json(config.run_dir / 'state.json', {
            'active_stage': stage, 'completed_epoch': epoch,
            'best_score': best_score, 'stale_checks': stale_checks,
        })
        if finished and epoch < config.epochs:
            LOGGER.info('Early stopping %s at epoch %d', stage, epoch)
            break
    result = torch.load(best_path, map_location='cpu', weights_only=False)
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {'epoch': int(result['epoch']), 'score': float(result['score']), 'thresholds': result.get('thresholds')}


def _pipeline_predictions(
    torch: Any,
    transformers: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    rows: list[LabeledRow],
) -> list[tuple[str, str, str]]:
    reviews = group_aspect_targets(rows)
    aspect_checkpoint = torch.load(config.run_dir / 'best/aspect.pt', map_location='cpu', weights_only=False)
    aspect_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        config.model, num_labels=len(ASPECTS)
    )
    aspect_model.load_state_dict(aspect_checkpoint['model'])
    aspect_dataset = _attach_tokenizer(AspectDataset(reviews, tokenizer, config.max_length), tokenizer)
    aspect_logits = _predict_logits(torch, transformers, aspect_model, aspect_dataset, config.eval_batch_size, 'Pipeline aspects')
    detected = decode_aspects(torch.sigmoid(aspect_logits).tolist(), aspect_checkpoint['thresholds'])
    del aspect_model, aspect_logits, aspect_checkpoint
    conditioned_rows = [
        LabeledRow(review.item_id, review.text, ASPECTS[aspect], POLARITIES[0], position)
        for position, (review, aspects) in enumerate(zip(reviews, detected))
        for aspect in aspects
    ]
    polarity_checkpoint = torch.load(config.run_dir / 'best/polarity.pt', map_location='cpu', weights_only=False)
    polarity_model = transformers.AutoModelForSequenceClassification.from_pretrained(
        config.model, num_labels=len(POLARITIES)
    )
    polarity_model.load_state_dict(polarity_checkpoint['model'])
    polarity_dataset = _attach_tokenizer(PolarityDataset(conditioned_rows, tokenizer, config.max_length), tokenizer)
    polarity_logits = _predict_logits(torch, transformers, polarity_model, polarity_dataset, config.eval_batch_size, 'Pipeline polarities')
    polarities = polarity_logits.argmax(dim=-1).tolist()
    del polarity_model, polarity_logits, polarity_checkpoint
    return [
        (row.item_id, row.aspect, POLARITIES[polarities[index]])
        for index, row in enumerate(conditioned_rows)
    ]


def _legacy_predictions(
    torch: Any,
    transformers: Any,
    tokenizer: Any,
    config: ExperimentConfig,
    rows: list[LabeledRow],
) -> list[tuple[str, str, str]]:
    from .roberta_training import TrainingConfig, _predict

    outputs: dict[str, list[int]] = {}
    for stage, run_dir, labels in (
        ('aspect', config.baseline_aspect_run, ASPECTS),
        ('polarity', config.baseline_polarity_run, POLARITIES),
    ):
        manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
        label2id = {label: index for index, label in enumerate(labels)}
        id2label = {index: label for label, index in label2id.items()}
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            manifest['model'], num_labels=len(labels), label2id=label2id, id2label=id2label
        )
        checkpoint = torch.load(
            run_dir / 'best' / f'{stage}.pt', map_location='cpu', weights_only=False
        )
        model.load_state_dict(checkpoint['model'])
        outputs[stage] = _predict(
            torch, transformers, model, tokenizer, rows,
            TrainingConfig(
                model=manifest['model'], max_length=int(manifest['max_length']),
                eval_batch_size=config.eval_batch_size,
            ),
            torch.device('cpu'), f'Legacy baseline ({stage})',
        )
        del model, checkpoint
        gc.collect()
    return [
        (row.item_id, ASPECTS[outputs['aspect'][index]], POLARITIES[outputs['polarity'][index]])
        for index, row in enumerate(rows)
    ]


def _write_comparison(
    config: ExperimentConfig,
    selections: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    runtime_seconds: float,
) -> None:
    def values(metrics: dict[str, Any]) -> tuple[float, float, float]:
        source = metrics['source_of_truth_metrics']
        return (
            source['aspect']['micro']['f1'], source['polarity']['micro']['f1'],
            source['overall']['micro']['f1'],
        )
    validation = values(validation_metrics)
    test = values(test_metrics)
    baseline = values(baseline_metrics)
    text = f'''# Multilabel aspect and conditioned polarity experiment

## Selected checkpoints

| Stage | Epoch | Validation selection F1 |
|---|---:|---:|
| Aspect | {selections['aspect']['epoch']} | {selections['aspect']['score']:.6f} |
| Polarity | {selections['polarity']['epoch']} | {selections['polarity']['score']:.6f} |

Aspect thresholds: `{json.dumps(selections['aspect']['thresholds'])}`

## Composed pipeline metrics

| Split | Aspect micro-F1 | Polarity micro-F1 | Overall pair micro-F1 | Overall exact-set accuracy |
|---|---:|---:|---:|---:|
| Validation | {validation[0]:.6f} | {validation[1]:.6f} | {validation[2]:.6f} | {validation_metrics['supplemental_exact_match_accuracy']['overall']:.6f} |
| Untouched test | {test[0]:.6f} | {test[1]:.6f} | {test[2]:.6f} | {test_metrics['supplemental_exact_match_accuracy']['overall']:.6f} |
| Legacy baseline on test | {baseline[0]:.6f} | {baseline[1]:.6f} | {baseline[2]:.6f} | {baseline_metrics['supplemental_exact_match_accuracy']['overall']:.6f} |

Overall pair micro-F1 change versus legacy baseline: {test[2] - baseline[2]:+.6f}.

Runtime: {runtime_seconds / 60:.1f} minutes.

The test split was used only after checkpoint and threshold selection. Results come from one seed (`{config.seed}`).
'''
    atomic_write_text(config.run_dir / 'comparison.md', text)


def _validate_splits(config: ExperimentConfig) -> dict[str, list[LabeledRow]]:
    source_rows = read_labeled_csv(config.input)
    splits = {name: read_labeled_csv(config.split_dir / f'{name}.csv') for name in ('train', 'eval', 'test')}
    id_sets = {name: {row.item_id for row in rows} for name, rows in splits.items()}
    if id_sets['train'] & id_sets['eval'] or id_sets['train'] & id_sets['test'] or id_sets['eval'] & id_sets['test']:
        raise RuntimeError('Persisted splits contain ID leakage')
    if set().union(*id_sets.values()) != {row.item_id for row in source_rows}:
        raise RuntimeError('Persisted splits do not cover the source dataset')
    return splits


def run_experiment(config: ExperimentConfig) -> None:
    import mlflow
    import torch
    import transformers

    started = time.monotonic()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(config.run_dir / '.training.lock'):
        splits = _validate_splits(config)
        manifest = {
            **{key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
            'input_sha256': file_sha256(config.input),
            'split_sha256': {name: file_sha256(config.split_dir / f'{name}.csv') for name in splits},
            'rows': {name: len(rows) for name, rows in splits.items()},
            'ids': {name: len({row.item_id for row in rows}) for name, rows in splits.items()},
            'labels': {'aspect': list(ASPECTS), 'polarity': list(POLARITIES)},
        }
        manifest_path = config.run_dir / 'manifest.json'
        if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Run manifest differs from the requested experiment configuration')
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(config.run_dir / 'config.json', {
            key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()
        })
        _seed_everything(torch, config.seed)
        _prepare_mlflow_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        run_id_path = config.run_dir / 'mlflow_run.json'
        run_data = json.loads(run_id_path.read_text()) if run_id_path.is_file() else {}
        start_kwargs = {'run_id': run_data['run_id']} if run_data else {'run_name': config.mlflow_run_name}
        with mlflow.start_run(**start_kwargs) as active_run:
            if not run_data:
                atomic_write_json(run_id_path, {'run_id': active_run.info.run_id})
                mlflow.log_params({key: value for key, value in manifest.items() if not isinstance(value, dict)})
            tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, use_fast=True)
            selections = {}
            for stage in ('aspect', 'polarity'):
                LOGGER.info('Starting %s stage', stage)
                selections[stage] = train_stage(
                    torch, transformers, mlflow, tokenizer, config, stage,
                    splits['train'], splits['eval'],
                )
            results = {}
            for split_name in ('eval', 'test'):
                output_dir = config.run_dir / ('validation' if split_name == 'eval' else 'test')
                predictions = _pipeline_predictions(
                    torch, transformers, tokenizer, config, splits[split_name]
                )
                gold_path = output_dir / 'gold.csv'
                prediction_path = output_dir / 'predictions.csv'
                _write_labeled_csv(gold_path, splits[split_name])
                _write_predictions(prediction_path, predictions)
                results[split_name] = evaluate(
                    gold_path, prediction_path, config.evaluator,
                    output_dir / 'metrics.json', output_dir / 'official_evaluation.txt',
                )
            baseline_dir = config.run_dir / 'baseline-test'
            baseline_predictions = _legacy_predictions(
                torch, transformers, tokenizer, config, splits['test']
            )
            baseline_gold = baseline_dir / 'gold.csv'
            baseline_output = baseline_dir / 'predictions.csv'
            _write_labeled_csv(baseline_gold, splits['test'])
            _write_predictions(baseline_output, baseline_predictions)
            baseline_metrics = evaluate(
                baseline_gold, baseline_output, config.evaluator,
                baseline_dir / 'metrics.json', baseline_dir / 'official_evaluation.txt',
            )
            _write_comparison(
                config, selections, results['eval'], results['test'], baseline_metrics,
                time.monotonic() - started,
            )
            atomic_write_json(config.run_dir / 'selection.json', selections)
            test_source = results['test']['source_of_truth_metrics']
            mlflow.log_metric('test/aspect_micro_f1', test_source['aspect']['micro']['f1'])
            mlflow.log_metric('test/polarity_micro_f1', test_source['polarity']['micro']['f1'])
            mlflow.log_metric('test/overall_micro_f1', test_source['overall']['micro']['f1'])
            mlflow.log_artifacts(str(config.run_dir / 'test'), artifact_path='test')
            mlflow.log_artifact(str(config.run_dir / 'comparison.md'))


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
        LOGGER.exception('Experiment failed')
        return 1
    return 0
