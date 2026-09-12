from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from .absa_training import (
    ExperimentConfig as BaselineConfig,
    PolarityDataset,
    ReviewTarget,
    _attach_tokenizer,
    _atomic_torch_save,
    _model_state_cpu,
    _predict_logits,
    _validate_splits,
    group_aspect_targets,
)
from .checkpoint import RunLock
from .data import LabeledRow, read_labeled_csv
from .evaluation import calculate_metrics, evaluate, read_predictions
from .imbalance_losses import LossSpec, SCREENING_LOSSES, loss_weights, polarity_loss
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .labels import ASPECTS, POLARITIES
from .roberta_training import (
    _precision_settings,
    _prepare_mlflow_tracking_uri,
    _seed_everything,
    _write_labeled_csv,
    _write_predictions,
)


LOGGER = logging.getLogger(__name__)
CONFIRMATION_SEEDS = (17, 42, 73)


@dataclass(frozen=True)
class JointExperimentConfig:
    input: Path = Path('data/contest2_train.csv')
    split_dir: Path = Path('artifacts/training/roberta-aspect-exp1/splits')
    fixed_aspect_checkpoint: Path = Path(
        'artifacts/experiments/multilabel-conditioned-v1/best/aspect.pt'
    )
    existing_baseline_dir: Path = Path(
        'artifacts/experiments/multilabel-conditioned-v1'
    )
    run_dir: Path = Path('artifacts/experiments/absa-imbalance-joint-v1')
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
    mlflow_tracking_uri: str = 'sqlite:///artifacts/mlflow.db'
    mlflow_experiment: str = 'contest2-absa-imbalance-joint-v1'
    bootstrap_samples: int = 10_000


@dataclass(frozen=True)
class CandidateTarget:
    item_id: str
    text: str
    aspect_index: int
    present: int
    polarity_index: int
    review_index: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run class-imbalance screening and candidate-conditioned joint ABSA.'
    )
    for name, field in JointExperimentConfig.__dataclass_fields__.items():
        default = field.default
        option = f'--{name.replace("_", "-")}'
        if isinstance(default, Path):
            parser.add_argument(option, type=Path, default=default)
        elif isinstance(default, str):
            parser.add_argument(option, default=default)
        else:
            parser.add_argument(option, type=type(default), default=default)
    return parser


def config_from_args(argv: list[str] | None = None) -> JointExperimentConfig:
    config = JointExperimentConfig(**vars(build_parser().parse_args(argv)))
    positive = (
        config.epochs, config.evaluation_interval, config.early_stopping_patience,
        config.max_length, config.train_batch_size, config.eval_batch_size,
        config.gradient_accumulation_steps, config.log_steps, config.bootstrap_samples,
    )
    if any(value < 1 for value in positive):
        raise ValueError('Epoch, interval, batch, logging, and bootstrap values must be positive')
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError('Learning rate must be positive and weight decay nonnegative')
    if not 0 <= config.warmup_ratio < 1:
        raise ValueError('Warmup ratio must be in [0, 1)')
    return config


def candidate_targets(rows: Sequence[LabeledRow]) -> list[CandidateTarget]:
    reviews = group_aspect_targets(rows)
    by_id: dict[str, dict[int, int]] = {}
    for row in rows:
        mapping = by_id.setdefault(row.item_id, {})
        aspect = ASPECTS.index(row.aspect)
        polarity = POLARITIES.index(row.polarity)
        if aspect in mapping and mapping[aspect] != polarity:
            raise ValueError(f'ID {row.item_id!r} has conflicting polarity labels for {row.aspect}')
        mapping[aspect] = polarity
    return [
        CandidateTarget(
            review.item_id, review.text, aspect, int(aspect in by_id[review.item_id]),
            by_id[review.item_id].get(aspect, -100), review_index,
        )
        for review_index, review in enumerate(reviews)
        for aspect in range(len(ASPECTS))
    ]


class JointCandidateDataset:
    def __init__(self, targets: Sequence[CandidateTarget], tokenizer: Any, max_length: int):
        self.items = []
        for target in targets:
            encoded = tokenizer(
                target.text, f'aspect: {ASPECTS[target.aspect_index]}',
                truncation=True, max_length=max_length,
            )
            encoded.update({
                'aspect_labels': float(target.present),
                'polarity_labels': target.polarity_index,
                'aspect_index': target.aspect_index,
                'review_index': target.review_index,
            })
            self.items.append(encoded)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def aspect_positive_weights(torch: Any, rows: Sequence[LabeledRow]) -> Any:
    reviews = group_aspect_targets(rows)
    counts = [0] * len(ASPECTS)
    for review in reviews:
        for aspect in review.aspects:
            counts[aspect] += 1
    return torch.tensor(
        [(len(reviews) - count) / count for count in counts], dtype=torch.float32
    )


def masked_joint_loss(
    torch: Any,
    aspect_logits: Any,
    polarity_logits: Any,
    aspect_targets: Any,
    polarity_targets: Any,
    aspect_indices: Any,
    positive_weights: Any,
    polarity_weights: Any,
    spec: LossSpec,
) -> tuple[Any, Any, Any]:
    positive_term = -aspect_targets * positive_weights[aspect_indices] * torch.nn.functional.logsigmoid(aspect_logits)
    negative_term = -(1.0 - aspect_targets) * torch.nn.functional.logsigmoid(-aspect_logits)
    aspect_loss = (positive_term + negative_term).mean()
    present = aspect_targets.bool()
    sentiment_loss = polarity_loss(
        torch, polarity_logits[present], polarity_targets[present], polarity_weights, spec
    )
    return aspect_loss + sentiment_loss, aspect_loss, sentiment_loss


def all_candidate_rows(reviews: Sequence[ReviewTarget]) -> list[LabeledRow]:
    return [
        LabeledRow(
            review.item_id, review.text, aspect, POLARITIES[0],
            review_index * len(ASPECTS) + aspect_index,
        )
        for review_index, review in enumerate(reviews)
        for aspect_index, aspect in enumerate(ASPECTS)
    ]


def decode_pairs(
    reviews: Sequence[ReviewTarget],
    aspect_probabilities: Sequence[Sequence[float]],
    polarity_predictions: Sequence[Sequence[int]],
    thresholds: Sequence[float],
) -> list[tuple[str, str, str]]:
    predictions = []
    for review, probabilities, polarities in zip(
        reviews, aspect_probabilities, polarity_predictions
    ):
        selected = [
            index for index, probability in enumerate(probabilities)
            if probability >= thresholds[index]
        ]
        if not selected:
            selected = [max(range(len(ASPECTS)), key=lambda index: probabilities[index])]
        predictions.extend(
            (review.item_id, ASPECTS[index], POLARITIES[polarities[index]])
            for index in selected
        )
    return predictions


def pair_f1(rows: Sequence[LabeledRow], predictions: Sequence[tuple[str, str, str]]) -> float:
    return float(calculate_metrics(rows, predictions)['source_of_truth_metrics']['overall']['micro']['f1'])


def tune_pair_thresholds(
    rows: Sequence[LabeledRow],
    aspect_probabilities: Sequence[Sequence[float]],
    polarity_predictions: Sequence[Sequence[int]],
    initial: Sequence[float] | None = None,
) -> list[float]:
    reviews = group_aspect_targets(rows)
    grid = [value / 100 for value in range(10, 91, 5)]
    thresholds = list(initial or [0.5] * len(ASPECTS))
    for _ in range(10):
        changed = False
        for aspect in range(len(ASPECTS)):
            candidates = []
            for threshold in grid:
                proposed = thresholds.copy()
                proposed[aspect] = threshold
                predictions = decode_pairs(
                    reviews, aspect_probabilities, polarity_predictions, proposed
                )
                metrics = calculate_metrics(rows, predictions)['source_of_truth_metrics']['overall']['micro']
                candidates.append((
                    float(metrics['f1']), float(metrics['precision']),
                    -abs(threshold - 0.5), threshold,
                ))
            best_threshold = max(candidates)[-1]
            changed |= best_threshold != thresholds[aspect]
            thresholds[aspect] = best_threshold
        if not changed:
            break
    return thresholds


def _reshape(values: Sequence[Any], reviews: int) -> list[list[Any]]:
    width = len(ASPECTS)
    if len(values) != reviews * width:
        raise ValueError('Candidate output count does not match five candidates per review')
    return [list(values[index * width:(index + 1) * width]) for index in range(reviews)]


def _build_joint_model(torch: Any, transformers: Any, model_name: str) -> Any:
    class JointModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = transformers.AutoModel.from_pretrained(model_name)
            hidden = int(self.encoder.config.hidden_size)
            dropout = float(getattr(self.encoder.config, 'hidden_dropout_prob', 0.1))
            self.dropout = torch.nn.Dropout(dropout)
            self.aspect_head = torch.nn.Linear(hidden, 1)
            self.polarity_head = torch.nn.Linear(hidden, len(POLARITIES))

        def forward(self, **inputs: Any) -> tuple[Any, Any]:
            hidden = self.encoder(**inputs).last_hidden_state[:, 0]
            hidden = self.dropout(hidden)
            return self.aspect_head(hidden).squeeze(-1), self.polarity_head(hidden)

    return JointModel()


def _joint_predict(
    torch: Any, transformers: Any, model: Any, dataset: JointCandidateDataset,
    batch_size: int, label: str,
) -> tuple[Any, Any]:
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=transformers.DataCollatorWithPadding(tokenizer=dataset.tokenizer),
    )
    aspect_logits, polarity_logits = [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=label, unit='batch', dynamic_ncols=True):
            for key in ('aspect_labels', 'polarity_labels', 'aspect_index', 'review_index'):
                batch.pop(key)
            aspect, polarity = model(**{key: value.to(device) for key, value in batch.items()})
            aspect_logits.append(aspect.cpu())
            polarity_logits.append(polarity.cpu())
    return torch.cat(aspect_logits), torch.cat(polarity_logits)


def _fixed_aspect_probabilities(
    torch: Any, transformers: Any, tokenizer: Any, config: JointExperimentConfig,
    rows: Sequence[LabeledRow], label: str,
) -> list[list[float]]:
    reviews = group_aspect_targets(rows)
    checkpoint = torch.load(config.fixed_aspect_checkpoint, map_location='cpu', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        config.model, num_labels=len(ASPECTS)
    ).to(device)
    model.load_state_dict(checkpoint['model'])
    from .absa_training import AspectDataset
    dataset = _attach_tokenizer(AspectDataset(reviews, tokenizer, config.max_length), tokenizer)
    logits = _predict_logits(torch, transformers, model, dataset, config.eval_batch_size, label)
    probabilities = torch.sigmoid(logits).tolist()
    del model, logits, checkpoint
    gc.collect()
    return probabilities


def _run_manifest(
    config: JointExperimentConfig, architecture: str, spec: LossSpec, seed: int
) -> dict[str, Any]:
    return {
        'architecture': architecture,
        'loss': spec.serialized(),
        'seed': seed,
        'model': config.model,
        'input': str(config.input),
        'input_sha256': file_sha256(config.input),
        'split_dir': str(config.split_dir),
        'split_sha256': {
            name: file_sha256(config.split_dir / f'{name}.csv')
            for name in ('train', 'eval', 'test')
        },
        'epochs': config.epochs,
        'evaluation_interval': config.evaluation_interval,
        'early_stopping_patience': config.early_stopping_patience,
        'early_stopping_min_delta': config.early_stopping_min_delta,
        'max_length': config.max_length,
        'learning_rate': config.learning_rate,
        'train_batch_size': config.train_batch_size,
        'eval_batch_size': config.eval_batch_size,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'weight_decay': config.weight_decay,
        'warmup_ratio': config.warmup_ratio,
        'mixed_precision': config.mixed_precision,
    }


def _prepare_run(
    run_dir: Path, manifest: dict[str, Any]
) -> dict[str, Any] | None:
    result_path = run_dir / 'result.json'
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result.get('finished'):
            existing = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
            if existing != manifest:
                raise RuntimeError(f'Completed run manifest mismatch: {run_dir}')
            LOGGER.info('Reusing completed run %s', run_dir)
            return result
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / 'manifest.json'
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError(f'Run manifest mismatch: {run_dir}')
    atomic_write_json(manifest_path, manifest)
    return None


def _optimizer_and_schedule(
    torch: Any, transformers: Any, model: Any, dataset_size: int,
    config: JointExperimentConfig,
) -> tuple[Any, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    batches = math.ceil(dataset_size / config.train_batch_size)
    updates = math.ceil(batches / config.gradient_accumulation_steps)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(updates * config.epochs * config.warmup_ratio),
        num_training_steps=updates * config.epochs,
    )
    return optimizer, scheduler


def _start_or_resume_mlflow(
    mlflow: Any, config: JointExperimentConfig, run_dir: Path,
    manifest: dict[str, Any], run_name: str,
) -> Any:
    _prepare_mlflow_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)
    path = run_dir / 'mlflow_run.json'
    data = json.loads(path.read_text()) if path.is_file() else {}
    active = mlflow.start_run(
        **({'run_id': data['run_id']} if data else {'run_name': run_name})
    )
    if not data:
        atomic_write_json(path, {'run_id': active.info.run_id})
        mlflow.log_params({
            'architecture': manifest['architecture'], 'loss': manifest['loss']['name'],
            'gamma': manifest['loss']['gamma'], 'beta': manifest['loss']['beta'],
            'neutral_multiplier': manifest['loss'].get('neutral_multiplier', 1.0),
            'conflict_multiplier': manifest['loss'].get('conflict_multiplier', 1.0),
            'seed': manifest['seed'], 'model': manifest['model'],
        })
    return active


def _load_training_checkpoint(
    torch: Any, path: Path, model: Any, optimizer: Any, scheduler: Any, scaler: Any,
) -> tuple[int, float, int, int]:
    if not path.is_file():
        return 0, -1.0, 0, 0
    checkpoint = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    return (
        int(checkpoint['epoch']), float(checkpoint['best_score']),
        int(checkpoint['stale_checks']), int(checkpoint['global_step']),
    )


def _save_epoch_checkpoint(
    torch: Any, path: Path, model: Any, optimizer: Any, scheduler: Any,
    scaler: Any, epoch: int, best_score: float, stale_checks: int,
    global_step: int,
) -> None:
    _atomic_torch_save(torch, {
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
        'epoch': epoch, 'best_score': best_score, 'stale_checks': stale_checks,
        'global_step': global_step,
    }, path)


def _separate_validation(
    torch: Any, transformers: Any, tokenizer: Any, model: Any,
    config: JointExperimentConfig, validation_rows: list[LabeledRow],
    aspect_probabilities: list[list[float]], label: str,
) -> tuple[float, list[float], float, dict[str, dict[str, float | int]]]:
    reviews = group_aspect_targets(validation_rows)
    dataset = _attach_tokenizer(
        PolarityDataset(all_candidate_rows(reviews), tokenizer, config.max_length), tokenizer
    )
    logits = _predict_logits(
        torch, transformers, model, dataset, config.eval_batch_size, label
    )
    polarities = _reshape(logits.argmax(dim=-1).tolist(), len(reviews))
    thresholds = tune_pair_thresholds(
        validation_rows, aspect_probabilities, polarities
    )
    predictions = decode_pairs(reviews, aspect_probabilities, polarities, thresholds)
    score = pair_f1(validation_rows, predictions)
    gold = [POLARITIES.index(row.polarity) for row in validation_rows]
    gold_dataset = _attach_tokenizer(
        PolarityDataset(validation_rows, tokenizer, config.max_length), tokenizer
    )
    gold_logits = _predict_logits(
        torch, transformers, model, gold_dataset, config.eval_batch_size,
        f'{label} gold aspects',
    )
    from .absa_training import _macro_f1
    gold_predictions = [
        (row.item_id, row.aspect, POLARITIES[prediction])
        for row, prediction in zip(
            validation_rows, gold_logits.argmax(dim=-1).tolist()
        )
    ]
    macro = _macro_f1(gold, gold_logits.argmax(dim=-1).tolist(), len(POLARITIES))
    return score, thresholds, macro, _pair_class_metrics(
        validation_rows, gold_predictions
    )


def train_separate_polarity(
    torch: Any, transformers: Any, mlflow: Any, tokenizer: Any,
    config: JointExperimentConfig, run_dir: Path, spec: LossSpec, seed: int,
    train_rows: list[LabeledRow], validation_rows: list[LabeledRow],
    validation_aspect_probabilities: list[list[float]],
    minimum_pair_f1: float | None = None,
    optimizer_checkpoint: Path | None = None,
    checkpoint_on_evaluation_only: bool = False,
) -> dict[str, Any]:
    manifest = _run_manifest(config, 'separate', spec, seed)
    completed = _prepare_run(run_dir, manifest)
    if completed:
        return completed
    _seed_everything(torch, seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer_dataset = _attach_tokenizer(
        PolarityDataset(train_rows, tokenizer, config.max_length), tokenizer
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        config.model, num_labels=len(POLARITIES)
    ).float().to(device)
    optimizer, scheduler = _optimizer_and_schedule(
        torch, transformers, model, len(tokenizer_dataset), config
    )
    use_fp16, autocast_dtype = _precision_settings(torch, config.mixed_precision, device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    weights = loss_weights(torch, train_rows, spec).to(device)
    latest = optimizer_checkpoint or run_dir / 'checkpoints/latest.pt'
    best = run_dir / 'best.pt'
    epoch_start, best_score, stale, global_step = _load_training_checkpoint(
        torch, latest, model, optimizer, scheduler, scaler
    )
    with _start_or_resume_mlflow(
        mlflow, config, run_dir, manifest, f'separate-{spec.slug}-seed-{seed}'
    ):
        mlflow.log_params({
            f'effective_weight_{label}': float(weights[index].detach().cpu())
            for index, label in enumerate(POLARITIES)
        })
        for epoch in range(epoch_start + 1, config.epochs + 1):
            generator = torch.Generator().manual_seed(seed + epoch)
            loader = torch.utils.data.DataLoader(
                tokenizer_dataset, batch_size=config.train_batch_size,
                sampler=torch.utils.data.RandomSampler(tokenizer_dataset, generator=generator),
                collate_fn=transformers.DataCollatorWithPadding(tokenizer=tokenizer),
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            progress = tqdm(loader, desc=f'Separate {spec.slug} {epoch}/{config.epochs}', unit='batch', dynamic_ncols=True)
            for batch_index, batch in enumerate(progress):
                targets = batch.pop('labels').to(device)
                inputs = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    logits = model(**inputs).logits
                    raw_loss = polarity_loss(torch, logits, targets, weights, spec)
                    loss = raw_loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
                running += float(raw_loss.detach().cpu())
                progress.set_postfix(loss=f'{running / (batch_index + 1):.4f}')
                update = (
                    (batch_index + 1) % config.gradient_accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if update:
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True); scheduler.step()
                    global_step += 1
                    if global_step % config.log_steps == 0:
                        mlflow.log_metric('train/loss', float(raw_loss.detach().cpu()), step=global_step)
            mlflow.log_metric('train/epoch_loss', running / len(loader), step=epoch)
            evaluate_now = epoch % config.evaluation_interval == 0 or epoch == config.epochs
            if evaluate_now:
                score, thresholds, macro, class_metrics = _separate_validation(
                    torch, transformers, tokenizer, model, config,
                    validation_rows, validation_aspect_probabilities,
                    f'Validate separate {spec.slug} {epoch}',
                )
                mlflow.log_metric('validation/pair_micro_f1', score, step=epoch)
                mlflow.log_metric('validation/polarity_macro_f1', macro, step=epoch)
                minority_score = (
                    float(class_metrics['neutral']['f1'])
                    + float(class_metrics['conflict']['f1'])
                ) / 2
                eligible = minimum_pair_f1 is None or score >= minimum_pair_f1
                selection_score = (
                    score if minimum_pair_f1 is None
                    else (2.0 + minority_score + score * 1e-6 if eligible else score)
                )
                mlflow.log_metric('validation/minority_f1', minority_score, step=epoch)
                mlflow.log_metric('validation/eligible', float(eligible), step=epoch)
                if selection_score > best_score + config.early_stopping_min_delta:
                    best_score, stale = selection_score, 0
                    compact = (
                        spec.neutral_multiplier != 1
                        or spec.conflict_multiplier != 1
                    )
                    model_state = {
                        name: value.detach().cpu().half()
                        if compact and value.is_floating_point() else value.detach().cpu()
                        for name, value in model.state_dict().items()
                    }
                    _atomic_torch_save(torch, {
                        'model': model_state, 'epoch': epoch,
                        'score': score, 'thresholds': thresholds,
                        'selection_score': selection_score,
                        'eligible': eligible,
                        'minimum_pair_f1': minimum_pair_f1,
                        'minority_f1': minority_score,
                        'polarity_classes': class_metrics,
                        'architecture': 'separate', 'loss': spec.serialized(), 'seed': seed,
                    }, best)
                    LOGGER.info(
                        'New best separate %s seed %d: pair=%.6f minority=%.6f eligible=%s',
                        spec.slug, seed, score, minority_score, eligible,
                    )
                else:
                    stale += 1
            if not checkpoint_on_evaluation_only or evaluate_now:
                _save_epoch_checkpoint(
                    torch, latest, model, optimizer, scheduler, scaler, epoch,
                    best_score, stale, global_step,
                )
            if evaluate_now and stale >= config.early_stopping_patience:
                break
        checkpoint = torch.load(best, map_location='cpu', weights_only=False)
        result = {
            'finished': True, 'architecture': 'separate', 'loss': spec.serialized(),
            'seed': seed, 'epoch': int(checkpoint['epoch']),
            'validation_pair_micro_f1': float(checkpoint['score']),
            'validation_minority_f1': float(checkpoint.get('minority_f1', 0.0)),
            'validation_polarity_classes': checkpoint.get('polarity_classes', {}),
            'eligible': bool(checkpoint.get('eligible', True)),
            'minimum_pair_f1': checkpoint.get('minimum_pair_f1'),
            'thresholds': checkpoint['thresholds'],
        }
        atomic_write_json(run_dir / 'result.json', result)
        mlflow.log_metric('best/validation_pair_micro_f1', result['validation_pair_micro_f1'])
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _joint_validation(
    torch: Any, transformers: Any, tokenizer: Any, model: Any,
    config: JointExperimentConfig, validation_rows: list[LabeledRow], label: str,
) -> tuple[float, list[float], float]:
    reviews = group_aspect_targets(validation_rows)
    dataset = JointCandidateDataset(
        candidate_targets(validation_rows), tokenizer, config.max_length
    )
    dataset.tokenizer = tokenizer
    aspect_logits, polarity_logits = _joint_predict(
        torch, transformers, model, dataset, config.eval_batch_size, label
    )
    probabilities = _reshape(torch.sigmoid(aspect_logits).tolist(), len(reviews))
    polarities = _reshape(polarity_logits.argmax(dim=-1).tolist(), len(reviews))
    thresholds = tune_pair_thresholds(validation_rows, probabilities, polarities)
    predictions = decode_pairs(reviews, probabilities, polarities, thresholds)
    score = pair_f1(validation_rows, predictions)
    present_targets = candidate_targets(validation_rows)
    gold_indices = [target.polarity_index for target in present_targets if target.present]
    predicted_indices = [
        int(polarity_logits[index].argmax())
        for index, target in enumerate(present_targets) if target.present
    ]
    from .absa_training import _macro_f1
    macro = _macro_f1(gold_indices, predicted_indices, len(POLARITIES))
    return score, thresholds, macro


def train_joint(
    torch: Any, transformers: Any, mlflow: Any, tokenizer: Any,
    config: JointExperimentConfig, run_dir: Path, spec: LossSpec, seed: int,
    train_rows: list[LabeledRow], validation_rows: list[LabeledRow],
) -> dict[str, Any]:
    manifest = _run_manifest(config, 'joint', spec, seed)
    completed = _prepare_run(run_dir, manifest)
    if completed:
        return completed
    _seed_everything(torch, seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    targets = candidate_targets(train_rows)
    dataset = JointCandidateDataset(targets, tokenizer, config.max_length)
    dataset.tokenizer = tokenizer
    model = _build_joint_model(torch, transformers, config.model).to(device)
    optimizer, scheduler = _optimizer_and_schedule(
        torch, transformers, model, len(dataset), config
    )
    use_fp16, autocast_dtype = _precision_settings(torch, config.mixed_precision, device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    aspect_weights = aspect_positive_weights(torch, train_rows).to(device)
    sentiment_weights = loss_weights(torch, train_rows, spec).to(device)
    latest = run_dir / 'checkpoints/latest.pt'
    best = run_dir / 'best.pt'
    epoch_start, best_score, stale, global_step = _load_training_checkpoint(
        torch, latest, model, optimizer, scheduler, scaler
    )
    with _start_or_resume_mlflow(
        mlflow, config, run_dir, manifest, f'joint-{spec.slug}-seed-{seed}'
    ):
        for epoch in range(epoch_start + 1, config.epochs + 1):
            generator = torch.Generator().manual_seed(seed + 20_000 + epoch)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=config.train_batch_size,
                sampler=torch.utils.data.RandomSampler(dataset, generator=generator),
                collate_fn=transformers.DataCollatorWithPadding(tokenizer=tokenizer),
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running = running_aspect = running_polarity = 0.0
            progress = tqdm(loader, desc=f'Joint {spec.slug} {epoch}/{config.epochs}', unit='batch', dynamic_ncols=True)
            for batch_index, batch in enumerate(progress):
                aspect_targets = batch.pop('aspect_labels').to(device)
                polarity_targets = batch.pop('polarity_labels').to(device)
                aspect_indices = batch.pop('aspect_index').to(device)
                batch.pop('review_index')
                inputs = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    aspect_logits, polarity_logits = model(**inputs)
                    raw_loss, aspect_loss, sentiment_loss = masked_joint_loss(
                        torch, aspect_logits, polarity_logits, aspect_targets,
                        polarity_targets, aspect_indices, aspect_weights,
                        sentiment_weights, spec,
                    )
                    loss = raw_loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
                running += float(raw_loss.detach().cpu())
                running_aspect += float(aspect_loss.detach().cpu())
                running_polarity += float(sentiment_loss.detach().cpu())
                progress.set_postfix(loss=f'{running / (batch_index + 1):.4f}')
                update = (
                    (batch_index + 1) % config.gradient_accumulation_steps == 0
                    or batch_index + 1 == len(loader)
                )
                if update:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    if global_step % config.log_steps == 0:
                        mlflow.log_metrics({
                            'train/loss': float(raw_loss.detach().cpu()),
                            'train/aspect_loss': float(aspect_loss.detach().cpu()),
                            'train/polarity_loss': float(sentiment_loss.detach().cpu()),
                        }, step=global_step)
            mlflow.log_metrics({
                'train/epoch_loss': running / len(loader),
                'train/epoch_aspect_loss': running_aspect / len(loader),
                'train/epoch_polarity_loss': running_polarity / len(loader),
            }, step=epoch)
            evaluate_now = epoch % config.evaluation_interval == 0 or epoch == config.epochs
            if evaluate_now:
                score, thresholds, macro = _joint_validation(
                    torch, transformers, tokenizer, model, config,
                    validation_rows, f'Validate joint {spec.slug} {epoch}',
                )
                mlflow.log_metric('validation/pair_micro_f1', score, step=epoch)
                mlflow.log_metric('validation/polarity_macro_f1', macro, step=epoch)
                if score > best_score + config.early_stopping_min_delta:
                    best_score, stale = score, 0
                    _atomic_torch_save(torch, {
                        'model': _model_state_cpu(model), 'epoch': epoch,
                        'score': score, 'thresholds': thresholds,
                        'architecture': 'joint', 'loss': spec.serialized(), 'seed': seed,
                    }, best)
                    LOGGER.info('New best joint %s seed %d: %.6f', spec.slug, seed, score)
                else:
                    stale += 1
            _save_epoch_checkpoint(
                torch, latest, model, optimizer, scheduler, scaler, epoch,
                best_score, stale, global_step,
            )
            if evaluate_now and stale >= config.early_stopping_patience:
                break
        checkpoint = torch.load(best, map_location='cpu', weights_only=False)
        result = {
            'finished': True, 'architecture': 'joint', 'loss': spec.serialized(),
            'seed': seed, 'epoch': int(checkpoint['epoch']),
            'validation_pair_micro_f1': float(checkpoint['score']),
            'thresholds': checkpoint['thresholds'],
        }
        atomic_write_json(run_dir / 'result.json', result)
        mlflow.log_metric('best/validation_pair_micro_f1', result['validation_pair_micro_f1'])
    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _separate_logits(
    torch: Any, transformers: Any, tokenizer: Any, config: JointExperimentConfig,
    checkpoint_path: Path, rows: list[LabeledRow], label: str,
) -> Any:
    reviews = group_aspect_targets(rows)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        config.model, num_labels=len(POLARITIES)
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    dataset = _attach_tokenizer(
        PolarityDataset(all_candidate_rows(reviews), tokenizer, config.max_length), tokenizer
    )
    logits = _predict_logits(torch, transformers, model, dataset, config.eval_batch_size, label)
    del model, checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits


def _joint_logits(
    torch: Any, transformers: Any, tokenizer: Any, config: JointExperimentConfig,
    checkpoint_path: Path, rows: list[LabeledRow], label: str,
) -> tuple[Any, Any]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _build_joint_model(torch, transformers, config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    dataset = JointCandidateDataset(candidate_targets(rows), tokenizer, config.max_length)
    dataset.tokenizer = tokenizer
    logits = _joint_predict(
        torch, transformers, model, dataset, config.eval_batch_size, label
    )
    del model, checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits


def _mean_logits(torch: Any, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError('At least one set of logits is required')
    return torch.stack(list(values)).mean(dim=0)


def _ensemble_predictions(
    torch: Any, transformers: Any, tokenizer: Any, config: JointExperimentConfig,
    architecture: str, checkpoints: Sequence[Path], rows: list[LabeledRow],
    fixed_aspect_probabilities: list[list[float]] | None,
    thresholds: Sequence[float] | None,
) -> tuple[list[tuple[str, str, str]], list[float]]:
    reviews = group_aspect_targets(rows)
    if architecture == 'separate':
        polarity_values = [
            _separate_logits(
                torch, transformers, tokenizer, config, checkpoint, rows,
                f'Ensemble separate seed {index + 1}/{len(checkpoints)}',
            )
            for index, checkpoint in enumerate(checkpoints)
        ]
        polarity_logits = _mean_logits(torch, polarity_values)
        aspect_probabilities = fixed_aspect_probabilities
        if aspect_probabilities is None:
            raise ValueError('Separate ensemble requires fixed aspect probabilities')
    else:
        joint_values = [
            _joint_logits(
                torch, transformers, tokenizer, config, checkpoint, rows,
                f'Ensemble joint seed {index + 1}/{len(checkpoints)}',
            )
            for index, checkpoint in enumerate(checkpoints)
        ]
        aspect_logits = _mean_logits(torch, [value[0] for value in joint_values])
        polarity_logits = _mean_logits(torch, [value[1] for value in joint_values])
        aspect_probabilities = _reshape(
            torch.sigmoid(aspect_logits).tolist(), len(reviews)
        )
    polarities = _reshape(polarity_logits.argmax(dim=-1).tolist(), len(reviews))
    selected_thresholds = list(thresholds) if thresholds is not None else tune_pair_thresholds(
        rows, aspect_probabilities, polarities
    )
    return (
        decode_pairs(reviews, aspect_probabilities, polarities, selected_thresholds),
        selected_thresholds,
    )


def _write_evaluation(
    config: JointExperimentConfig, output_dir: Path, rows: list[LabeledRow],
    predictions: list[tuple[str, str, str]],
) -> dict[str, Any]:
    gold_path = output_dir / 'gold.csv'
    prediction_path = output_dir / 'predictions.csv'
    _write_labeled_csv(gold_path, rows)
    _write_predictions(prediction_path, predictions)
    return evaluate(
        gold_path, prediction_path, config.evaluator,
        output_dir / 'metrics.json', output_dir / 'official_evaluation.txt',
    )


def _pair_class_metrics(
    rows: Sequence[LabeledRow], predictions: Sequence[tuple[str, str, str]]
) -> dict[str, dict[str, float | int]]:
    gold = {(row.item_id, row.aspect, row.polarity) for row in rows}
    predicted = set(predictions)
    output = {}
    for label in POLARITIES:
        label_gold = {value for value in gold if value[2] == label}
        label_predicted = {value for value in predicted if value[2] == label}
        correct = len(label_gold & label_predicted)
        precision = correct / len(label_predicted) if label_predicted else 0.0
        recall = correct / len(label_gold) if label_gold else 0.0
        output[label] = {
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            'support': len(label_gold),
        }
    return output


def _group_accuracy(
    rows: Sequence[LabeledRow], predictions: Sequence[tuple[str, str, str]]
) -> dict[str, dict[str, float | int]]:
    gold: dict[str, set[tuple[str, str]]] = {}
    predicted: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        gold.setdefault(row.item_id, set()).add((row.aspect, row.polarity))
    for item_id, aspect, polarity in predictions:
        predicted.setdefault(item_id, set()).add((aspect, polarity))
    output = {}
    for name, ids in (
        ('single', [item_id for item_id, values in gold.items() if len(values) == 1]),
        ('multi', [item_id for item_id, values in gold.items() if len(values) > 1]),
    ):
        output[name] = {
            'reviews': len(ids),
            'exact_set_accuracy': (
                sum(gold[item_id] == predicted.get(item_id, set()) for item_id in ids) / len(ids)
                if ids else 0.0
            ),
        }
    return output


def _review_counts(
    rows: Sequence[LabeledRow], predictions: Sequence[tuple[str, str, str]]
) -> dict[str, tuple[int, int, int]]:
    gold: dict[str, set[tuple[str, str]]] = {}
    predicted: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        gold.setdefault(row.item_id, set()).add((row.aspect, row.polarity))
    for item_id, aspect, polarity in predictions:
        predicted.setdefault(item_id, set()).add((aspect, polarity))
    return {
        item_id: (
            len(gold_set & predicted.get(item_id, set())),
            len(predicted.get(item_id, set()) - gold_set),
            len(gold_set - predicted.get(item_id, set())),
        )
        for item_id, gold_set in gold.items()
    }


def _bootstrap_delta(
    rows: Sequence[LabeledRow], left: Sequence[tuple[str, str, str]],
    right: Sequence[tuple[str, str, str]], samples: int, seed: int = 42,
) -> dict[str, float]:
    left_counts = _review_counts(rows, left)
    right_counts = _review_counts(rows, right)
    item_ids = sorted(left_counts)
    generator = random.Random(seed)

    def f1(counts: list[tuple[int, int, int]]) -> float:
        tp = sum(value[0] for value in counts)
        fp = sum(value[1] for value in counts)
        fn = sum(value[2] for value in counts)
        return 2 * tp / (2 * tp + fp + fn) if tp or fp or fn else 0.0

    deltas = []
    for _ in range(samples):
        selected = [generator.choice(item_ids) for _ in item_ids]
        deltas.append(
            f1([left_counts[item_id] for item_id in selected])
            - f1([right_counts[item_id] for item_id in selected])
        )
    deltas.sort()
    return {
        'mean': sum(deltas) / len(deltas),
        'lower_95': deltas[math.floor(0.025 * (len(deltas) - 1))],
        'upper_95': deltas[math.floor(0.975 * (len(deltas) - 1))],
    }


def _write_report(
    config: JointExperimentConfig, screening: dict[str, Any],
    confirmation: dict[str, Any], final: dict[str, Any], runtime: float,
) -> None:
    loss_rows = '\n'.join(
        f"| {name} | {value['validation_pair_micro_f1']:.6f} | {value['epoch']} |"
        for name, value in screening['loss'].items()
    )
    joint_rows = '\n'.join(
        f"| {name} | {value['validation_pair_micro_f1']:.6f} | {value['epoch']} |"
        for name, value in screening['joint'].items()
    )
    confirm_rows = '\n'.join(
        f"| {architecture} | {value['loss']} | {value['mean']:.6f} | {value['std']:.6f} | {value['ensemble_validation_f1']:.6f} |"
        for architecture, value in confirmation.items()
    )
    final_rows = '\n'.join(
        f"| {name} | {value['overall_pair_micro_f1']:.6f} | {value['exact_set_accuracy']:.6f} | {value['single_exact_accuracy']:.6f} | {value['multi_exact_accuracy']:.6f} |"
        for name, value in final['systems'].items()
    )
    polarity_rows = '\n'.join(
        f"| {system} | {polarity} | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | {metrics['support']} |"
        for system, value in final['systems'].items()
        for polarity, metrics in value['pair_polarity_classes'].items()
    )
    text = f'''# Class imbalance and joint ABSA experiment

## Loss screening (seed 42)

| Separate polarity loss | Validation pair micro-F1 | Best epoch |
|---|---:|---:|
{loss_rows}

Selected non-baseline alternative: `{screening['selected_alternative']}`.

## Joint architecture screening (seed 42)

| Joint polarity loss | Validation pair micro-F1 | Best epoch |
|---|---:|---:|
{joint_rows}

## Three-seed confirmation

| Architecture | Loss | Validation mean | Validation std. dev. | Ensemble validation F1 |
|---|---|---:|---:|---:|
{confirm_rows}

Seeds: `{list(CONFIRMATION_SEEDS)}`. The final configuration was locked as
`{final['locked_winner']}` before test inference.

## Untouched test results

| System | Pair micro-F1 | Exact-set accuracy | Single-aspect exact | Multi-aspect exact |
|---|---:|---:|---:|---:|
{final_rows}

### Pair-level polarity breakdown

| System | Polarity | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
{polarity_rows}

Joint minus separate paired-bootstrap delta: {final['bootstrap']['joint_vs_separate']['mean']:+.6f}
(95% CI {final['bootstrap']['joint_vs_separate']['lower_95']:+.6f} to
{final['bootstrap']['joint_vs_separate']['upper_95']:+.6f}).

Locked winner minus existing pipeline delta: {final['bootstrap']['winner_vs_existing']['mean']:+.6f}
(95% CI {final['bootstrap']['winner_vs_existing']['lower_95']:+.6f} to
{final['bootstrap']['winner_vs_existing']['upper_95']:+.6f}).

Runtime: {runtime / 60:.1f} minutes. All configuration and threshold choices
used validation data only; test results were generated after `locked_selection.json`.
'''
    atomic_write_text(config.run_dir / 'report.md', text)


def _run_path(
    config: JointExperimentConfig, architecture: str, spec: LossSpec, seed: int,
    screening: bool,
) -> Path:
    section = 'screening' if screening else 'confirmation'
    return config.run_dir / section / architecture / spec.slug / f'seed-{seed}'


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def run_experiment(config: JointExperimentConfig) -> None:
    import mlflow
    import torch
    import transformers

    started = time.monotonic()
    config.run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(config.run_dir / '.experiment.lock'):
        root_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        }
        config_path = config.run_dir / 'config.json'
        if config_path.is_file() and json.loads(config_path.read_text()) != root_config:
            raise RuntimeError('Experiment configuration differs from the existing run')
        atomic_write_json(config_path, root_config)

        train_rows = read_labeled_csv(config.split_dir / 'train.csv')
        validation_rows = read_labeled_csv(config.split_dir / 'eval.csv')
        if {row.item_id for row in train_rows} & {row.item_id for row in validation_rows}:
            raise RuntimeError('Train and validation IDs overlap')
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, use_fast=True)
        validation_aspects = _fixed_aspect_probabilities(
            torch, transformers, tokenizer, config, validation_rows,
            'Cache fixed validation aspects',
        )

        loss_screen: dict[str, Any] = {}
        loss_specs = {spec.slug: spec for spec in SCREENING_LOSSES}
        for spec in SCREENING_LOSSES:
            path = _run_path(config, 'loss', spec, 42, screening=True)
            loss_screen[spec.slug] = train_separate_polarity(
                torch, transformers, mlflow, tokenizer, config, path, spec, 42,
                train_rows, validation_rows, validation_aspects,
            )
        alternatives = [spec for spec in SCREENING_LOSSES if spec.name != 'weighted-ce']
        selected_alternative = max(
            alternatives,
            key=lambda spec: loss_screen[spec.slug]['validation_pair_micro_f1'],
        )
        separate_spec = max(
            SCREENING_LOSSES,
            key=lambda spec: loss_screen[spec.slug]['validation_pair_micro_f1'],
        )

        joint_screen: dict[str, Any] = {}
        joint_specs = [LossSpec('weighted-ce'), selected_alternative]
        for spec in joint_specs:
            path = _run_path(config, 'joint', spec, 42, screening=True)
            joint_screen[spec.slug] = train_joint(
                torch, transformers, mlflow, tokenizer, config, path, spec, 42,
                train_rows, validation_rows,
            )
        joint_spec = max(
            joint_specs,
            key=lambda spec: joint_screen[spec.slug]['validation_pair_micro_f1'],
        )

        finalists = {'separate': separate_spec, 'joint': joint_spec}
        run_results: dict[str, list[dict[str, Any]]] = {}
        checkpoint_paths: dict[str, list[Path]] = {}
        for architecture, spec in finalists.items():
            results = []
            paths = []
            for seed in CONFIRMATION_SEEDS:
                screening_path = _run_path(
                    config, 'loss' if architecture == 'separate' else 'joint',
                    spec, seed, screening=True,
                )
                path = screening_path if seed == 42 else _run_path(
                    config, architecture, spec, seed, screening=False
                )
                if seed == 42:
                    result = (
                        loss_screen[spec.slug] if architecture == 'separate'
                        else joint_screen[spec.slug]
                    )
                elif architecture == 'separate':
                    result = train_separate_polarity(
                        torch, transformers, mlflow, tokenizer, config, path,
                        spec, seed, train_rows, validation_rows, validation_aspects,
                    )
                else:
                    result = train_joint(
                        torch, transformers, mlflow, tokenizer, config, path,
                        spec, seed, train_rows, validation_rows,
                    )
                results.append(result)
                paths.append(path / 'best.pt')
            run_results[architecture] = results
            checkpoint_paths[architecture] = paths

        confirmation: dict[str, Any] = {}
        validation_thresholds: dict[str, list[float]] = {}
        for architecture, spec in finalists.items():
            predictions, thresholds = _ensemble_predictions(
                torch, transformers, tokenizer, config, architecture,
                checkpoint_paths[architecture], validation_rows,
                validation_aspects if architecture == 'separate' else None, None,
            )
            output_dir = config.run_dir / 'ensembles' / architecture / 'validation'
            metrics = _write_evaluation(config, output_dir, validation_rows, predictions)
            scores = [result['validation_pair_micro_f1'] for result in run_results[architecture]]
            mean, standard_deviation = _mean_std(scores)
            confirmation[architecture] = {
                'loss': spec.slug, 'seeds': list(CONFIRMATION_SEEDS),
                'scores': scores, 'mean': mean, 'std': standard_deviation,
                'ensemble_validation_f1': metrics['source_of_truth_metrics']['overall']['micro']['f1'],
                'thresholds': thresholds,
            }
            validation_thresholds[architecture] = thresholds

        locked_winner = max(
            confirmation,
            key=lambda architecture: confirmation[architecture]['ensemble_validation_f1'],
        )
        locked = {
            'winner': locked_winner,
            'finalists': {
                architecture: {
                    'loss': finalists[architecture].slug,
                    'seeds': list(CONFIRMATION_SEEDS),
                    'thresholds': validation_thresholds[architecture],
                    'ensemble_validation_f1': confirmation[architecture]['ensemble_validation_f1'],
                    'checkpoints': [str(path) for path in checkpoint_paths[architecture]],
                }
                for architecture in finalists
            },
            'test_accessed': False,
        }
        atomic_write_json(config.run_dir / 'locked_selection.json', locked)

        # Gold test labels are loaded only after every finalist and threshold is locked.
        splits = _validate_splits(BaselineConfig(input=config.input, split_dir=config.split_dir))
        test_rows = splits['test']
        test_aspects = _fixed_aspect_probabilities(
            torch, transformers, tokenizer, config, test_rows,
            'Cache fixed test aspects',
        )
        final_predictions: dict[str, list[tuple[str, str, str]]] = {}
        systems: dict[str, Any] = {}
        for architecture in finalists:
            predictions, _ = _ensemble_predictions(
                torch, transformers, tokenizer, config, architecture,
                checkpoint_paths[architecture], test_rows,
                test_aspects if architecture == 'separate' else None,
                validation_thresholds[architecture],
            )
            final_predictions[architecture] = predictions
            metrics = _write_evaluation(
                config, config.run_dir / 'ensembles' / architecture / 'test',
                test_rows, predictions,
            )
            groups = _group_accuracy(test_rows, predictions)
            systems[architecture] = {
                'overall_pair_micro_f1': metrics['source_of_truth_metrics']['overall']['micro']['f1'],
                'exact_set_accuracy': metrics['supplemental_exact_match_accuracy']['overall'],
                'single_exact_accuracy': groups['single']['exact_set_accuracy'],
                'multi_exact_accuracy': groups['multi']['exact_set_accuracy'],
                'pair_polarity_classes': _pair_class_metrics(test_rows, predictions),
            }

        existing_predictions = read_predictions(
            config.existing_baseline_dir / 'test/predictions.csv'
        )
        legacy_predictions = read_predictions(
            config.existing_baseline_dir / 'baseline-test/predictions.csv'
        )
        for name, predictions, metrics_path in (
            ('existing_pipeline', existing_predictions, config.existing_baseline_dir / 'test/metrics.json'),
            ('legacy_baseline', legacy_predictions, config.existing_baseline_dir / 'baseline-test/metrics.json'),
        ):
            metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
            groups = _group_accuracy(test_rows, predictions)
            systems[name] = {
                'overall_pair_micro_f1': metrics['source_of_truth_metrics']['overall']['micro']['f1'],
                'exact_set_accuracy': metrics['supplemental_exact_match_accuracy']['overall'],
                'single_exact_accuracy': groups['single']['exact_set_accuracy'],
                'multi_exact_accuracy': groups['multi']['exact_set_accuracy'],
                'pair_polarity_classes': _pair_class_metrics(test_rows, predictions),
            }

        winner_predictions = final_predictions[locked_winner]
        bootstrap = {
            'joint_vs_separate': _bootstrap_delta(
                test_rows, final_predictions['joint'], final_predictions['separate'],
                config.bootstrap_samples,
            ),
            'winner_vs_existing': _bootstrap_delta(
                test_rows, winner_predictions, existing_predictions,
                config.bootstrap_samples,
            ),
        }
        final = {
            'locked_winner': locked_winner, 'systems': systems,
            'bootstrap': bootstrap,
        }
        screening = {
            'loss': loss_screen, 'joint': joint_screen,
            'selected_alternative': selected_alternative.slug,
        }
        results = {
            'screening': screening, 'confirmation': confirmation,
            'final': final, 'runtime_seconds': time.monotonic() - started,
        }
        atomic_write_json(config.run_dir / 'results.json', results)
        locked['test_accessed'] = True
        atomic_write_json(config.run_dir / 'locked_selection.json', locked)
        _write_report(
            config, screening, confirmation, final, results['runtime_seconds']
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    try:
        run_experiment(config_from_args(argv))
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted; rerun the same command to resume from the last epoch')
        return 130
    except Exception:
        LOGGER.exception('Joint ABSA experiment failed')
        return 1
    return 0


def build_variant_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = 'Train one separate-polarity or joint ABSA variant.'
    parser.add_argument('--architecture', choices=('separate', 'joint'), required=True)
    parser.add_argument(
        '--polarity-loss',
        choices=('weighted-ce', 'weighted-focal', 'class-balanced-focal'),
        required=True,
    )
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--class-balance-beta', type=float, default=0.999)
    parser.add_argument('--neutral-multiplier', type=float, default=1.0)
    parser.add_argument('--conflict-multiplier', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def variant_main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    try:
        values = vars(build_variant_parser().parse_args(argv))
        architecture = values.pop('architecture')
        loss_name = values.pop('polarity_loss')
        gamma = values.pop('focal_gamma')
        beta = values.pop('class_balance_beta')
        neutral_multiplier = values.pop('neutral_multiplier')
        conflict_multiplier = values.pop('conflict_multiplier')
        seed = values.pop('seed')
        config = JointExperimentConfig(**values)
        spec = LossSpec(
            loss_name, gamma=gamma, beta=beta,
            neutral_multiplier=neutral_multiplier,
            conflict_multiplier=conflict_multiplier,
        )
        spec.validate()
        import mlflow
        import torch
        import transformers

        train_rows = read_labeled_csv(config.split_dir / 'train.csv')
        validation_rows = read_labeled_csv(config.split_dir / 'eval.csv')
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, use_fast=True)
        run_dir = config.run_dir
        if architecture == 'separate':
            aspect_probabilities = _fixed_aspect_probabilities(
                torch, transformers, tokenizer, config, validation_rows,
                'Cache fixed validation aspects',
            )
            train_separate_polarity(
                torch, transformers, mlflow, tokenizer, config, run_dir, spec,
                seed, train_rows, validation_rows, aspect_probabilities,
            )
        else:
            train_joint(
                torch, transformers, mlflow, tokenizer, config, run_dir, spec,
                seed, train_rows, validation_rows,
            )
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted; rerun the same command to resume from the last epoch')
        return 130
    except Exception:
        LOGGER.exception('ABSA variant training failed')
        return 1
    return 0
