from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import logging
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from .absa_training import PolarityDataset, _attach_tokenizer, _atomic_torch_save
from .checkpoint import RunLock
from .data import LabeledRow, read_labeled_csv, stratified_group_split
from .imbalance_losses import inverse_frequency_weights, polarity_counts
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .labels import POLARITIES
from .roberta_training import _precision_settings, _prepare_mlflow_tracking_uri, _seed_everything


LOGGER = logging.getLogger(__name__)
CONFIGURATIONS = ('reference', 'loss-control', 'oversampling', 'sampling-corrected')
SAMPLING_MULTIPLIERS = {'positive': 1.0, 'negative': 1.0, 'neutral': 2.0, 'conflict': 3.0}


@dataclass(frozen=True)
class CvConfig:
    train: Path = Path('artifacts/training/roberta-aspect-exp1/splits/train.csv')
    validation: Path = Path('artifacts/training/roberta-aspect-exp1/splits/eval.csv')
    run_dir: Path = Path('artifacts/experiments/polarity-oversampling-cv-v1')
    optimizer_dir: Path = Path('/tmp/contest2-polarity-oversampling-cv-v1')
    model: str = 'FacebookAI/roberta-base'
    folds: int = 5
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
    mixed_precision: str = 'auto'
    mlflow_tracking_uri: str = 'sqlite:////home/kami/Projects/NLP/Contest2/artifacts/mlflow.db'
    mlflow_experiment: str = 'contest2-polarity-oversampling-cv-v1'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run grouped polarity cross-validation with controlled oversampling.')
    for name, field in CvConfig.__dataclass_fields__.items():
        default = field.default
        option = f'--{name.replace("_", "-")}'
        if isinstance(default, Path):
            parser.add_argument(option, type=Path, default=default)
        elif isinstance(default, str):
            parser.add_argument(option, default=default)
        else:
            parser.add_argument(option, type=type(default), default=default)
    return parser


def config_from_args(argv: list[str] | None = None) -> CvConfig:
    config = CvConfig(**vars(build_parser().parse_args(argv)))
    if config.folds < 2:
        raise ValueError('folds must be at least 2')
    for name in ('epochs', 'evaluation_interval', 'early_stopping_patience',
                 'max_length', 'train_batch_size', 'eval_batch_size',
                 'gradient_accumulation_steps'):
        if getattr(config, name) < 1:
            raise ValueError(f'{name} must be positive')
    return config


def grouped_folds(rows: Sequence[LabeledRow], folds: int, seed: int) -> list[set[str]]:
    grouped: dict[str, list[LabeledRow]] = {}
    for row in rows:
        grouped.setdefault(row.item_id, []).append(row)
    if folds > len(grouped):
        raise ValueError('fold count exceeds review count')
    labels = list(POLARITIES)
    totals = Counter(row.polarity for row in rows)
    targets = {label: totals[label] / folds for label in labels}
    target_groups = len(grouped) / folds
    rng = random.Random(seed)
    tie_breakers = {item_id: rng.random() for item_id in grouped}
    ordered = sorted(grouped, key=lambda item_id: (
        -sum(1 / max(totals[row.polarity], 1) for row in grouped[item_id]),
        -len(grouped[item_id]), tie_breakers[item_id], item_id,
    ))
    assignments = [set() for _ in range(folds)]
    counts = [Counter() for _ in range(folds)]
    for item_id in ordered:
        contribution = Counter(row.polarity for row in grouped[item_id])
        destination = min(range(folds), key=lambda index: (
            sum(((counts[index][label] + contribution[label]) / max(targets[label], 1)) ** 2 for label in labels),
            ((len(assignments[index]) + 1) / target_groups) ** 2,
            len(assignments[index]), index,
        ))
        assignments[destination].add(item_id)
        counts[destination].update(contribution)
    return assignments


def sampling_weights(rows: Sequence[LabeledRow], configuration: str) -> list[float] | None:
    if configuration not in CONFIGURATIONS:
        raise ValueError(f'Unknown configuration: {configuration}')
    if configuration not in {'oversampling', 'sampling-corrected'}:
        return None
    return [SAMPLING_MULTIPLIERS[row.polarity] for row in rows]


def class_weights(torch: Any, rows: Sequence[LabeledRow], configuration: str) -> Any:
    if configuration in {'loss-control', 'oversampling'}:
        return torch.ones(len(POLARITIES), dtype=torch.float32)
    weights = inverse_frequency_weights(torch, polarity_counts(rows))
    if configuration == 'sampling-corrected':
        divisors = torch.tensor([SAMPLING_MULTIPLIERS[label] for label in POLARITIES])
        weights = weights / divisors
    return weights


def _prf(gold: Sequence[int], predicted: Sequence[int]) -> dict[str, Any]:
    classes = {}
    for index, label in enumerate(POLARITIES):
        tp = sum(g == index and p == index for g, p in zip(gold, predicted))
        fp = sum(g != index and p == index for g, p in zip(gold, predicted))
        fn = sum(g == index and p != index for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        classes[label] = {'precision': precision, 'recall': recall,
                          'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                          'support': sum(g == index for g in gold)}
    return {
        'classes': classes,
        'macro_f1': sum(value['f1'] for value in classes.values()) / len(POLARITIES),
        'accuracy': sum(g == p for g, p in zip(gold, predicted)) / len(gold),
    }


def _predict(torch: Any, transformers: Any, model: Any, dataset: Any,
             batch_size: int, label: str) -> Any:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
        collate_fn=transformers.DataCollatorWithPadding(tokenizer=dataset.tokenizer))
    values = []; device = next(model.parameters()).device; model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=label, unit='batch', dynamic_ncols=True):
            batch.pop('labels')
            values.append(model(**{key: value.to(device) for key, value in batch.items()}).logits.cpu())
    return torch.cat(values)


def _optimizer(torch: Any, transformers: Any, model: Any, row_count: int, config: CvConfig) -> tuple[Any, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    batches = math.ceil(row_count / config.train_batch_size)
    updates = math.ceil(batches / config.gradient_accumulation_steps) * config.epochs
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(updates * config.warmup_ratio), num_training_steps=updates)
    return optimizer, scheduler


def _train_fold(torch: Any, transformers: Any, mlflow: Any, tokenizer: Any,
                config: CvConfig, configuration: str, fold: int,
                train_rows: list[LabeledRow], selection_rows: list[LabeledRow],
                heldout_rows: list[LabeledRow]) -> dict[str, Any]:
    run_dir = config.run_dir / 'runs' / configuration / f'fold-{fold}'
    best_path = run_dir / 'best.pt'; result_path = run_dir / 'result.json'
    if result_path.is_file():
        return json.loads(result_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(torch, config.seed + fold)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = _attach_tokenizer(PolarityDataset(train_rows, tokenizer, config.max_length), tokenizer)
    selection_dataset = _attach_tokenizer(PolarityDataset(selection_rows, tokenizer, config.max_length), tokenizer)
    heldout_dataset = _attach_tokenizer(PolarityDataset(heldout_rows, tokenizer, config.max_length), tokenizer)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        config.model, num_labels=len(POLARITIES)).to(device)
    optimizer, scheduler = _optimizer(torch, transformers, model, len(train_rows), config)
    use_fp16, autocast_dtype = _precision_settings(torch, config.mixed_precision, device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    weights = class_weights(torch, train_rows, configuration).to(device)
    sampler_values = sampling_weights(train_rows, configuration)
    latest = config.optimizer_dir / configuration / f'fold-{fold}' / 'latest.pt'
    start_epoch = 0; best_score = -1.0; stale = 0
    if latest.is_file():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']); scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch']; best_score = checkpoint['best_score']; stale = checkpoint['stale']
    with mlflow.start_run(run_name=f'{configuration}-fold-{fold}'):
        mlflow.log_params({'configuration': configuration, 'fold': fold, 'seed': config.seed + fold})
        for epoch in range(start_epoch + 1, config.epochs + 1):
            generator = torch.Generator().manual_seed(config.seed * 1000 + fold * 100 + epoch)
            if sampler_values is None:
                sampler = torch.utils.data.RandomSampler(train_dataset, generator=generator)
            else:
                sampler = torch.utils.data.WeightedRandomSampler(
                    sampler_values, num_samples=len(train_dataset), replacement=True, generator=generator)
            loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size,
                sampler=sampler, collate_fn=transformers.DataCollatorWithPadding(tokenizer=tokenizer))
            model.train(); optimizer.zero_grad(set_to_none=True); running = 0.0
            progress = tqdm(loader, desc=f'{configuration} fold {fold} epoch {epoch}', unit='batch', dynamic_ncols=True)
            for batch_index, batch in enumerate(progress):
                targets = batch.pop('labels').to(device)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
                    per_example = torch.nn.functional.cross_entropy(logits, targets, weight=weights, reduction='none')
                    loss = (per_example.sum() / weights[targets].sum().clamp_min(1e-8)) / config.gradient_accumulation_steps
                scaler.scale(loss).backward(); running += float(loss.detach().cpu()) * config.gradient_accumulation_steps
                if (batch_index + 1) % config.gradient_accumulation_steps == 0 or batch_index + 1 == len(loader):
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
                progress.set_postfix(loss=f'{running / (batch_index + 1):.4f}')
            evaluate_now = epoch % config.evaluation_interval == 0 or epoch == config.epochs
            if evaluate_now:
                logits = _predict(torch, transformers, model, selection_dataset, config.eval_batch_size,
                                  f'select {configuration} fold {fold}')
                metrics = _prf([POLARITIES.index(row.polarity) for row in selection_rows], logits.argmax(-1).tolist())
                score = metrics['macro_f1']; mlflow.log_metric('selection/macro_f1', score, step=epoch)
                if score > best_score + config.early_stopping_min_delta:
                    best_score = score; stale = 0
                    state = {name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
                             for name, value in model.state_dict().items()}
                    _atomic_torch_save(torch, {'model': state, 'epoch': epoch, 'selection': metrics}, best_path)
                else:
                    stale += 1
            _atomic_torch_save(torch, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
                'best_score': best_score, 'stale': stale}, latest)
            if evaluate_now and stale >= config.early_stopping_patience:
                break
    checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model']); model.to(device)
    logits = _predict(torch, transformers, model, heldout_dataset, config.eval_batch_size,
                      f'heldout {configuration} fold {fold}')
    predicted = logits.argmax(-1).tolist(); gold = [POLARITIES.index(row.polarity) for row in heldout_rows]
    result = {'finished': True, 'fold': fold, 'configuration': configuration,
              'best_epoch': checkpoint['epoch'], 'selection': checkpoint['selection'],
              'heldout': _prf(gold, predicted),
              'predictions': [{'id': row.item_id, 'text': row.text, 'aspect': row.aspect,
                               'gold': row.polarity, 'predicted': POLARITIES[value]}
                              for row, value in zip(heldout_rows, predicted)]}
    atomic_write_json(result_path, result); latest.unlink(missing_ok=True)
    del model, optimizer, scheduler, scaler; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def _error_csv(predictions: Sequence[dict[str, str]]) -> str:
    output = io.StringIO(newline=''); writer = csv.writer(output, lineterminator='\n')
    writer.writerow(('id', 'text', 'aspectCategory', 'gold', 'predicted', 'error_type'))
    for value in predictions:
        if value['gold'] in {'neutral', 'conflict'} and value['gold'] != value['predicted']:
            writer.writerow((value['id'], value['text'], value['aspect'], value['gold'],
                             value['predicted'], f"{value['gold']}->{value['predicted']}"))
    return output.getvalue()


def _report(results: dict[str, Any]) -> str:
    rows = []
    for name in CONFIGURATIONS:
        value = results['configurations'][name]['pooled']
        rows.append(f"| {name} | {value['accuracy']:.6f} | {value['macro_f1']:.6f} | "
                    f"{value['classes']['neutral']['f1']:.6f} | {value['classes']['conflict']['f1']:.6f} |")
    return '''# Grouped polarity oversampling cross-validation

Five grouped outer folds estimate gold-aspect polarity performance. Every outer
training set contains its own grouped checkpoint-selection split.

| Configuration | Accuracy | Macro F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|
''' + '\n'.join(rows) + f'''\n
Promotion gate winner: `{results['winner']}`.

The historical test split was not used by this experiment.
'''


def run_experiment(config: CvConfig) -> None:
    import mlflow
    import torch
    import transformers
    started = time.monotonic(); config.run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(config.run_dir / '.experiment.lock'):
        if (config.run_dir / 'results.json').is_file():
            LOGGER.info('Experiment already complete'); return
        rows = read_labeled_csv(config.train) + read_labeled_csv(config.validation)
        folds = grouped_folds(rows, config.folds, config.seed)
        manifest = {**{key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                    'configurations': list(CONFIGURATIONS), 'sampling_multipliers': SAMPLING_MULTIPLIERS,
                    'train_sha256': file_sha256(config.train), 'validation_sha256': file_sha256(config.validation),
                    'fold_ids': [sorted(values) for values in folds]}
        manifest_path = config.run_dir / 'manifest.json'
        if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Existing manifest differs')
        atomic_write_json(manifest_path, manifest)
        _prepare_mlflow_tracking_uri(config.mlflow_tracking_uri); mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, use_fast=True)
        output = {'configurations': {}}
        all_ids = {row.item_id for row in rows}
        for configuration in CONFIGURATIONS:
            fold_results = []
            for fold_index, heldout_ids in enumerate(folds, 1):
                outer_train = [row for row in rows if row.item_id not in heldout_ids]
                heldout = [row for row in rows if row.item_id in heldout_ids]
                inner = stratified_group_split(outer_train, test_ratio=0.1, seed=config.seed + fold_index)
                result = _train_fold(torch, transformers, mlflow, tokenizer, config, configuration,
                                     fold_index, list(inner.train_rows), list(inner.test_rows), heldout)
                fold_results.append(result)
                atomic_write_json(config.run_dir / 'progress.json', {'configuration': configuration, 'fold': fold_index})
            predictions = [value for fold_result in fold_results for value in fold_result['predictions']]
            if {value['id'] for value in predictions} != all_ids:
                raise RuntimeError('Out-of-fold predictions do not cover every review')
            gold = [POLARITIES.index(value['gold']) for value in predictions]
            predicted = [POLARITIES.index(value['predicted']) for value in predictions]
            pooled = _prf(gold, predicted)
            output['configurations'][configuration] = {
                'pooled': pooled, 'folds': [{key: value for key, value in result.items() if key != 'predictions'} for result in fold_results],
                'fold_macro_f1_mean': statistics.mean(result['heldout']['macro_f1'] for result in fold_results),
                'fold_macro_f1_stdev': statistics.pstdev(result['heldout']['macro_f1'] for result in fold_results),
            }
            atomic_write_text(config.run_dir / 'errors' / f'{configuration}.csv', _error_csv(predictions))
            atomic_write_json(config.run_dir / 'oof' / f'{configuration}.json', predictions)
        reference = output['configurations']['reference']
        eligible = []
        for name in CONFIGURATIONS[1:]:
            candidate = output['configurations'][name]
            pooled = candidate['pooled']; baseline = reference['pooled']
            fold_wins = {label: sum(
                candidate['folds'][index]['heldout']['classes'][label]['f1'] >= reference['folds'][index]['heldout']['classes'][label]['f1']
                for index in range(config.folds)) for label in ('neutral', 'conflict')}
            if (pooled['classes']['neutral']['f1'] > baseline['classes']['neutral']['f1']
                    and pooled['classes']['conflict']['f1'] > baseline['classes']['conflict']['f1']
                    and min(fold_wins.values()) >= 4
                    and pooled['accuracy'] >= baseline['accuracy'] - 0.005):
                eligible.append(name)
            candidate['fold_non_regressions'] = fold_wins
        winner = max(eligible, key=lambda name: (
            (output['configurations'][name]['pooled']['classes']['neutral']['f1'] +
             output['configurations'][name]['pooled']['classes']['conflict']['f1']) / 2,
            output['configurations'][name]['pooled']['macro_f1'],
        )) if eligible else 'reference'
        output.update({'winner': winner, 'eligible': eligible, 'runtime_seconds': time.monotonic() - started,
                       'test_accessed': False})
        atomic_write_json(config.run_dir / 'results.json', output)
        atomic_write_text(config.run_dir / 'report.md', _report(output))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    try:
        run_experiment(config_from_args(argv))
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted; rerun to resume'); return 130
    except Exception:
        LOGGER.exception('Polarity CV experiment failed'); return 1
    return 0
