from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from .absa_composition_experiment import _write_evaluation
from .absa_training import PolarityDataset, _attach_tokenizer, _atomic_torch_save, group_aspect_targets
from .checkpoint import RunLock
from .data import LabeledRow, read_labeled_csv
from .evaluation import calculate_metrics
from .imbalance_losses import LossSpec, loss_weights, polarity_loss
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .joint_absa_experiment import (CONFIRMATION_SEEDS, JointExperimentConfig, _mean_logits,
    _optimizer_and_schedule, _pair_class_metrics, _reshape, _separate_logits,
    all_candidate_rows, decode_pairs)
from .labels import ASPECTS, POLARITIES
from .multilabel_aspect_experiment import AspectExperimentConfig, _ensemble_probabilities
from .roberta_training import _precision_settings, _seed_everything

LOGGER = logging.getLogger(__name__)
LAMBDA_GRID = (0.25, 0.5, 1.0)
TEMPERATURE_GRID = (0.75, 1.0, 1.25, 1.5, 2.0)
ALPHA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
BIAS_GRID = tuple(value / 4 for value in range(-4, 7))
FROZEN_ASPECT_THRESHOLDS = (0.55, 0.8, 0.9, 0.7, 0.45)
EVIDENCE_TARGETS = {'positive': (1.0, 0.0), 'negative': (0.0, 1.0),
                    'neutral': (0.0, 0.0), 'conflict': (1.0, 1.0)}


@dataclass(frozen=True)
class EvidenceConfig:
    input: Path = Path('data/contest2_train.csv')
    split_dir: Path = Path('artifacts/training/roberta-aspect-exp1/splits')
    aspect_run_dir: Path = Path('artifacts/experiments/multilabel-aspect-v1')
    baseline_run_dir: Path = Path('artifacts/experiments/multilabel-aspect-old-polarity-v1')
    run_dir: Path = Path('artifacts/experiments/polarity-evidence-v1')
    optimizer_dir: Path = Path('/tmp/contest2-polarity-evidence-v1')
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
    bootstrap_samples: int = 10_000
    bootstrap_pass_rate: float = 0.75
    mlflow_tracking_uri: str = 'sqlite:////home/kami/Projects/NLP/Contest2/artifacts/mlflow.db'
    mlflow_experiment: str = 'contest2-polarity-evidence-v1'


@dataclass(frozen=True)
class Calibration:
    temperature: float = 1.0
    alpha: float = 0.0
    neutral_bias: float = 0.0
    conflict_bias: float = 0.0

    @property
    def magnitude(self) -> float:
        return sum((abs(self.temperature - 1), abs(self.alpha),
                    abs(self.neutral_bias), abs(self.conflict_bias)))


def evidence_target(polarity: str) -> tuple[float, float]:
    if polarity not in EVIDENCE_TARGETS:
        raise ValueError(f'Unsupported polarity: {polarity}')
    return EVIDENCE_TARGETS[polarity]


def evidence_class_scores(torch: Any, evidence_logits: Any) -> Any:
    pos = torch.nn.functional.logsigmoid(evidence_logits[:, 0])
    no_pos = torch.nn.functional.logsigmoid(-evidence_logits[:, 0])
    neg = torch.nn.functional.logsigmoid(evidence_logits[:, 1])
    no_neg = torch.nn.functional.logsigmoid(-evidence_logits[:, 1])
    return torch.stack((pos + no_neg, no_pos + neg, no_pos + no_neg, pos + neg), dim=-1)


def adjusted_logits(torch: Any, polarity_logits: Any, evidence_logits: Any | None,
                    calibration: Calibration) -> Any:
    output = polarity_logits / calibration.temperature
    if evidence_logits is not None and calibration.alpha:
        output = output + calibration.alpha * evidence_class_scores(torch, evidence_logits)
    return output + torch.tensor((0., 0., calibration.neutral_bias, calibration.conflict_bias),
                                 dtype=output.dtype, device=output.device)


def evidence_positive_weights(torch: Any, rows: Sequence[LabeledRow]) -> Any:
    targets = [evidence_target(row.polarity) for row in rows]
    positives = [sum(target[index] for target in targets) for index in range(2)]
    return torch.tensor([(len(rows) - count) / count if count else 1. for count in positives],
                        dtype=torch.float32)


def combined_loss(torch: Any, polarity_logits: Any, evidence_logits: Any,
                  polarity_targets: Any, evidence_targets: Any, class_weights: Any,
                  evidence_weights: Any, evidence_lambda: float) -> tuple[Any, Any, Any]:
    classification = polarity_loss(torch, polarity_logits, polarity_targets,
                                   class_weights, LossSpec('weighted-ce'))
    evidence = torch.nn.functional.binary_cross_entropy_with_logits(
        evidence_logits, evidence_targets, pos_weight=evidence_weights)
    return classification + evidence_lambda * evidence, classification, evidence


class EvidenceDataset(PolarityDataset):
    def __init__(self, rows: Sequence[LabeledRow], tokenizer: Any, max_length: int):
        super().__init__(rows, tokenizer, max_length)
        for item, row in zip(self.items, rows):
            item['evidence_labels'] = list(evidence_target(row.polarity))


def build_evidence_model(torch: Any, transformers: Any, model_name: str) -> Any:
    class EvidenceModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = transformers.AutoModel.from_pretrained(model_name)
            hidden = int(self.encoder.config.hidden_size)
            self.dropout = torch.nn.Dropout(float(getattr(self.encoder.config, 'hidden_dropout_prob', .1)))
            self.polarity_head = torch.nn.Linear(hidden, len(POLARITIES))
            self.evidence_head = torch.nn.Linear(hidden, 2)

        def forward(self, **inputs: Any) -> tuple[Any, Any]:
            hidden = self.dropout(self.encoder(**inputs).last_hidden_state[:, 0])
            return self.polarity_head(hidden), self.evidence_head(hidden)
    return EvidenceModel()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train and validation-calibrate polarity evidence models.')
    for name, field in EvidenceConfig.__dataclass_fields__.items():
        default = field.default; option = f'--{name.replace("_", "-")}'
        if isinstance(default, Path): parser.add_argument(option, type=Path, default=default)
        elif isinstance(default, str): parser.add_argument(option, default=default)
        else: parser.add_argument(option, type=type(default), default=default)
    return parser


def config_from_args(argv: list[str] | None = None) -> EvidenceConfig:
    config = EvidenceConfig(**vars(build_parser().parse_args(argv)))
    positive = ('epochs', 'evaluation_interval', 'early_stopping_patience', 'max_length',
                'train_batch_size', 'eval_batch_size', 'gradient_accumulation_steps',
                'log_steps', 'bootstrap_samples')
    if any(getattr(config, name) < 1 for name in positive):
        raise ValueError('Epoch, batch, logging, and bootstrap values must be positive')
    if not 0 <= config.pair_f1_tolerance < 1 or not 0 < config.bootstrap_pass_rate <= 1:
        raise ValueError('Invalid tolerance or bootstrap pass rate')
    return config


def _training_config(config: EvidenceConfig) -> JointExperimentConfig:
    return JointExperimentConfig(input=config.input, split_dir=config.split_dir,
        run_dir=config.run_dir, evaluator=config.evaluator, model=config.model,
        epochs=config.epochs, evaluation_interval=config.evaluation_interval,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        max_length=config.max_length, learning_rate=config.learning_rate,
        train_batch_size=config.train_batch_size, eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        log_steps=config.log_steps, mixed_precision=config.mixed_precision,
        mlflow_tracking_uri=config.mlflow_tracking_uri, mlflow_experiment=config.mlflow_experiment)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file(): raise FileNotFoundError(f'Required artifact not found: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _source_paths(config: EvidenceConfig) -> tuple[list[Path], list[Path]]:
    aspect = _load_json(config.aspect_run_dir / 'locked_selection.json')
    baseline = _load_json(config.baseline_run_dir / 'locked_selection.json')
    aspect_paths = [Path(value) for value in aspect['checkpoints']]
    baseline_paths = [Path(value) for value in baseline['polarity_checkpoints']]
    if tuple(baseline['thresholds']) != FROZEN_ASPECT_THRESHOLDS:
        raise RuntimeError('Baseline aspect thresholds differ from frozen thresholds')
    for path in (*aspect_paths, *baseline_paths):
        if not path.is_file(): raise FileNotFoundError(f'Required checkpoint not found: {path}')
    return aspect_paths, baseline_paths


def _serialized(config: EvidenceConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _aspect_probabilities(torch: Any, transformers: Any, tokenizer: Any,
                          config: EvidenceConfig, checkpoints: Sequence[Path],
                          rows: Sequence[LabeledRow], label: str) -> list[list[float]]:
    aspect_config = AspectExperimentConfig(model=config.model, seeds=(42, 43, 44),
        max_length=config.max_length, eval_batch_size=config.eval_batch_size)
    return _ensemble_probabilities(torch, transformers, tokenizer, aspect_config,
                                   checkpoints, rows, label)


def _predict_evidence(torch: Any, transformers: Any, model: Any,
                      dataset: EvidenceDataset, batch_size: int, label: str) -> tuple[Any, Any]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
        collate_fn=transformers.DataCollatorWithPadding(tokenizer=dataset.tokenizer))
    polarity_values, evidence_values = [], []; device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=label, unit='batch', dynamic_ncols=True):
            batch.pop('labels'); batch.pop('evidence_labels')
            polarity, evidence = model(**{key: value.to(device) for key, value in batch.items()})
            polarity_values.append(polarity.cpu()); evidence_values.append(evidence.cpu())
    return torch.cat(polarity_values), torch.cat(evidence_values)


def _evidence_logits(torch: Any, transformers: Any, tokenizer: Any, config: EvidenceConfig,
                     checkpoint_path: Path, rows: list[LabeledRow], label: str) -> tuple[Any, Any]:
    reviews = group_aspect_targets(rows); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_evidence_model(torch, transformers, config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    dataset = _attach_tokenizer(EvidenceDataset(all_candidate_rows(reviews), tokenizer,
                                                config.max_length), tokenizer)
    values = _predict_evidence(torch, transformers, model, dataset, config.eval_batch_size, label)
    del model, checkpoint; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return values


def _gold_predictions(rows: Sequence[LabeledRow], indices: Sequence[int]) -> list[tuple[str, str, str]]:
    reviews = group_aspect_targets(rows); positions = {review.item_id: i for i, review in enumerate(reviews)}
    return [(row.item_id, row.aspect, POLARITIES[indices[positions[row.item_id] * len(ASPECTS) + ASPECTS.index(row.aspect)]]) for row in rows]


def evaluate_logits(torch: Any, rows: list[LabeledRow], aspects: list[list[float]],
                    polarity_logits: Any, evidence_logits: Any | None,
                    calibration: Calibration) -> dict[str, Any]:
    reviews = group_aspect_targets(rows)
    logits = adjusted_logits(torch, polarity_logits, evidence_logits, calibration)
    indices = logits.argmax(dim=-1).tolist()
    predictions = decode_pairs(reviews, aspects, _reshape(indices, len(reviews)), FROZEN_ASPECT_THRESHOLDS)
    metrics = calculate_metrics(rows, predictions); source = metrics['source_of_truth_metrics']
    gold_predictions = _gold_predictions(rows, indices); classes = _pair_class_metrics(rows, gold_predictions)
    probabilities = torch.softmax(logits, dim=-1).tolist(); positions = {r.item_id: i for i, r in enumerate(reviews)}
    summaries = {}
    for gold_label in POLARITIES:
        values = [probabilities[positions[row.item_id] * len(ASPECTS) + ASPECTS.index(row.aspect)]
                  for row in rows if row.polarity == gold_label]
        summaries[gold_label] = {label: sum(v[i] for v in values) / len(values) if values else 0.
                                 for i, label in enumerate(POLARITIES)}
    return {'calibration': asdict(calibration), 'predictions': predictions,
        'gold_aspect_predictions': gold_predictions,
        'pair_micro_f1': float(source['overall']['micro']['f1']),
        'polarity_micro_f1': float(source['polarity']['micro']['f1']),
        'exact_set_accuracy': float(metrics['supplemental_exact_match_accuracy']['overall']),
        'gold_aspect_polarity_classes': classes,
        'minority_f1': (float(classes['neutral']['f1']) + float(classes['conflict']['f1'])) / 2,
        'probability_summary': summaries}


def deterministic_eligible(candidate: dict[str, Any], baseline: dict[str, Any], tolerance: float) -> bool:
    return (candidate['gold_aspect_polarity_classes']['neutral']['f1'] >= baseline['gold_aspect_polarity_classes']['neutral']['f1']
        and candidate['gold_aspect_polarity_classes']['conflict']['f1'] >= baseline['gold_aspect_polarity_classes']['conflict']['f1']
        and candidate['pair_micro_f1'] >= baseline['pair_micro_f1'] - tolerance)


def _per_id_counts(rows: Sequence[LabeledRow], predictions: Sequence[tuple[str, str, str]],
                   label: str | None) -> dict[str, tuple[int, int, int]]:
    gold: dict[str, set[tuple[str, str]]] = {}; predicted: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        if label is None or row.polarity == label: gold.setdefault(row.item_id, set()).add((row.aspect, row.polarity))
    for item_id, aspect, polarity in predictions:
        if label is None or polarity == label: predicted.setdefault(item_id, set()).add((aspect, polarity))
    return {item_id: (len(gold.get(item_id, set()) & predicted.get(item_id, set())),
                      len(predicted.get(item_id, set()) - gold.get(item_id, set())),
                      len(gold.get(item_id, set()) - predicted.get(item_id, set())))
            for item_id in {row.item_id for row in rows}}


def _f1(counts: Sequence[tuple[int, int, int]]) -> float:
    tp = sum(v[0] for v in counts); fp = sum(v[1] for v in counts); fn = sum(v[2] for v in counts)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.


def bootstrap_stability(rows: Sequence[LabeledRow], baseline_predictions: Sequence[tuple[str, str, str]],
                        candidate_predictions: Sequence[tuple[str, str, str]], samples: int,
                        tolerance: float, seed: int = 20260910,
                        baseline_gold_predictions: Sequence[tuple[str, str, str]] | None = None,
                        candidate_gold_predictions: Sequence[tuple[str, str, str]] | None = None) -> dict[str, Any]:
    ids = sorted({row.item_id for row in rows}); rng = random.Random(seed); output = {}
    for label, key in (('neutral', 'neutral'), ('conflict', 'conflict'), (None, 'pair')):
        baseline_source = baseline_predictions if label is None else (baseline_gold_predictions or baseline_predictions)
        candidate_source = candidate_predictions if label is None else (candidate_gold_predictions or candidate_predictions)
        baseline = _per_id_counts(rows, baseline_source, label); candidate = _per_id_counts(rows, candidate_source, label)
        deltas = []
        for _ in range(samples):
            selected = [ids[rng.randrange(len(ids))] for _ in ids]
            deltas.append(_f1([candidate[i] for i in selected]) - _f1([baseline[i] for i in selected]))
        ordered = sorted(deltas); minimum = -tolerance if key == 'pair' else 0.
        output[key] = {'pass_rate': sum(value >= minimum for value in deltas) / len(deltas),
                       'delta_ci95': [ordered[int(.025 * len(ordered))], ordered[min(len(ordered)-1, int(.975 * len(ordered)))]]}
    return output


def _public(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in {'predictions', 'gold_aspect_predictions'}}


def confusion_matrix(rows: Sequence[LabeledRow], predictions: Sequence[tuple[str, str, str]]) -> dict[str, dict[str, int]]:
    predicted = {(item_id, aspect): polarity for item_id, aspect, polarity in predictions}
    return {gold: {guess: sum(
        row.polarity == gold and predicted.get((row.item_id, row.aspect)) == guess
        for row in rows
    ) for guess in POLARITIES} for gold in POLARITIES}


def calibrate(torch: Any, rows: list[LabeledRow], aspects: list[list[float]], polarity_logits: Any,
              evidence_logits: Any | None, baseline: dict[str, Any], config: EvidenceConfig
              ) -> tuple[Calibration, dict[str, Any], dict[str, Any]]:
    candidates = []
    for temperature in TEMPERATURE_GRID:
        for alpha in ALPHA_GRID if evidence_logits is not None else (0.,):
            for neutral_bias in BIAS_GRID:
                for conflict_bias in BIAS_GRID:
                    parameters = Calibration(temperature, alpha, neutral_bias, conflict_bias)
                    value = evaluate_logits(torch, rows, aspects, polarity_logits, evidence_logits, parameters)
                    if deterministic_eligible(value, baseline, config.pair_f1_tolerance):
                        rank = (value['minority_f1'], value['pair_micro_f1'],
                                value['gold_aspect_polarity_classes']['conflict']['f1'], -parameters.magnitude)
                        candidates.append((rank, parameters, value))
    candidates.sort(reverse=True, key=lambda item: item[0]); tried = []
    for _, parameters, value in candidates:
        stability = bootstrap_stability(rows, baseline['predictions'], value['predictions'],
            config.bootstrap_samples, config.pair_f1_tolerance,
            baseline_gold_predictions=baseline['gold_aspect_predictions'],
            candidate_gold_predictions=value['gold_aspect_predictions'])
        tried.append({'calibration': asdict(parameters), 'metrics': _public(value), 'bootstrap': stability})
        if all(result['pass_rate'] >= config.bootstrap_pass_rate for result in stability.values()):
            return parameters, value, {'stable': True, 'candidates': len(candidates), 'tried': tried}
    identity = Calibration(); value = evaluate_logits(torch, rows, aspects, polarity_logits, evidence_logits, identity)
    return identity, value, {'stable': False, 'candidates': len(candidates), 'tried': tried}


def _train_model(torch: Any, transformers: Any, mlflow: Any, tokenizer: Any, config: EvidenceConfig,
                 train_rows: list[LabeledRow], validation_rows: list[LabeledRow],
                 evidence_lambda: float, seed: int) -> Path:
    run_dir = config.run_dir / 'runs' / f'lambda-{evidence_lambda:g}' / f'seed-{seed}'
    best = run_dir / 'best.pt'; result_path = run_dir / 'result.json'
    manifest = {'architecture': 'polarity-evidence', 'lambda': evidence_lambda, 'seed': seed,
        'model': config.model, 'input_sha256': file_sha256(config.input),
        'split_sha256': {name: file_sha256(config.split_dir / f'{name}.csv') for name in ('train', 'eval', 'test')},
        'epochs': config.epochs, 'evaluation_interval': config.evaluation_interval,
        'max_length': config.max_length, 'learning_rate': config.learning_rate}
    run_dir.mkdir(parents=True, exist_ok=True); manifest_path = run_dir / 'manifest.json'
    if result_path.is_file() and _load_json(result_path).get('finished'):
        if _load_json(manifest_path) != manifest: raise RuntimeError(f'Completed run manifest mismatch: {run_dir}')
        return best
    if manifest_path.is_file() and _load_json(manifest_path) != manifest: raise RuntimeError(f'Run manifest mismatch: {run_dir}')
    atomic_write_json(manifest_path, manifest); _seed_everything(torch, seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = _attach_tokenizer(EvidenceDataset(train_rows, tokenizer, config.max_length), tokenizer)
    validation_dataset = _attach_tokenizer(EvidenceDataset(validation_rows, tokenizer, config.max_length), tokenizer)
    model = build_evidence_model(torch, transformers, config.model).to(device)
    optimizer, scheduler = _optimizer_and_schedule(torch, transformers, model, len(dataset), _training_config(config))
    use_fp16, autocast_dtype = _precision_settings(torch, config.mixed_precision, device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    class_weights = loss_weights(torch, train_rows, LossSpec('weighted-ce')).to(device)
    evidence_weights = evidence_positive_weights(torch, train_rows).to(device)
    latest = config.optimizer_dir / f'lambda-{evidence_lambda:g}' / f'seed-{seed}' / 'latest.pt'
    start_epoch = 0; best_score = -1.; stale = 0; global_step = 0
    if latest.is_file():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler']); scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = int(checkpoint['epoch']); best_score = float(checkpoint['best_score']); stale = int(checkpoint['stale']); global_step = int(checkpoint['global_step'])
    from .roberta_training import _prepare_mlflow_tracking_uri
    _prepare_mlflow_tracking_uri(config.mlflow_tracking_uri); mlflow.set_tracking_uri(config.mlflow_tracking_uri); mlflow.set_experiment(config.mlflow_experiment)
    with mlflow.start_run(run_name=f'evidence-lambda-{evidence_lambda:g}-seed-{seed}'):
        mlflow.log_params({'lambda': evidence_lambda, 'seed': seed, 'architecture': 'polarity-evidence'})
        for epoch in range(start_epoch + 1, config.epochs + 1):
            generator = torch.Generator().manual_seed(seed + epoch)
            loader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size,
                sampler=torch.utils.data.RandomSampler(dataset, generator=generator),
                collate_fn=transformers.DataCollatorWithPadding(tokenizer=tokenizer))
            model.train(); optimizer.zero_grad(set_to_none=True); running = 0.
            progress = tqdm(loader, desc=f'Evidence l={evidence_lambda:g} s={seed} {epoch}/{config.epochs}', unit='batch', dynamic_ncols=True)
            for batch_index, batch in enumerate(progress):
                targets = batch.pop('labels').to(device); ev_targets = batch.pop('evidence_labels').to(device)
                inputs = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                    polarity, evidence = model(**inputs)
                    raw_loss, _, _ = combined_loss(torch, polarity, evidence, targets, ev_targets,
                                                   class_weights, evidence_weights, evidence_lambda)
                    loss = raw_loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward(); running += float(raw_loss.detach().cpu())
                if (batch_index + 1) % config.gradient_accumulation_steps == 0 or batch_index + 1 == len(loader):
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step(); global_step += 1
                progress.set_postfix(loss=f'{running / (batch_index + 1):.4f}')
            evaluate_now = epoch % config.evaluation_interval == 0 or epoch == config.epochs
            if evaluate_now:
                validation_logits, _ = _predict_evidence(torch, transformers, model, validation_dataset,
                                                          config.eval_batch_size, f'Validate evidence {epoch}')
                indices = validation_logits.argmax(dim=-1).tolist()
                classes = _pair_class_metrics(validation_rows, [(row.item_id, row.aspect, POLARITIES[index]) for row, index in zip(validation_rows, indices)])
                score = (float(classes['neutral']['f1']) + float(classes['conflict']['f1'])) / 2
                if score > best_score + config.early_stopping_min_delta:
                    best_score = score; stale = 0
                    compact = {name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu() for name, value in model.state_dict().items()}
                    _atomic_torch_save(torch, {'model': compact, 'epoch': epoch, 'minority_f1': score,
                                               'classes': classes, 'lambda': evidence_lambda, 'seed': seed}, best)
                else: stale += 1
                mlflow.log_metrics({'validation/minority_f1': score, 'train/epoch_loss': running / len(loader)}, step=epoch)
            _atomic_torch_save(torch, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
                'best_score': best_score, 'stale': stale, 'global_step': global_step}, latest)
            if evaluate_now and stale >= config.early_stopping_patience: break
    checkpoint = torch.load(best, map_location='cpu', weights_only=False)
    atomic_write_json(result_path, {'finished': True, 'epoch': checkpoint['epoch'],
                                    'minority_f1': checkpoint['minority_f1'], 'lambda': evidence_lambda, 'seed': seed})
    latest.unlink(missing_ok=True); del model, optimizer, scheduler, scaler; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return best


def _baseline_logits(torch: Any, transformers: Any, tokenizer: Any, config: EvidenceConfig,
                     paths: Sequence[Path], rows: list[LabeledRow], label: str) -> Any:
    return _mean_logits(torch, [_separate_logits(torch, transformers, tokenizer, _training_config(config),
        path, rows, f'{label} {index+1}/{len(paths)}') for index, path in enumerate(paths)])


def _ensemble_evidence(torch: Any, transformers: Any, tokenizer: Any, config: EvidenceConfig,
                       paths: Sequence[Path], rows: list[LabeledRow], label: str) -> tuple[Any, Any]:
    values = [_evidence_logits(torch, transformers, tokenizer, config, path, rows,
                               f'{label} {index+1}/{len(paths)}') for index, path in enumerate(paths)]
    return _mean_logits(torch, [v[0] for v in values]), _mean_logits(torch, [v[1] for v in values])


def _write_report(config: EvidenceConfig, results: dict[str, Any]) -> None:
    def table(values: dict[str, Any]) -> str:
        return '\n'.join(f"| {name} | {v['pair_micro_f1']:.6f} | {v['polarity_micro_f1']:.6f} | {v['exact_set_accuracy']:.6f} | {v['gold_aspect_polarity_classes']['neutral']['f1']:.6f} | {v['gold_aspect_polarity_classes']['conflict']['f1']:.6f} |" for name, v in values.items())
    atomic_write_text(config.run_dir / 'report.md', f'''# Validation-calibrated polarity evidence experiment

Evidence targets encode positive and negative evidence independently: positive `10`, negative `01`, neutral `00`, and conflict `11`. Aspect checkpoints and thresholds were frozen.

Selected lambda: `{results['selected_lambda']}`. Locked winner: `{results['winner']}`.

## Validation

| System | Pair F1 | Polarity F1 | Exact set | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|
{table(results['validation'])}

## Held-out test

| System | Pair F1 | Polarity F1 | Exact set | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|
{table(results['test'])}

Selection used validation only. Test results are analysis-only and did not change the locked winner. This test split was accessed by prior experiments.

Runtime: {results['runtime_seconds']/60:.1f} minutes.
''')


def run_experiment(config: EvidenceConfig) -> None:
    import mlflow
    import torch
    import transformers
    started = time.monotonic(); config.run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(config.run_dir / '.experiment.lock'):
        if (config.run_dir / 'results.json').is_file(): LOGGER.info('Experiment already complete'); return
        aspect_paths, baseline_paths = _source_paths(config)
        manifest = {**_serialized(config), 'lambda_grid': list(LAMBDA_GRID),
            'temperature_grid': list(TEMPERATURE_GRID), 'alpha_grid': list(ALPHA_GRID),
            'bias_grid': list(BIAS_GRID), 'aspect_thresholds': list(FROZEN_ASPECT_THRESHOLDS),
            'aspect_checkpoints': [str(p) for p in aspect_paths], 'baseline_checkpoints': [str(p) for p in baseline_paths],
            'input_sha256': file_sha256(config.input)}
        path = config.run_dir / 'manifest.json'
        if path.is_file() and _load_json(path) != manifest: raise RuntimeError('Existing manifest differs')
        atomic_write_json(path, manifest); atomic_write_json(config.run_dir / 'config.json', _serialized(config))
        train_rows = read_labeled_csv(config.split_dir / 'train.csv'); validation_rows = read_labeled_csv(config.split_dir / 'eval.csv')
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, use_fast=True)
        aspects = _aspect_probabilities(torch, transformers, tokenizer, config, aspect_paths, validation_rows, 'Validation aspects')
        base_logits = _baseline_logits(torch, transformers, tokenizer, config, baseline_paths, validation_rows, 'Validation baseline')
        baseline = evaluate_logits(torch, validation_rows, aspects, base_logits, None, Calibration())
        base_cal, calibrated_base, base_search = calibrate(torch, validation_rows, aspects, base_logits, None, baseline, config)
        screening = {}
        for lam in LAMBDA_GRID:
            model_path = _train_model(torch, transformers, mlflow, tokenizer, config, train_rows, validation_rows, lam, 42)
            polarity, evidence = _ensemble_evidence(torch, transformers, tokenizer, config, [model_path], validation_rows, f'Validation lambda {lam:g}')
            cal, value, search = calibrate(torch, validation_rows, aspects, polarity, evidence, baseline, config)
            screening[str(lam)] = {'checkpoint': str(model_path), 'calibration': asdict(cal), 'metrics': _public(value), 'search': search}
            atomic_write_json(config.run_dir / 'screening.json', screening)
        stable = [name for name, value in screening.items() if value['search']['stable']]; pool = stable or list(screening)
        selected_lambda = float(max(pool, key=lambda name: (screening[name]['metrics']['minority_f1'], screening[name]['metrics']['pair_micro_f1'], -float(name))))
        evidence_paths = [_train_model(torch, transformers, mlflow, tokenizer, config, train_rows, validation_rows, selected_lambda, seed) for seed in CONFIRMATION_SEEDS]
        ev_polarity, ev_evidence = _ensemble_evidence(torch, transformers, tokenizer, config, evidence_paths, validation_rows, 'Validation evidence ensemble')
        raw_evidence = evaluate_logits(torch, validation_rows, aspects, ev_polarity, ev_evidence, Calibration())
        ev_cal, calibrated_evidence, ev_search = calibrate(torch, validation_rows, aspects, ev_polarity, ev_evidence, baseline, config)
        validation = {'original': baseline, 'calibrated_original': calibrated_base,
                      'evidence_raw': raw_evidence, 'evidence_calibrated': calibrated_evidence}
        stable_finalists = []
        if base_search['stable']: stable_finalists.append('calibrated_original')
        if ev_search['stable']: stable_finalists.append('evidence_calibrated')
        winner = max(stable_finalists, key=lambda name: (
            validation[name]['minority_f1'], validation[name]['pair_micro_f1'],
            validation[name]['gold_aspect_polarity_classes']['conflict']['f1'],
        )) if stable_finalists else 'original'
        seed_validation = {}
        for seed, model_path in zip(CONFIRMATION_SEEDS, evidence_paths):
            seed_polarity, seed_evidence = _ensemble_evidence(
                torch, transformers, tokenizer, config, [model_path], validation_rows,
                f'Validation evidence seed {seed}',
            )
            seed_validation[str(seed)] = _public(evaluate_logits(
                torch, validation_rows, aspects, seed_polarity, seed_evidence, ev_cal,
            ))
        locked = {'winner': winner, 'selected_lambda': selected_lambda,
            'aspect_checkpoints': [str(p) for p in aspect_paths], 'baseline_checkpoints': [str(p) for p in baseline_paths],
            'evidence_checkpoints': [str(p) for p in evidence_paths], 'aspect_thresholds': list(FROZEN_ASPECT_THRESHOLDS),
            'baseline_calibration': asdict(base_cal), 'evidence_calibration': asdict(ev_cal),
            'validation': {name: _public(v) for name, v in validation.items()}, 'test_accessed': False}
        atomic_write_json(config.run_dir / 'locked_selection.json', locked)
        test_rows = read_labeled_csv(config.split_dir / 'test.csv')
        test_aspects = _aspect_probabilities(torch, transformers, tokenizer, config, aspect_paths, test_rows, 'Test aspects')
        test_base = _baseline_logits(torch, transformers, tokenizer, config, baseline_paths, test_rows, 'Test baseline')
        test_pol, test_ev = _ensemble_evidence(torch, transformers, tokenizer, config, evidence_paths, test_rows, 'Test evidence ensemble')
        test = {'original': evaluate_logits(torch, test_rows, test_aspects, test_base, None, Calibration()),
            'calibrated_original': evaluate_logits(torch, test_rows, test_aspects, test_base, None, base_cal),
            'evidence_raw': evaluate_logits(torch, test_rows, test_aspects, test_pol, test_ev, Calibration()),
            'evidence_calibrated': evaluate_logits(torch, test_rows, test_aspects, test_pol, test_ev, ev_cal)}
        for name, value in test.items():
            _write_evaluation(_training_config(config), config.run_dir / 'test' / name, test_rows, value['predictions'])
        results = {'selected_lambda': selected_lambda, 'screening': screening,
            'calibration_search': {'baseline': base_search, 'evidence': ev_search},
            'validation': {name: _public(v) for name, v in validation.items()}, 'winner': winner,
            'test': {name: _public(v) for name, v in test.items()}, 'seed_models': [str(p) for p in evidence_paths],
            'seed_validation': seed_validation,
            'confusion_matrices': {
                'validation': {name: confusion_matrix(validation_rows, value['gold_aspect_predictions']) for name, value in validation.items()},
                'test': {name: confusion_matrix(test_rows, value['gold_aspect_predictions']) for name, value in test.items()},
            }, 'runtime_seconds': time.monotonic() - started}
        atomic_write_json(config.run_dir / 'results.json', results); _write_report(config, results)
        locked['test_accessed'] = True; atomic_write_json(config.run_dir / 'locked_selection.json', locked)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    try: run_experiment(config_from_args(argv))
    except KeyboardInterrupt: LOGGER.warning('Interrupted; rerun to resume'); return 130
    except Exception: LOGGER.exception('Polarity evidence experiment failed'); return 1
    return 0
