"""Multi-seed comparison of synthetic and repeated real conflict examples."""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from .absa_training import ExperimentConfig
from .checkpoint import RunLock
from .conflict_augmentation import (
    DIAGNOSTICS, digest, metrics, paired_intervals, source_data, train_one,
)
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .labels import POLARITIES
from .learning_curve_experiment import rows_for_ids, _write_or_validate
from .roberta_training import _prepare_mlflow_tracking_uri

LOGGER = logging.getLogger(__name__)
CONDITIONS = ('repetition', 'synthetic')
SEEDS = (42, 1337, 2024)
BOOTSTRAP_SAMPLES = 10_000


def conflict_examples(prepared_fold: dict) -> list[dict]:
    examples = [value for value in prepared_fold['examples'] if value['kind'] == 'conflict']
    if len(examples) != 40 or any(value['polarity'] != 'conflict' for value in examples):
        raise ValueError('Each fold requires exactly 40 reviewed synthetic conflicts')
    return examples


def matched_counts(fit: Sequence[Any], examples: Sequence[dict]) -> dict[str, Counter]:
    requested = Counter(value['aspect'] for value in examples)
    available = Counter(row.aspect for row in fit if row.polarity == 'conflict')
    if any(available[aspect] == 0 for aspect in requested):
        raise ValueError('Every requested synthetic aspect needs a real conflict control')
    return {'requested': requested, 'available': available}


def freeze_repetitions(fit: Sequence[Any], examples: Sequence[dict], seed: int) -> list[dict]:
    """Bind every synthetic conflict to one same-aspect real conflict row."""
    rng = random.Random(seed)
    frozen = []
    for example in examples:
        matches = [row for row in fit if row.aspect == example['aspect'] and row.polarity == 'conflict']
        if not matches:
            raise ValueError('Missing same-aspect real conflict row')
        row = rng.choice(matches)
        frozen.append({**example, 'repetition_position': row.position,
                       'repetition_id': row.item_id,
                       'repetition_text_sha256': file_sha256_text(row.text)})
    return frozen


def file_sha256_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def average_records(runs: Sequence[list[dict]]) -> list[dict]:
    if not runs:
        raise ValueError('At least one prediction run is required')
    indexed = [{row['position']: row for row in run} for run in runs]
    if any(len(value) != len(run) for value, run in zip(indexed, runs)):
        raise ValueError('Duplicate prediction positions')
    positions = sorted(indexed[0])
    if not positions or any(sorted(value) != positions for value in indexed[1:]):
        raise ValueError('Prediction coverage differs across seeds')
    result = []
    for position in positions:
        rows = [value[position] for value in indexed]
        if any(any(row[key] != rows[0][key] for key in ('id', 'aspect', 'gold')) for row in rows[1:]):
            raise ValueError('Prediction identity or gold differs across seeds')
        logits = [statistics.fmean(row['logits'][label] for row in rows) for label in range(4)]
        if not all(math.isfinite(value) for value in logits):
            raise ValueError('Non-finite averaged logits')
        result.append({**rows[0], 'logits': logits,
                       'predicted': POLARITIES[max(range(4), key=logits.__getitem__)]})
    return result


def collect(run_dir: Path, folds: Sequence[dict]) -> tuple[dict, dict]:
    results: dict[str, dict[int, list[dict]]] = {
        condition: {seed: [] for seed in SEEDS} for condition in CONDITIONS
    }
    raw = {}
    for fold in folds:
        fold_number = fold['fold']
        raw[fold_number] = {}
        for seed in SEEDS:
            raw[fold_number][seed] = {}
            for condition in CONDITIONS:
                path = run_dir / f'fold-{fold_number}' / f'seed-{seed}' / condition / 'result.json'
                value = json.loads(path.read_text())
                if not value.get('finished'):
                    raise ValueError(f'Unfinished result: {path}')
                raw[fold_number][seed][condition] = value
                results[condition][seed].extend(value['predictions']['heldout'])
    return results, raw


def report(run_dir: Path, samples: int = BOOTSTRAP_SAMPLES) -> dict:
    prepared = json.loads((run_dir / 'preparation.json').read_text())
    folds = [{'fold': value['fold']} for value in prepared['folds']]
    predictions, raw = collect(run_dir, folds)
    per_seed = {}
    seed_ensembles = {condition: [] for condition in CONDITIONS}
    for seed in SEEDS:
        per_seed[str(seed)] = {}
        for condition in CONDITIONS:
            records = predictions[condition][seed]
            per_seed[str(seed)][condition] = metrics(records)
        per_seed[str(seed)]['synthetic_minus_repetition'] = paired_intervals(
            predictions['synthetic'][seed], predictions['repetition'][seed], samples
        )
    for condition in CONDITIONS:
        seed_ensembles[condition] = average_records(
            [predictions[condition][seed] for seed in SEEDS]
        )
    ensemble = {condition: metrics(records) for condition, records in seed_ensembles.items()}
    interval = paired_intervals(seed_ensembles['synthetic'], seed_ensembles['repetition'], samples)
    fold_seed_deltas = []
    for fold in folds:
        for seed in SEEDS:
            left = raw[fold['fold']][seed]['synthetic']['metrics']['heldout']['classes']['conflict']['f1']
            right = raw[fold['fold']][seed]['repetition']['metrics']['heldout']['classes']['conflict']['f1']
            fold_seed_deltas.append({'fold': fold['fold'], 'seed': seed, 'delta': left - right,
                                     'synthetic': left, 'repetition': right})
    seed_deltas = [per_seed[str(seed)]['synthetic']['classes']['conflict']['f1'] -
                   per_seed[str(seed)]['repetition']['classes']['conflict']['f1'] for seed in SEEDS]
    primary = interval['conflict']
    decision = {
        'synthetic_wording_supported': (
            primary['delta'] >= 0.03 and primary['lower_95'] > 0
            and sum(value > 0 for value in seed_deltas) >= 2
            and ensemble['synthetic']['macro_f1'] >= ensemble['repetition']['macro_f1'] - 0.01
            and ensemble['synthetic']['classes']['neutral']['f1'] >= ensemble['repetition']['classes']['neutral']['f1'] - 0.01
            and ensemble['synthetic']['conflict_false_positive_rate'] <= ensemble['repetition']['conflict_false_positive_rate'] + 0.01
        ),
        'thresholds': {'minimum_conflict_gain': 0.03, 'minimum_winning_seeds': 2,
                       'maximum_macro_neutral_drop': 0.01,
                       'maximum_conflict_false_positive_increase': 0.01},
    }
    summary = {
        'design': {'conditions': list(CONDITIONS), 'seeds': list(SEEDS), 'folds': 5,
                   'added_conflicts_per_fold': 40, 'fits': 30,
                   'primary': 'three-seed logit ensemble synthetic minus repetition conflict F1'},
        'ensemble': ensemble, 'ensemble_synthetic_minus_repetition': interval,
        'per_seed': per_seed, 'seed_conflict_deltas': seed_deltas,
        'fold_seed_conflict_deltas': fold_seed_deltas,
        'decision': decision,
        'training': {condition: {
            'seconds': sum(raw[fold['fold']][seed][condition]['training_seconds'] for fold in folds for seed in SEEDS),
            'selected_epochs': [raw[fold['fold']][seed][condition]['training']['epoch'] for fold in folds for seed in SEEDS],
            'peak_allocated_gib': max(raw[fold['fold']][seed][condition]['peak_training_allocated_gib'] for fold in folds for seed in SEEDS),
        } for condition in CONDITIONS},
        'limitations': [
            'The review bootstrap conditions on the fitted three-seed ensembles.',
            'Three seeds characterize but do not eliminate training uncertainty.',
            'The same synthetic text is reused across seeds; only training randomness changes.',
            'This polarity-only experiment does not select a complete-system winner.',
        ],
    }
    atomic_write_json(run_dir / 'summary.json', summary)
    lines = [
        '# Multi-seed synthetic-conflict isolation', '',
        'RoBERTa polarity-only comparison on five grouped outer folds. Both arms add exactly '
        '40 conflict rows per fold with identical aspect counts. Repetition duplicates frozen '
        'real conflict rows; synthetic uses reviewed same-aspect positive/negative clause combinations. '
        'Natural selection and heldout partitions are unchanged.', '',
        '| Three-seed ensemble | Accuracy | Macro-F1 | Neutral F1 | Conflict F1 | Conflict FP rate |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for condition in CONDITIONS:
        value = ensemble[condition]
        lines.append(f"| {condition} | {value['accuracy']:.4f} | {value['macro_f1']:.4f} | "
                     f"{value['classes']['neutral']['f1']:.4f} | {value['classes']['conflict']['f1']:.4f} | "
                     f"{value['conflict_false_positive_rate']:.4f} |")
    lines += ['', f"Primary conflict-F1 delta: {primary['delta']:+.4f}; paired review-bootstrap 95% interval "
              f"[{primary['lower_95']:+.4f}, {primary['upper_95']:+.4f}].",
              f"Per-seed pooled conflict-F1 deltas: {', '.join(f'{value:+.4f}' for value in seed_deltas)}.",
              f"Synthetic wording supported: {decision['synthetic_wording_supported']}.", '',
              'Full per-seed and fold-seed results, class metrics, confusion matrices, timing, '
              'selected epochs and decision thresholds are in `summary.json`. Historical validation '
              'and test partitions were not used.', '']
    atomic_write_text(run_dir / 'report.md', '\n'.join(lines))
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'report'))
    parser.add_argument('--source-run', type=Path, default=Path('artifacts/experiments/conflict-augmentation-v1'))
    parser.add_argument('--source-splits', type=Path, default=Path('artifacts/experiments/learning-curve-v1'))
    parser.add_argument('--run-dir', type=Path, default=Path('artifacts/experiments/conflict-augmentation-multiseed-v1'))
    args = parser.parse_args(argv)
    source_preparation = json.loads((args.source_run / 'preparation.json').read_text())
    prepared = {
        'source_preparation_sha256': digest(source_preparation),
        'source_reviews_sha256': file_sha256(args.source_run / 'reviews.json'),
        'source_manifest_sha256': file_sha256(args.source_splits / 'manifest.json'),
        'source_splits_sha256': file_sha256(args.source_splits / 'splits.json'),
        'conditions': list(CONDITIONS), 'seeds': list(SEEDS), 'folds': [],
    }
    rows, folds = source_data(args.source_splits)
    for fold in folds:
        value = next(item for item in source_preparation['folds'] if item['fold'] == fold['fold'])
        fit = rows_for_ids(rows, fold['fit_ids'])
        examples = conflict_examples(value)
        counts = matched_counts(fit, examples)
        data_seed = 50_000 + fold['fold']
        examples = freeze_repetitions(fit, examples, data_seed)
        prepared['folds'].append({'fold': fold['fold'], 'examples': examples,
                                  'synthetic_aspect_counts': dict(counts['requested']),
                                  'available_real_conflicts': dict(counts['available']),
                                  'repetition_data_seed': data_seed})
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare':
        _write_or_validate(args.run_dir / 'preparation.json', prepared)
        print(json.dumps({'folds': [{'fold': f['fold'], 'counts': f['synthetic_aspect_counts']}
                                   for f in prepared['folds']]}, indent=2))
        return 0
    frozen = json.loads((args.run_dir / 'preparation.json').read_text())
    if frozen != prepared:
        raise ValueError('Prepared source or matching design changed')
    if args.mode == 'report':
        report(args.run_dir)
        return 0
    import mlflow
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    config = ExperimentConfig(
        input=Path('artifacts/training/roberta-aspect-exp1/splits/train.csv'),
        model='FacebookAI/roberta-base', run_dir=args.run_dir, checkpoint_steps=0,
        mlflow_tracking_uri='sqlite:///' + str(Path('artifacts/mlflow.db').resolve()),
        mlflow_experiment='contest2-conflict-augmentation-multiseed-v1',
    )
    with RunLock(args.run_dir / '.experiment.lock'):
        _write_or_validate(args.run_dir / 'manifest.json', {
            'preparation_sha256': digest(prepared), 'config': {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            }, 'conditions': list(CONDITIONS), 'seeds': list(SEEDS), 'fits': 30,
        })
        _prepare_mlflow_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, use_fast=True)
        completed = 0
        for fold in folds:
            data = next(value for value in prepared['folds'] if value['fold'] == fold['fold'])
            fit = rows_for_ids(rows, fold['fit_ids'])
            for seed in SEEDS:
                # Pair conditions within a seed and fold. The repetition data seed
                # stays fixed across model seeds, freezing which real rows repeat.
                training_seed = seed + fold['fold']
                for condition in CONDITIONS:
                    run_dir = args.run_dir / f"fold-{fold['fold']}" / f'seed-{seed}' / condition
                    train_one(replace(config, run_dir=run_dir, seed=training_seed), condition,
                              fit, rows_for_ids(rows, fold['selection_ids']),
                              rows_for_ids(rows, fold['heldout_ids']), data['examples'],
                              torch, transformers, mlflow, tokenizer,
                              data_seed=data['repetition_data_seed'])
                    completed += 1
                    atomic_write_json(args.run_dir / 'progress.json', {'completed': completed, 'total': 30})
                    LOGGER.info('Completed %d/30 fits', completed)
        report(args.run_dir)
        atomic_write_text(args.run_dir / 'completed_at.txt', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')
    return 0
