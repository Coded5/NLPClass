"""Assistant-reviewed synthetic conflict pilot with matched repetition controls."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import random
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .absa_training import (
    ExperimentConfig, PolarityDataset, _attach_tokenizer, _predict_logits, train_stage,
)
from .checkpoint import RunLock
from .data import LabeledRow, read_labeled_csv
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .labels import POLARITIES
from .learning_curve_experiment import (
    classification_metrics, rows_for_ids, validate_split_plan, _write_or_validate,
)
from .roberta_training import _prepare_mlflow_tracking_uri, _seed_everything

LOGGER = logging.getLogger(__name__)
CONDITIONS = ('reference', 'repetition', 'synthetic')
RULES = {'target_conflicts': 40, 'minimum_conflicts': 30, 'max_clause_uses': 5,
         'primary_gain': 0.03, 'required_positive_folds': 4,
         'maximum_macro_neutral_drop': 0.01, 'maximum_false_positive_increase': 0.01,
         'bootstrap_samples': 10000}

# Authored independently of the source bank; diagnostic only, never selection.
DIAGNOSTICS = [
    ('The waiter explained the menu clearly, but forgot our drinks.', 'service', 'conflict', 'same-aspect'),
    ('The waiter forgot our drinks, but explained the menu clearly.', 'service', 'conflict', 'reversed'),
    ('The broth had a wonderful flavor. The noodles were unpleasantly rubbery.', 'food', 'conflict', 'no-conjunction'),
    ('The staff greeted us warmly, although clearing the table took far too long.', 'service', 'conflict', 'same-aspect'),
    ('The tables were beautifully arranged. The music was painfully loud.', 'ambience', 'conflict', 'no-conjunction'),
    ('The soup was excellent, but the waiter was rude.', 'food', 'positive', 'cross-aspect'),
    ('The soup was excellent, but the waiter was rude.', 'service', 'negative', 'cross-aspect'),
    ('Although the room was noisy, the fish was delicious.', 'food', 'positive', 'cross-aspect'),
    ('Although the room was noisy, the fish was delicious.', 'ambience', 'negative', 'cross-aspect'),
    ('The waiter was friendly, but the dessert was stale.', 'service', 'positive', 'cross-aspect'),
    ('The waiter was friendly, but the dessert was stale.', 'food', 'negative', 'cross-aspect'),
    ('The dishes were tasty, though quite unfamiliar to me.', 'food', 'positive', 'non-conflict-contrast'),
]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_data(source: Path) -> tuple[list[LabeledRow], list[dict]]:
    manifest = json.loads((source / 'manifest.json').read_text())
    path = Path(manifest['config']['input'])
    if file_sha256(path) != manifest['input_sha256']:
        raise ValueError('Source training data changed')
    rows = read_labeled_csv(path)
    raw = json.loads((source / 'splits.json').read_text())['folds']
    plan = [{**f, **{k: set(f[k]) for k in ('fit_ids', 'selection_ids', 'heldout_ids')},
             'subsets': {int(k): set(v) for k, v in f['subsets'].items()}} for f in raw]
    validate_split_plan(rows, plan)
    if len(plan) != 5 or len({f['fold'] for f in plan}) != 5:
        raise ValueError('Expected five unique source folds')
    return rows, plan


def clause_bank(path: Path, rows: list[LabeledRow]) -> list[dict]:
    contents = json.loads(path.read_text())
    result = []
    for item_id, aspect, polarity, span in contents['clauses']:
        matches = [r for r in rows if (r.item_id, r.aspect, r.polarity) == (item_id, aspect, polarity) and span in r.text]
        if len(matches) != 1 or polarity not in ('positive', 'negative'):
            raise ValueError(f'Invalid or nonunique clause source: {item_id} {span}')
        row = matches[0]
        start = row.text.index(span)
        result.append({'key': digest([item_id, aspect, span])[:16], 'id': item_id,
                       'aspect': aspect, 'polarity': polarity, 'span': span,
                       'start': start, 'end': start + len(span),
                       'rationale': contents['rationale']})
    if len({c['key'] for c in result}) != len(result):
        raise ValueError('Duplicate source clauses')
    return result


def paired_intervals(left: list[dict], right: list[dict], samples: int = 10000) -> dict:
    import numpy as np

    if samples < 1:
        raise ValueError('Bootstrap samples must be positive')
    a = {r['position']: r for r in left}
    b = {r['position']: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or not a or a.keys() != b.keys():
        raise ValueError('Paired predictions must cover identical unique annotation positions')
    ids = {value: i for i, value in enumerate(sorted({r['id'] for r in left}))}
    counts = np.zeros((len(ids), 2, 4, 4), dtype=np.int64)
    for position, row in a.items():
        other = b[position]
        if any(row[k] != other[k] for k in ('id', 'aspect', 'gold')):
            raise ValueError('Paired identity or gold differs')
        for model, prediction in enumerate((row, other)):
            counts[ids[row['id']], model, POLARITIES.index(row['gold']), POLARITIES.index(prediction['predicted'])] += 1
    rng = np.random.default_rng(42)
    changes = []
    for _ in range(samples):
        matrix = counts[rng.integers(len(ids), size=len(ids))].sum(axis=0)
        denominator = matrix.sum(axis=1) + matrix.sum(axis=2)
        f1 = np.divide(2 * np.diagonal(matrix, axis1=1, axis2=2), denominator,
                       out=np.zeros((2, 4)), where=denominator != 0)
        changes.append([*(f1[0] - f1[1]), float(f1[0].mean() - f1[1].mean())])
    intervals = np.quantile(changes, [0.025, 0.975], axis=0)
    lm, rm = metrics(left), metrics(right)
    return {name: {'delta': (lm['macro_f1'] - rm['macro_f1']) if name == 'macro_f1'
                  else lm['classes'][name]['f1'] - rm['classes'][name]['f1'],
                  'lower_95': float(intervals[0, i]), 'upper_95': float(intervals[1, i]),
                  'samples': samples} for i, name in enumerate((*POLARITIES, 'macro_f1'))}


def promotion(models: dict, intervals: dict, improved_folds: int) -> dict:
    synthetic = models['synthetic']['pooled']
    checks = {
        'conflict_gain': intervals['conflict']['delta'] >= RULES['primary_gain'],
        'positive_interval': intervals['conflict']['lower_95'] > 0,
        'four_folds': improved_folds >= RULES['required_positive_folds'],
        'beats_reference': synthetic['classes']['conflict']['f1'] > models['reference']['pooled']['classes']['conflict']['f1'],
    }
    for name in ('reference', 'repetition'):
        baseline = models[name]['pooled']
        checks[f'{name}_macro_guard'] = synthetic['macro_f1'] >= baseline['macro_f1'] - 0.01
        checks[f'{name}_neutral_guard'] = synthetic['classes']['neutral']['f1'] >= baseline['classes']['neutral']['f1'] - 0.01
        checks[f'{name}_false_positive_guard'] = synthetic['conflict_false_positive_rate'] <= baseline['conflict_false_positive_rate'] + 0.01
    return {'advance_to_confirmation': all(checks.values()), 'checks': checks}


def report(run_dir: Path, samples: int = 10000) -> dict:
    prepared = json.loads((run_dir / 'preparation.json').read_text())
    models = {}
    predictions = {}
    folds = {}
    for condition in CONDITIONS:
        results = [json.loads((run_dir / f"fold-{f['fold']}" / condition / 'result.json').read_text()) for f in prepared['folds']]
        if not all(r.get('finished') for r in results):
            raise ValueError('Cannot report an unfinished condition')
        predictions[condition] = [p for r in results for p in r['predictions']['heldout']]
        folds[condition] = [r['metrics']['heldout'] for r in results]
        contrast = [p for p in predictions[condition] if re.search(r'\b(but|though|although|however)\b', p['text'], re.I)]
        models[condition] = {
            'pooled': metrics(predictions[condition]), 'folds': folds[condition],
            'fold_macro_mean': statistics.fmean(m['macro_f1'] for m in folds[condition]),
            'fold_macro_std': statistics.stdev(m['macro_f1'] for m in folds[condition]),
            'fold_conflict_mean': statistics.fmean(m['classes']['conflict']['f1'] for m in folds[condition]),
            'fold_conflict_std': statistics.stdev(m['classes']['conflict']['f1'] for m in folds[condition]),
            'contrastive_text': metrics(contrast) if contrast else None,
            'diagnostic': [r['metrics']['diagnostic'] for r in results],
            'selected_epochs': [r['training']['epoch'] for r in results],
            'training_seconds': sum(r['training_seconds'] for r in results),
            'peak_training_allocated_gib': max(r['peak_training_allocated_gib'] for r in results),
        }
    comparisons = {name: paired_intervals(predictions['synthetic'], predictions[name], samples)
                   for name in ('reference', 'repetition')}
    improved = sum(a['classes']['conflict']['f1'] > b['classes']['conflict']['f1']
                   for a, b in zip(folds['synthetic'], folds['repetition']))
    summary = {'models': models, 'synthetic_minus': comparisons,
               'improved_folds_over_repetition': improved,
               'decision': promotion(models, comparisons['repetition'], improved)}
    atomic_write_json(run_dir / 'summary.json', summary)
    lines = ['# Synthetic conflict augmentation pilot', '',
             'Five grouped folds, RoBERTa-base, natural gold-aspect polarity evaluation. '
             'All checkpoints selected using natural inner-selection macro-F1. '
             'Synthetic labels are provisional assistant judgments. No historical validation or test was evaluated.', '',
             '| Condition | Accuracy | Macro-F1 | Positive F1 | Negative F1 | Neutral F1 | Conflict F1 | Conflict FP rate |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, value in models.items():
        m = value['pooled']
        scores = [m['accuracy'], m['macro_f1'], *[m['classes'][p]['f1'] for p in POLARITIES], m['conflict_false_positive_rate']]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{s:.4f}' for s in scores) + ' |')
    interval = comparisons['repetition']['conflict']
    lines += ['', f"Primary conflict-F1 delta: {interval['delta']:+.4f}; paired review-bootstrap 95% interval "
              f"[{interval['lower_95']:+.4f}, {interval['upper_95']:+.4f}]. Improved folds: {improved}/5.",
              '', f"Advance to ensemble confirmation: {summary['decision']['advance_to_confirmation']}.", '',
              'The intervals condition on fitted models and do not include training-seed uncertainty. '
              'Cross-aspect diagnostic and conjunction-subset results are descriptive only. '
              'The augmentation and repetition arms have matched epoch lengths and maximum update budgets; '
              'early stopping can produce different realized update counts. This pilot does not select a new complete-system winner.', '',
              'Full class precision/recall/support, confusion matrices, per-fold variation, guardrail checks, '
              'epochs, timing and memory are in `summary.json`; logits and texts are in each `result.json`.', '']
    atomic_write_text(run_dir / 'report.md', '\n'.join(lines))
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'report'))
    parser.add_argument('--run-dir', type=Path, default=Path('artifacts/experiments/conflict-augmentation-v1'))
    parser.add_argument('--source-dir', type=Path, default=Path('artifacts/experiments/learning-curve-v1'))
    parser.add_argument('--clause-bank', type=Path, default=Path('scripts/conflict_clause_bank.json'))
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args(argv)
    if args.mode == 'prepare':
        value = prepare(args.source_dir, args.clause_bank, args.run_dir)
        print('Ready for review:', value['ready_for_review'])
        return 0 if value['ready_for_review'] else 2
    if args.mode == 'report':
        report(args.run_dir)
        return 0
    rows, plan = source_data(args.source_dir)
    prepared = json.loads((args.run_dir / 'preparation.json').read_text())
    reviews = json.loads((args.run_dir / 'reviews.json').read_text())
    validate_reviews(prepared, reviews)
    for key, path in (('source_manifest_sha256', args.source_dir / 'manifest.json'),
                      ('source_splits_sha256', args.source_dir / 'splits.json'), ('bank_sha256', args.clause_bank)):
        if prepared[key] != file_sha256(path):
            raise ValueError('Prepared inputs changed: ' + key)
    import mlflow
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    config = ExperimentConfig(input=Path('artifacts/training/roberta-aspect-exp1/splits/train.csv'),
                              model='FacebookAI/roberta-base', run_dir=args.run_dir, checkpoint_steps=0,
                              mlflow_tracking_uri='sqlite:///' + str(Path('artifacts/mlflow.db').resolve()),
                              mlflow_experiment='contest2-conflict-augmentation-v1')
    with RunLock(args.run_dir / '.experiment.lock'):
        _write_or_validate(args.run_dir / 'manifest.json', {
            'preparation_sha256': digest(prepared), 'reviews_sha256': digest(reviews), 'rules': RULES,
            'config': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
            'conditions': list(CONDITIONS), 'fits': 15,
        })
        _prepare_mlflow_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, use_fast=True)
        completed = 0
        for fold in plan[:1] if args.smoke_only else plan:
            examples = next(f['examples'] for f in prepared['folds'] if f['fold'] == fold['fold'])
            fit = rows_for_ids(rows, fold['fit_ids'])
            for condition in CONDITIONS:
                directory = args.run_dir / ('smoke' if args.smoke_only else f"fold-{fold['fold']}") / condition
                fit_config = replace(config, run_dir=directory, seed=42 + fold['fold'])
                if args.smoke_only:
                    fit_config = replace(fit_config, epochs=1, evaluation_interval=1)
                train_one(fit_config, condition, fit, rows_for_ids(rows, fold['selection_ids']),
                          rows_for_ids(rows, fold['heldout_ids']), examples, torch, transformers, mlflow, tokenizer)
                completed += 1
                LOGGER.info('Completed %d/%d %s fits', completed, 3 if args.smoke_only else 15,
                            'smoke' if args.smoke_only else 'full')
        if not args.smoke_only:
            report(args.run_dir)
            atomic_write_text(args.run_dir / 'completed_at.txt', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + '\n')
    return 0


def join_clauses(first: str, second: str, template: str) -> str:
    # Retain exact spans in provenance; only sentence-initial case is adjusted.
    def lower(text: str) -> str:
        return text if text.startswith('I ') else text[0].lower() + text[1:]
    first = first[0].upper() + first[1:]
    if template == 'sentences':
        return first + '. ' + second[0].upper() + second[1:] + '.'
    if template == 'however':
        return first + '; however, ' + lower(second) + '.'
    if template == 'although':
        return 'Although ' + lower(first) + ', ' + lower(second) + '.'
    return first + ', ' + template + ' ' + lower(second) + '.'


def generate(bank: list[dict], fit: list[LabeledRow], seed: int) -> list[dict]:
    ids = {r.item_id for r in fit}
    available = [c for c in bank if c['id'] in ids]
    labels = Counter((r.aspect, r.polarity) for r in fit)
    maximum = min(RULES['target_conflicts'], sum(r.polarity == 'conflict' for r in fit), 100)
    # Multiples of five make the 20% no-conjunction share exact.
    maximum -= maximum % 5
    positive = [c for c in available if c['polarity'] == 'positive']
    negative = [c for c in available if c['polarity'] == 'negative']
    pairs = [(a, b) for a in positive for b in negative if a['id'] != b['id']]
    for count in range(maximum, RULES['minimum_conflicts'] - 1, -5):
        for attempt in range(30):
            rng = random.Random(seed + attempt)
            uses: Counter = Counter()
            seen_pairs: set = set()
            seen_text: set = set()
            examples = []
            for index in range(count):
                for kind in ('conflict', 'control'):
                    target_polarity = 'conflict' if kind == 'conflict' else ('positive' if index % 2 == 0 else 'negative')
                    candidates = []
                    for a, b in pairs:
                        target_aspect = b['aspect'] if target_polarity == 'negative' else a['aspect']
                        if (a['aspect'] == b['aspect']) != (kind == 'conflict'):
                            continue
                        if not labels[target_aspect, target_polarity]:
                            continue
                        key = (a['key'], b['key'])
                        if key in seen_pairs or max(uses[a['key']], uses[b['key']]) >= RULES['max_clause_uses']:
                            continue
                        template = 'sentences' if index % 5 == 0 else ('but', 'though', 'however', 'although')[(index - 1) % 5]
                        first, second = (a, b) if (index // 2 if kind == 'control' else index) % 2 == 0 else (b, a)
                        text = join_clauses(first['span'], second['span'], template)
                        if text.lower() in seen_text:
                            continue
                        candidates.append((uses[a['key']] + uses[b['key']], rng.random(), a, b, text, target_aspect, template))
                    if not candidates:
                        break
                    _, _, a, b, text, aspect, template = min(candidates, key=lambda v: v[:2])
                    seen_pairs.add((a['key'], b['key']))
                    seen_text.add(text.lower())
                    uses[a['key']] += 1
                    uses[b['key']] += 1
                    examples.append({'key': digest([text, aspect, target_polarity])[:16], 'text': text,
                                     'aspect': aspect, 'polarity': target_polarity,
                                     'kind': kind, 'template': template, 'sources': [a, b]})
                else:
                    continue
                break
            if len(examples) == count * 2:
                return examples
    return []


def prepare(source: Path, bank_path: Path, run_dir: Path) -> dict:
    rows, plan = source_data(source)
    bank = clause_bank(bank_path, rows)
    prepared = {'rules': RULES, 'diagnostics': [list(value) for value in DIAGNOSTICS],
                'source_manifest_sha256': file_sha256(source / 'manifest.json'),
                'source_splits_sha256': file_sha256(source / 'splits.json'),
                'bank_sha256': file_sha256(bank_path), 'folds': []}
    for fold in plan:
        fit = rows_for_ids(rows, fold['fit_ids'])
        examples = generate(bank, fit, 42 + fold['fold'])
        prepared['folds'].append({'fold': fold['fold'], 'examples': examples,
                                  'counts': dict(Counter(e['polarity'] for e in examples))})
    prepared['ready_for_review'] = all(f['counts'].get('conflict', 0) >= 30 for f in prepared['folds'])
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate(run_dir / 'preparation.json', prepared)
    atomic_write_text(run_dir / 'preparation.md', '# Synthetic conflict preparation\n\n' +
                      '\n'.join(f"Fold {f['fold']}: {f['counts']}" for f in prepared['folds']) +
                      '\n\nAssistant judgments are provisional. Training requires a hash-matched review for every example.\n')
    return prepared


def validate_reviews(prepared: dict, reviews: dict) -> None:
    if not prepared['ready_for_review'] or reviews['preparation_sha256'] != digest(prepared):
        raise ValueError('Preparation is insufficient or review is stale')
    expected = {f"{f['fold']}:{e['key']}" for f in prepared['folds'] for e in f['examples']}
    if set(reviews['accepted']) != expected or reviews.get('reviewer') != 'assistant':
        raise ValueError('Every frozen example requires explicit assistant review')


def training_rows(fit: list[LabeledRow], examples: list[dict], condition: str, seed: int) -> list[LabeledRow]:
    if condition == 'reference':
        return fit
    if condition not in CONDITIONS:
        raise ValueError('Unknown condition')
    rng = random.Random(seed)
    additions = []
    for index, example in enumerate(examples):
        if condition == 'synthetic':
            additions.append(LabeledRow('synthetic-' + example['key'], example['text'], example['aspect'],
                                        example['polarity'], len(fit) + index))
        else:
            matches = [r for r in fit if (r.aspect, r.polarity) == (example['aspect'], example['polarity'])]
            if 'repetition_position' in example:
                frozen = [r for r in matches if r.position == example['repetition_position']]
                if len(frozen) != 1:
                    raise ValueError('Frozen repetition row is missing or mismatched')
                additions.append(frozen[0])
            else:
                additions.append(rng.choice(matches))
    return fit + additions


def record_predictions(rows: list[LabeledRow], logits: Any) -> list[dict]:
    values = logits.tolist()
    if len(values) != len(rows) or any(len(v) != 4 or not all(math.isfinite(x) for x in v) for v in values):
        raise ValueError('Logit annotation coverage mismatch')
    return [{'position': r.position, 'id': r.item_id, 'aspect': r.aspect, 'text': r.text,
             'gold': r.polarity, 'predicted': POLARITIES[max(range(4), key=v.__getitem__)], 'logits': v}
            for r, v in zip(rows, values)]


def metrics(records: list[dict]) -> dict:
    result = classification_metrics([POLARITIES.index(r['gold']) for r in records],
                                    [POLARITIES.index(r['predicted']) for r in records])
    negatives = [r for r in records if r['gold'] != 'conflict']
    result['conflict_false_positive_rate'] = sum(r['predicted'] == 'conflict' for r in negatives) / len(negatives) if negatives else 0.0
    return result


def train_one(config: ExperimentConfig, condition: str, fit: list[LabeledRow],
              selection: list[LabeledRow], heldout: list[LabeledRow], examples: list[dict],
              torch: Any, transformers: Any, mlflow: Any, tokenizer: Any,
              data_seed: int | None = None) -> dict:
    path = config.run_dir / 'result.json'
    if path.is_file():
        result = json.loads(path.read_text())
        if result.get('finished'):
            return result
    _seed_everything(torch, config.seed)
    train = training_rows(
        fit, examples, condition, config.seed if data_seed is None else data_seed
    )
    LOGGER.info('Starting %s with %d original and %d added rows', config.run_dir, len(fit), len(train) - len(fit))
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with mlflow.start_run(run_name=str(config.run_dir)):
        mlflow.log_params({'condition': condition, 'seed': config.seed, 'train_rows': len(train),
                           'selection': 'natural-gold-aspect-macro-f1', 'weight_source': 'original-fit'})
        training = train_stage(torch, transformers, mlflow, tokenizer, config, 'polarity', train, selection,
                               checkpoint_on_evaluation_only=True, polarity_weight_rows=fit)
    training_seconds = time.monotonic() - started
    if not math.isfinite(training['score']):
        raise RuntimeError('Non-finite checkpoint selection score')
    peak = torch.cuda.max_memory_allocated() / 1024**3
    model = transformers.AutoModelForSequenceClassification.from_pretrained(config.model, num_labels=4).float().to('cuda')
    checkpoint = torch.load(config.run_dir / 'best/polarity.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    diagnostic_rows = [LabeledRow(f'diagnostic-{i}', text, aspect, polarity, i) for i, (text, aspect, polarity, _) in enumerate(DIAGNOSTICS)]
    predictions = {}
    for name, rows in (('heldout', heldout), ('diagnostic', diagnostic_rows)):
        dataset = _attach_tokenizer(PolarityDataset(rows, tokenizer, config.max_length), tokenizer)
        logits = _predict_logits(torch, transformers, model, dataset, config.eval_batch_size, name)
        predictions[name] = record_predictions(rows, logits)
    result = {'finished': True, 'condition': condition, 'seed': config.seed, 'training': training,
              'training_seconds': training_seconds, 'peak_training_allocated_gib': peak,
              'train_rows': len(train), 'predictions': predictions,
              'metrics': {name: metrics(values) for name, values in predictions.items()}}
    atomic_write_json(path, result)
    # This fit is now recoverable from best weights and result; discard only its
    # newly generated optimizer snapshot to bound pilot disk use.
    (config.run_dir / 'checkpoints/polarity/latest.pt').unlink(missing_ok=True)
    del model, checkpoint, logits
    gc.collect()
    torch.cuda.empty_cache()
    LOGGER.info('Finished %s in %.1fs', config.run_dir, time.monotonic() - started)
    return result
