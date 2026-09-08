#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any


repository_root = Path(__file__).resolve().parents[1]
source_root = repository_root / 'src'
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from zeroshot_classifier.data import LabeledRow, read_labeled_csv
from zeroshot_classifier.evaluation import evaluate
from zeroshot_classifier.io_utils import atomic_write_json, atomic_write_text
from zeroshot_classifier.labels import ASPECTS, POLARITIES
from zeroshot_classifier.roberta_training import TrainingConfig, _predict, _write_labeled_csv, _write_predictions


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Combine the best separately trained aspect and polarity checkpoints.'
    )
    parser.add_argument('--training-root', type=Path, default=Path('artifacts/training'))
    parser.add_argument(
        '--input', type=Path, nargs='+', default=[Path('data/contest2_train.csv')]
    )
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/ensemble/full-dataset'))
    parser.add_argument('--scope-label', default='complete labeled training dataset (in-sample)')
    parser.add_argument('--limitation')
    parser.add_argument('--evaluator', type=Path, default=Path('scripts/evaluate.py'))
    parser.add_argument('--eval-batch-size', type=int, default=32)
    return parser


def select_best_run(training_root: Path, task: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    candidates = []
    for run_dir in sorted(training_root.iterdir() if training_root.is_dir() else ()):
        manifest_path = run_dir / 'manifest.json'
        state_path = run_dir / 'state.json'
        checkpoint_path = run_dir / 'best' / f'{task}.pt'
        if not (manifest_path.is_file() and state_path.is_file() and checkpoint_path.is_file()):
            continue
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if manifest.get('task') != task or state.get('best_epoch') is None:
            continue
        candidates.append((float(state['best_score']), run_dir, manifest, state))
    if not candidates:
        raise RuntimeError(f'No completed {task} run with a best checkpoint found in {training_root}')
    _, run_dir, manifest, state = max(candidates, key=lambda candidate: candidate[0])
    return run_dir, manifest, state


def predict_task(
    task: str,
    run_dir: Path,
    manifest: dict[str, Any],
    rows: list[LabeledRow],
    eval_batch_size: int,
) -> list[int]:
    import torch
    import transformers

    labels = ASPECTS if task == 'aspect' else POLARITIES
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    model_name = str(manifest['model'])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), label2id=label2id, id2label=id2label
    )
    checkpoint_path = run_dir / 'best' / f'{task}.pt'
    LOGGER.info('Loading %s checkpoint: %s', task, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    config = TrainingConfig(
        model=model_name,
        max_length=int(manifest['max_length']),
        eval_batch_size=eval_batch_size,
    )
    predictions = _predict(
        torch,
        transformers,
        model,
        tokenizer,
        rows,
        config,
        torch.device('cpu'),
        f'Full dataset inference ({task})',
    )
    del model, tokenizer
    gc.collect()
    return predictions


def markdown_report(
    metrics: dict[str, Any],
    selections: dict[str, dict[str, Any]],
    row_count: int,
    item_count: int,
    scope_label: str,
    limitation: str,
) -> str:
    source = metrics['source_of_truth_metrics']
    exact = metrics['supplemental_exact_match_accuracy']
    lines = [
        '# RoBERTa ensemble evaluation',
        '',
        '## Scope',
        '',
        f'- Evaluated rows: {row_count}',
        f'- Unique review IDs: {item_count}',
        f'- Dataset: {scope_label}',
        '- Ensemble: aspect prediction from the aspect model plus polarity prediction from the polarity model',
        '',
        limitation,
        '',
        '## Selected checkpoints',
        '',
        '| Task | Run | Best epoch | Validation micro-F1 |',
        '|---|---|---:|---:|',
    ]
    for task in ('aspect', 'polarity'):
        selected = selections[task]
        lines.append(
            f"| {task} | `{selected['run_dir']}` | {selected['best_epoch']} | "
            f"{selected['best_score']:.6f} |"
        )
    lines.extend([
        '',
        '## Evaluation results',
        '',
        '| Target | Precision | Recall | F1 | Exact-set accuracy |',
        '|---|---:|---:|---:|---:|',
    ])
    for task in ('aspect', 'polarity'):
        micro = source[task]['micro']
        lines.append(
            f"| {task} | {micro['precision']:.6f} | {micro['recall']:.6f} | "
            f"{micro['f1']:.6f} | {exact[task]:.6f} |"
        )
    overall = source['overall']['micro']
    lines.append(
        f"| overall pair | {overall['precision']:.6f} | {overall['recall']:.6f} | "
        f"{overall['f1']:.6f} | {exact['overall']:.6f} |"
    )
    lines.extend([
        '',
        '## Output files',
        '',
        '- `predictions.csv`: combined row-level predictions',
        '- `gold.csv`: evaluation copy of the labeled input',
        '- `metrics.json`: structured metrics',
        '- `official_evaluation.txt`: output from the supplied evaluator',
        '- `selection.json`: checkpoint selection provenance',
        '',
    ])
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    args = build_parser().parse_args(argv)
    if args.eval_batch_size < 1:
        raise ValueError('eval-batch-size must be at least 1')
    rows = [row for input_path in args.input for row in read_labeled_csv(input_path)]
    selections: dict[str, dict[str, Any]] = {}
    predictions: dict[str, list[int]] = {}
    for task in ('aspect', 'polarity'):
        run_dir, manifest, state = select_best_run(args.training_root, task)
        selections[task] = {
            'run_dir': str(run_dir),
            'checkpoint': str(run_dir / 'best' / f'{task}.pt'),
            'best_epoch': int(state['best_epoch']),
            'best_score': float(state['best_score']),
        }
        LOGGER.info(
            'Selected %s run %s (epoch=%d validation_micro_f1=%.6f)',
            task, run_dir, state['best_epoch'], state['best_score'],
        )
        predictions[task] = predict_task(
            task, run_dir, manifest, rows, args.eval_batch_size
        )

    records = [
        (row.item_id, ASPECTS[predictions['aspect'][index]], POLARITIES[predictions['polarity'][index]])
        for index, row in enumerate(rows)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gold_path = args.output_dir / 'gold.csv'
    prediction_path = args.output_dir / 'predictions.csv'
    _write_labeled_csv(gold_path, rows)
    _write_predictions(prediction_path, records)
    atomic_write_json(args.output_dir / 'selection.json', selections)
    metrics = evaluate(
        gold_path,
        prediction_path,
        args.evaluator,
        args.output_dir / 'metrics.json',
        args.output_dir / 'official_evaluation.txt',
    )
    report = markdown_report(
        metrics,
        selections,
        len(rows),
        len({row.item_id for row in rows}),
        args.scope_label,
        args.limitation or (
            'This is a training-set diagnostic, not an unbiased generalization estimate. '
            'The models are single-label classifiers; repeated text for multi-aspect reviews '
            'therefore receives the same prediction and cannot recover every distinct gold pair.'
            if len(args.input) == 1 and args.input[0] == Path('data/contest2_train.csv')
            else 'The models are single-label classifiers; repeated text for multi-aspect reviews '
            'receives the same prediction and cannot recover every distinct gold pair.'
        ),
    )
    atomic_write_text(args.output_dir / 'report.md', report)
    LOGGER.info('Wrote ensemble report to %s', args.output_dir / 'report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
