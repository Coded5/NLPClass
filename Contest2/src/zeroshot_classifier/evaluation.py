from __future__ import annotations

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .data import LabeledRow, read_labeled_csv
from .io_utils import atomic_write_json, atomic_write_text
from .labels import ASPECTS, POLARITIES


class EvaluationError(RuntimeError):
    pass


OFFICIAL_EVALUATOR = Path(__file__).resolve().parents[2] / 'scripts' / 'evaluate.py'


def evaluate(
    gold_path: Path,
    prediction_path: Path,
    evaluator_path: Path,
    metrics_path: Path,
    report_path: Path,
) -> dict[str, object]:
    gold_rows = read_labeled_csv(gold_path)
    predictions = read_predictions(prediction_path)
    report = run_official_evaluator(evaluator_path, gold_path, prediction_path)
    print(report, end='' if report.endswith('\n') else '\n')
    atomic_write_text(report_path, report)

    metrics = calculate_metrics(gold_rows, predictions)
    metrics['official_evaluator'] = {
        'script': str(evaluator_path),
        'report': str(report_path),
        'source_of_truth': evaluator_path.resolve() == OFFICIAL_EVALUATOR.resolve(),
    }
    atomic_write_json(metrics_path, metrics)
    return metrics


def read_predictions(path: Path) -> list[tuple[str, str, str]]:
    if not path.is_file():
        raise EvaluationError(f'Prediction file not found: {path}')
    predictions: list[tuple[str, str, str]] = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = {'id', 'aspectCategory', 'polarity'} - set(reader.fieldnames or ())
        if missing:
            raise EvaluationError(f'Missing prediction columns: {", ".join(sorted(missing))}')
        for row_number, row in enumerate(reader, start=2):
            item_id = row['id'].strip()
            aspect = row['aspectCategory'].strip()
            polarity = row['polarity'].strip()
            if not item_id or not aspect or not polarity:
                raise EvaluationError(f'Empty prediction value at CSV row {row_number}')
            predictions.append((item_id, aspect, polarity))
    return predictions


def run_official_evaluator(
    evaluator_path: Path,
    gold_path: Path,
    prediction_path: Path,
) -> str:
    if not evaluator_path.is_file():
        raise EvaluationError(f'Official evaluator not found: {evaluator_path}')
    try:
        result = subprocess.run(
            [sys.executable, str(evaluator_path), str(gold_path), str(prediction_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        raise EvaluationError('Official evaluator timed out after 120 seconds') from error
    output = result.stdout
    if result.stderr:
        output += result.stderr
    if result.returncode != 0:
        raise EvaluationError(
            f'Official evaluator failed with exit code {result.returncode}:\n{output.rstrip()}'
        )
    evaluator_errors = ('NOT FOUND:', 'CANNOT READ:', 'INCORRECT COLUMN NAME')
    if any(marker in output for marker in evaluator_errors):
        raise EvaluationError(f'Official evaluator rejected the files:\n{output.rstrip()}')
    return output


def calculate_metrics(
    gold_rows: Iterable[LabeledRow],
    predictions: Iterable[tuple[str, str, str]],
) -> dict[str, object]:
    gold_rows = list(gold_rows)
    predictions = list(predictions)
    predicted_ids = {item_id for item_id, _, _ in predictions}

    gold_aspect = {
        (row.item_id, row.aspect) for row in gold_rows if row.item_id in predicted_ids
    }
    gold_sentiment = {
        (row.item_id, row.polarity) for row in gold_rows if row.item_id in predicted_ids
    }
    gold_overall = {
        (row.item_id, row.aspect, row.polarity)
        for row in gold_rows
        if row.item_id in predicted_ids
    }
    pred_aspect = {(item_id, aspect) for item_id, aspect, _ in predictions}
    pred_sentiment = {(item_id, polarity) for item_id, _, polarity in predictions}
    pred_overall = set(predictions)

    aspect_report = _target_report(gold_aspect, pred_aspect, ASPECTS)
    sentiment_report = _target_report(gold_sentiment, pred_sentiment, POLARITIES)
    overall_micro = _prf(gold_overall, pred_overall)

    gold_by_id = _sets_by_id(
        (row.item_id, row.aspect, row.polarity) for row in gold_rows
    )
    pred_by_id = _sets_by_id(predictions)
    all_ids = set(gold_by_id) | set(pred_by_id)
    aspect_accuracy = _exact_set_accuracy(gold_by_id, pred_by_id, all_ids, field=0)
    polarity_accuracy = _exact_set_accuracy(gold_by_id, pred_by_id, all_ids, field=1)
    overall_accuracy = _exact_set_accuracy(gold_by_id, pred_by_id, all_ids, field=None)

    expected_ids = set(gold_by_id)
    covered_ids = expected_ids & set(pred_by_id)
    return {
        'source_of_truth_metrics': {
            'aspect': aspect_report,
            'polarity': sentiment_report,
            'overall': {'micro': overall_micro},
        },
        'supplemental_exact_match_accuracy': {
            'aspect': aspect_accuracy,
            'polarity': polarity_accuracy,
            'overall': overall_accuracy,
            'definition': (
                'Per-ID exact set match. Overall is correct only when every predicted '
                '(aspectCategory, polarity) pair exactly matches the complete gold set.'
            ),
        },
        'coverage': {
            'expected_ids': len(expected_ids),
            'predicted_ids': len(set(pred_by_id)),
            'covered_expected_ids': len(covered_ids),
            'missing_ids': len(expected_ids - set(pred_by_id)),
            'unexpected_ids': len(set(pred_by_id) - expected_ids),
            'ratio': len(covered_ids) / len(expected_ids) if expected_ids else 0.0,
        },
    }


def _target_report(
    gold: set[tuple[str, str]],
    predicted: set[tuple[str, str]],
    classes: tuple[str, ...],
) -> dict[str, object]:
    class_reports: dict[str, dict[str, float | int]] = {}
    for class_name in classes:
        class_gold = {value for value in gold if value[1] == class_name}
        class_predicted = {value for value in predicted if value[1] == class_name}
        report = _prf(class_gold, class_predicted)
        report['support'] = len(class_gold)
        class_reports[class_name] = report

    macro = {
        metric: sum(float(report[metric]) for report in class_reports.values()) / len(classes)
        for metric in ('precision', 'recall', 'f1')
    }
    macro['support'] = sum(int(report['support']) for report in class_reports.values())
    micro = _prf(gold, predicted)
    micro['support'] = len(gold)
    return {'classes': class_reports, 'macro': macro, 'micro': micro}


def _prf(gold: set[tuple], predicted: set[tuple]) -> dict[str, float]:
    correct = len(gold & predicted)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def _sets_by_id(
    values: Iterable[tuple[str, str, str]],
) -> dict[str, set[tuple[str, str]]]:
    grouped: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for item_id, aspect, polarity in values:
        grouped[item_id].add((aspect, polarity))
    return grouped


def _exact_set_accuracy(
    gold: dict[str, set[tuple[str, str]]],
    predicted: dict[str, set[tuple[str, str]]],
    item_ids: set[str],
    field: int | None,
) -> float:
    if not item_ids:
        return 0.0

    def select(values: set[tuple[str, str]]) -> set[str] | set[tuple[str, str]]:
        if field is None:
            return values
        return {value[field] for value in values}

    correct = sum(
        select(gold.get(item_id, set())) == select(predicted.get(item_id, set()))
        for item_id in item_ids
    )
    return correct / len(item_ids)
