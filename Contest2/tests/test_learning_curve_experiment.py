from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from zeroshot_classifier.data import LabeledRow
from zeroshot_classifier.evaluation import calculate_metrics
from zeroshot_classifier.learning_curve_experiment import (
    FRACTIONS,
    LearningCurveConfig,
    classification_metrics,
    curve_diagnosis,
    generate_report,
    gold_aspect_polarity_metrics,
    grouped_label_folds,
    learning_curve_splits,
    rows_for_ids,
    serialized_split_plan,
    validate_split_plan,
)
from zeroshot_classifier.multilabel_aspect_experiment import calculate_aspect_metrics


class LearningCurveSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        aspects = ('food', 'price', 'service', 'ambience', 'anecdotes/miscellaneous')
        polarities = ('positive', 'negative', 'neutral', 'conflict')
        self.rows = []
        position = 0
        for index in range(80):
            item_id = str(index)
            self.rows.append(LabeledRow(
                item_id, f'text {index}', aspects[index % len(aspects)],
                polarities[index % len(polarities)], position,
            ))
            position += 1
            if index % 5 == 0:
                self.rows.append(LabeledRow(
                    item_id, f'text {index}', aspects[(index + 2) % len(aspects)],
                    polarities[(index + 1) % len(polarities)], position,
                ))
                position += 1

    def test_grouped_folds_are_deterministic_disjoint_and_complete(self) -> None:
        first = grouped_label_folds(self.rows, 5, 42)
        second = grouped_label_folds(self.rows, 5, 42)
        self.assertEqual(first, second)
        self.assertEqual(set().union(*first), {str(index) for index in range(80)})
        for index, fold in enumerate(first):
            self.assertFalse(fold & set().union(*first[:index], *first[index + 1:]))

    def test_learning_subsets_are_nested_and_retain_complete_reviews(self) -> None:
        plan = learning_curve_splits(self.rows, 5, 10, 42)
        validate_split_plan(self.rows, plan)
        for fold in plan:
            previous = set()
            for fraction in FRACTIONS:
                current = fold['subsets'][fraction]
                self.assertTrue(previous <= current)
                selected = rows_for_ids(self.rows, current)
                self.assertEqual(
                    len(selected), sum(row.item_id in current for row in self.rows),
                )
                previous = current

    def test_split_serialization_is_stable(self) -> None:
        plan = learning_curve_splits(self.rows, 5, 10, 42)
        first = serialized_split_plan(plan)
        second = serialized_split_plan(learning_curve_splits(self.rows, 5, 10, 42))
        self.assertEqual(first, second)

    def test_nested_subsets_preserve_rare_label_proportions(self) -> None:
        plan = learning_curve_splits(self.rows, 5, 10, 42)
        for fold in plan:
            fit_rows = rows_for_ids(self.rows, fold['fit_ids'])
            expected = Counter(row.polarity for row in fit_rows)
            expected_conflict = expected['conflict'] / len(fit_rows)
            for fraction in FRACTIONS:
                selected = rows_for_ids(self.rows, fold['subsets'][fraction])
                actual = Counter(row.polarity for row in selected)
                self.assertLess(
                    abs(actual['conflict'] / len(selected) - expected_conflict),
                    0.10,
                )


class LearningCurveMetricTests(unittest.TestCase):
    def test_gold_aspect_polarity_selects_the_correct_candidate(self) -> None:
        rows = [
            LabeledRow('1', 'text', 'food', 'positive', 0),
            LabeledRow('1', 'text', 'service', 'negative', 1),
        ]
        metrics = gold_aspect_polarity_metrics(rows, [[0, 2, 1, 3, 0]])
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['support'], 2)

    def test_classification_metrics_reject_bad_shapes(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics([0], [])
        with self.assertRaises(ValueError):
            classification_metrics([0], [9])

    def test_curve_decision_rules(self) -> None:
        self.assertEqual(
            curve_diagnosis(0.02, {'lower_95': 0.001, 'upper_95': 0.03}),
            'data-limited',
        )
        self.assertEqual(
            curve_diagnosis(0.002, {'lower_95': -0.004, 'upper_95': 0.009}),
            'plateaued',
        )
        self.assertEqual(
            curve_diagnosis(0.008, {'lower_95': -0.002, 'upper_95': 0.02}),
            'inconclusive',
        )


class LearningCurveReportTests(unittest.TestCase):
    def test_completed_mock_results_generate_all_lightweight_outputs(self) -> None:
        rows = [
            LabeledRow('1', 'good food', 'food', 'positive', 0),
            LabeledRow('2', 'bad service', 'service', 'negative', 1),
            LabeledRow('3', 'neutral room', 'ambience', 'neutral', 2),
            LabeledRow('4', 'mixed price', 'price', 'conflict', 3),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / 'train.csv'
            input_path.write_text(
                'id,text,aspectCategory,polarity\n'
                '1,good food,food,positive\n'
                '2,bad service,service,negative\n'
                '3,neutral room,ambience,neutral\n'
                '4,mixed price,price,conflict\n',
                encoding='utf-8',
            )
            run_dir = root / 'run'
            fold_ids = ({'1', '2'}, {'3', '4'})
            for fold, ids in enumerate(fold_ids, 1):
                heldout = rows_for_ids(rows, ids)
                pairs = [[row.item_id, row.aspect, row.polarity] for row in heldout]
                aspects = [[row.item_id, row.aspect] for row in heldout]
                polarity = [[row.item_id, row.aspect, row.polarity, row.polarity]
                            for row in heldout]
                for fraction in FRACTIONS:
                    pair_tuples = [tuple(value) for value in pairs]
                    aspect_tuples = [tuple(value) for value in aspects]
                    result = {
                        'finished': True, 'fold': fold, 'fraction': fraction,
                        'metrics': {
                            'pair': calculate_metrics(heldout, pair_tuples),
                            'aspect': calculate_aspect_metrics(heldout, aspect_tuples),
                            'polarity': {'mixed': {'macro_f1': 1.0}},
                        },
                        'predictions': {
                            'pair': pairs, 'aspect': aspects,
                            'polarity': {
                                'roberta': polarity, 'deberta': polarity, 'mixed': polarity,
                            },
                        },
                    }
                    output = run_dir / f'fold-{fold}' / f'fraction-{fraction}'
                    output.mkdir(parents=True)
                    (output / 'result.json').write_text(json.dumps(result), encoding='utf-8')
            config = LearningCurveConfig(
                input=input_path, run_dir=run_dir, folds=2,
                selection_folds=2, bootstrap_samples=20,
            )
            summary = generate_report(config)
            self.assertEqual(summary['primary_diagnosis'], 'plateaued')
            for name in ('summary.json', 'report.md', 'learning_curve.csv', 'learning_curve.svg'):
                self.assertTrue((run_dir / name).is_file())


if __name__ == '__main__':
    unittest.main()
