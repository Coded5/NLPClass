import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from zeroshot_classifier.data import LabeledRow
from zeroshot_classifier.evaluation import (
    OFFICIAL_EVALUATOR,
    calculate_metrics,
    evaluate,
    run_official_evaluator,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.gold = [
            LabeledRow('1', 'Good food', 'food', 'positive', 0),
            LabeledRow('2', 'Mixed review', 'service', 'negative', 1),
            LabeledRow('2', 'Mixed review', 'ambience', 'neutral', 2),
        ]
        self.predictions = [
            ('1', 'food', 'positive'),
            ('2', 'service', 'positive'),
        ]

    def test_metrics_follow_official_set_semantics_and_add_exact_accuracy(self):
        metrics = calculate_metrics(self.gold, self.predictions)

        official = metrics['source_of_truth_metrics']
        self.assertAlmostEqual(official['aspect']['micro']['precision'], 1.0)
        self.assertAlmostEqual(official['aspect']['micro']['recall'], 2 / 3)
        self.assertAlmostEqual(official['polarity']['micro']['precision'], 0.5)
        self.assertAlmostEqual(official['overall']['micro']['precision'], 0.5)
        accuracy = metrics['supplemental_exact_match_accuracy']
        self.assertEqual(accuracy['aspect'], 0.5)
        self.assertEqual(accuracy['polarity'], 0.5)
        self.assertEqual(accuracy['overall'], 0.5)

    def test_evaluate_executes_supplied_official_evaluator(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            gold_path = root / 'gold.csv'
            pred_path = root / 'pred.csv'
            evaluator_path = root / 'evaluate.py'
            metrics_path = root / 'metrics.json'
            report_path = root / 'report.txt'
            self._write_gold(gold_path)
            self._write_predictions(pred_path)
            evaluator_path.write_text("print('OFFICIAL REPORT')\n", encoding='utf-8')

            metrics = evaluate(
                gold_path,
                pred_path,
                evaluator_path,
                metrics_path,
                report_path,
            )

            self.assertTrue(metrics_path.is_file())
            self.assertEqual(report_path.read_text(encoding='utf-8'), 'OFFICIAL REPORT\n')
            self.assertFalse(metrics['official_evaluator']['source_of_truth'])

    @unittest.skipIf(importlib.util.find_spec('pandas') is None, 'official evaluator requires pandas')
    def test_actual_official_evaluator_accepts_generated_files(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            gold_path = root / 'gold.csv'
            pred_path = root / 'pred.csv'
            self._write_gold(gold_path)
            self._write_predictions(pred_path)

            report = run_official_evaluator(OFFICIAL_EVALUATOR, gold_path, pred_path)

            self.assertIn('=== CLASSIFICATION : ASPECT ===', report)
            self.assertIn('=== CLASSIFICATION : SENTIMENT ===', report)
            self.assertIn('=== CLASSIFICATION : OVERALL ===', report)

    def _write_gold(self, path):
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('id', 'text', 'aspectCategory', 'polarity'))
            for row in self.gold:
                writer.writerow((row.item_id, row.text, row.aspect, row.polarity))

    def _write_predictions(self, path):
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('id', 'aspectCategory', 'polarity'))
            writer.writerows(self.predictions)


if __name__ == '__main__':
    unittest.main()
