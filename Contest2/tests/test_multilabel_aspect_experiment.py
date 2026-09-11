import importlib.util
import tempfile
import unittest
from pathlib import Path

from zeroshot_classifier.data import LabeledRow
from zeroshot_classifier.multilabel_aspect_experiment import (
    _export_compact_checkpoint,
    aspect_predictions,
    calculate_aspect_metrics,
    config_from_args,
    mean_probabilities,
    paired_bootstrap_delta,
)


class MultilabelAspectExperimentTests(unittest.TestCase):
    def setUp(self):
        self.gold = [
            LabeledRow('1', 'Same review', 'food', 'positive', 0),
            LabeledRow('1', 'Same review', 'service', 'negative', 1),
        ]

    def test_duplicate_text_with_different_aspects_can_score_perfectly(self):
        metrics = calculate_aspect_metrics(
            self.gold,
            [('1', 'food'), ('1', 'service')],
        )

        self.assertEqual(metrics['aspect']['micro']['f1'], 1.0)
        self.assertEqual(metrics['exact_set_accuracy'], 1.0)
        self.assertEqual(
            metrics['subsets']['multi_aspect']['exact_set_accuracy'], 1.0
        )

    def test_exact_duplicate_annotation_does_not_change_metrics(self):
        duplicate = LabeledRow('1', 'Same review', 'food', 'positive', 2)

        metrics = calculate_aspect_metrics(
            self.gold + [duplicate],
            [('1', 'food'), ('1', 'service'), ('1', 'service')],
        )

        self.assertEqual(metrics['aspect']['micro']['support'], 2)
        self.assertEqual(metrics['aspect']['micro']['f1'], 1.0)

    def test_missing_one_aspect_receives_partial_f1_and_fails_exact_match(self):
        metrics = calculate_aspect_metrics(self.gold, [('1', 'food')])

        self.assertAlmostEqual(metrics['aspect']['micro']['f1'], 2 / 3)
        self.assertEqual(metrics['exact_set_accuracy'], 0.0)

    def test_identical_text_under_different_ids_is_keyed_by_id(self):
        gold = [
            LabeledRow('1', 'Identical text', 'food', 'positive', 0),
            LabeledRow('2', 'Identical text', 'service', 'negative', 1),
        ]

        metrics = calculate_aspect_metrics(
            gold, [('1', 'food'), ('2', 'service')]
        )

        self.assertEqual(metrics['aspect']['micro']['f1'], 1.0)
        self.assertEqual(metrics['exact_set_accuracy'], 1.0)

    def test_mean_probabilities_averages_seed_outputs(self):
        averaged = mean_probabilities([
            [[0.2, 0.8], [0.4, 0.6]],
            [[0.6, 0.4], [0.8, 0.2]],
        ])

        for actual_row, expected_row in zip(
            averaged, [[0.4, 0.6], [0.6, 0.4]]
        ):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected)

    def test_aspect_predictions_can_emit_multiple_labels(self):
        predictions = aspect_predictions(
            self.gold,
            [[0.8, 0.1, 0.7, 0.1, 0.1]],
            [0.5] * 5,
        )

        self.assertEqual(predictions, [('1', 'food'), ('1', 'service')])

    def test_bootstrap_delta_is_deterministic_and_favors_better_system(self):
        contender = [('1', 'food'), ('1', 'service')]
        baseline = [('1', 'food')]

        first = paired_bootstrap_delta(
            self.gold, contender, baseline, samples=100, seed=7
        )
        second = paired_bootstrap_delta(
            self.gold, contender, baseline, samples=100, seed=7
        )

        self.assertEqual(first, second)
        self.assertGreater(first['mean'], 0.0)

    def test_cli_accepts_three_explicit_seeds(self):
        config = config_from_args(['--seeds', '17', '42', '73'])

        self.assertEqual(config.seeds, (17, 42, 73))

    @unittest.skipIf(
        importlib.util.find_spec('torch') is None,
        'compact checkpoint test requires torch',
    )
    def test_compact_checkpoint_exports_floating_weights_as_float16(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.pt'
            destination = root / 'best' / 'aspect.pt'
            torch.save({
                'model': {
                    'weight': torch.tensor([1.0], dtype=torch.float32),
                    'counter': torch.tensor([1], dtype=torch.int64),
                },
                'epoch': 3,
            }, source)

            _export_compact_checkpoint(torch, source, destination)
            exported = torch.load(destination, map_location='cpu', weights_only=False)

        self.assertEqual(exported['model']['weight'].dtype, torch.float16)
        self.assertEqual(exported['model']['counter'].dtype, torch.int64)
        self.assertEqual(exported['storage_dtype'], 'float16')

    @unittest.skipIf(
        importlib.util.find_spec('pandas') is None,
        'official evaluator requires pandas',
    )
    def test_aspect_micro_metrics_match_official_evaluator(self):
        evaluator_path = Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate.py'
        spec = importlib.util.spec_from_file_location('official_evaluate', evaluator_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gold_path = root / 'gold.csv'
            prediction_path = root / 'predictions.csv'
            gold_path.write_text(
                'id,text,aspectCategory,polarity\n'
                '1,Same review,food,positive\n'
                '1,Same review,service,negative\n',
                encoding='utf-8',
            )
            prediction_path.write_text(
                'id,aspectCategory,polarity\n'
                '1,food,positive\n'
                '1,service,negative\n',
                encoding='utf-8',
            )
            evaluator = module.EvaluateModel(gold_path, prediction_path)
            evaluator.check_files()
            evaluator.make_tuple_set()

            official = evaluator.micro_PRF('aspect')
            actual = calculate_aspect_metrics(
                self.gold, [('1', 'food'), ('1', 'service')]
            )['aspect']['micro']

        self.assertEqual(actual['precision'], official[0])
        self.assertEqual(actual['recall'], official[1])
        self.assertEqual(actual['f1'], official[2])


if __name__ == '__main__':
    unittest.main()
