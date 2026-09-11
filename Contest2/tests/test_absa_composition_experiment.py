import unittest

from zeroshot_classifier.absa_composition_experiment import (
    _assert_full_coverage,
    _validate_manifest_values,
    oracle_aspect_predictions,
    select_candidate,
)
from zeroshot_classifier.data import LabeledRow


class AbsaCompositionExperimentTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {
            'source_of_truth_metrics': {
                'overall': {'micro': {'f1': 0.8}},
            },
            'supplemental_exact_match_accuracy': {'overall': 0.7},
        }

    def test_candidate_selection_prefers_pair_f1(self):
        weaker = {
            **self.metrics,
            'source_of_truth_metrics': {'overall': {'micro': {'f1': 0.79}}},
        }
        candidates = {
            'one_seed': {'metrics': weaker},
            'three_seed': {'metrics': self.metrics},
        }

        self.assertEqual(select_candidate(candidates), 'three_seed')

    def test_candidate_tie_break_prefers_exact_set_then_one_seed(self):
        candidates = {
            'one_seed': {'metrics': self.metrics},
            'three_seed': {'metrics': self.metrics},
        }

        self.assertEqual(select_candidate(candidates), 'one_seed')

    def test_oracle_predictions_use_every_gold_aspect(self):
        rows = [
            LabeledRow('1', 'Review', 'food', 'positive', 0),
            LabeledRow('1', 'Review', 'service', 'negative', 1),
        ]
        polarities = [[0, 2, 1, 3, 0]]

        predictions = oracle_aspect_predictions(rows, polarities)

        self.assertEqual(
            predictions,
            [('1', 'food', 'positive'), ('1', 'service', 'negative')],
        )

    def test_full_coverage_rejects_missing_review(self):
        rows = [
            LabeledRow('1', 'First', 'food', 'positive', 0),
            LabeledRow('2', 'Second', 'service', 'negative', 1),
        ]

        with self.assertRaisesRegex(RuntimeError, 'missing=1'):
            _assert_full_coverage(rows, [('1', 'food', 'positive')])

    def test_manifest_validation_rejects_split_mismatch(self):
        from zeroshot_classifier.absa_composition_experiment import CompositionConfig

        config = CompositionConfig()
        manifest = {
            'model': config.model,
            'max_length': config.max_length,
            'input_sha256': 'input',
            'split_sha256': {'eval': 'wrong'},
        }

        with self.assertRaisesRegex(RuntimeError, 'split_sha256'):
            _validate_manifest_values(
                manifest, config, 'input', {'eval': 'expected'}, 'Test'
            )


if __name__ == '__main__':
    unittest.main()
