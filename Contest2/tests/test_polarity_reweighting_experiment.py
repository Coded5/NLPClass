import unittest

from zeroshot_classifier.polarity_reweighting_experiment import (
    config_from_args,
    select_final_winner,
    select_screening_candidate,
)


class PolarityReweightingExperimentTests(unittest.TestCase):
    def test_cli_parses_pair_tolerance_and_paths(self):
        config = config_from_args([
            '--pair-f1-tolerance', '0.01',
            '--run-dir', 'artifacts/example',
        ])

        self.assertEqual(config.pair_f1_tolerance, 0.01)
        self.assertEqual(str(config.run_dir), 'artifacts/example')

    def test_screening_prefers_minority_f1_among_eligible_candidates(self):
        candidates = {
            'small': {
                'eligible': True, 'validation_minority_f1': 0.51,
                'validation_pair_micro_f1': 0.76, 'multipliers': [1.25, 1.5],
            },
            'large': {
                'eligible': True, 'validation_minority_f1': 0.55,
                'validation_pair_micro_f1': 0.755, 'multipliers': [2.0, 3.0],
            },
            'ineligible': {
                'eligible': False, 'validation_minority_f1': 0.80,
                'validation_pair_micro_f1': 0.70, 'multipliers': [2.0, 3.0],
            },
        }

        self.assertEqual(select_screening_candidate(candidates), 'large')

    def test_screening_tie_prefers_smaller_multipliers(self):
        candidates = {
            'small': {
                'eligible': True, 'validation_minority_f1': 0.55,
                'validation_pair_micro_f1': 0.76, 'multipliers': [1.25, 1.5],
            },
            'large': {
                'eligible': True, 'validation_minority_f1': 0.55,
                'validation_pair_micro_f1': 0.76, 'multipliers': [2.0, 3.0],
            },
        }

        self.assertEqual(select_screening_candidate(candidates), 'small')

    def test_final_winner_requires_minority_gain_and_pair_constraint(self):
        baseline = {'pair_micro_f1': 0.76, 'minority_f1': 0.50}

        self.assertEqual(
            select_final_winner(
                baseline, {'pair_micro_f1': 0.755, 'minority_f1': 0.51}, 0.005
            ),
            'reweighted',
        )
        self.assertEqual(
            select_final_winner(
                baseline, {'pair_micro_f1': 0.754, 'minority_f1': 0.60}, 0.005
            ),
            'baseline',
        )
        self.assertEqual(
            select_final_winner(
                baseline, {'pair_micro_f1': 0.77, 'minority_f1': 0.50}, 0.005
            ),
            'baseline',
        )


if __name__ == '__main__':
    unittest.main()
