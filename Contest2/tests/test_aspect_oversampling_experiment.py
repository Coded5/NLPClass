from __future__ import annotations

import unittest

import torch

from zeroshot_classifier.absa_training import _training_order
from zeroshot_classifier.aspect_oversampling_experiment import (
    eligible,
    grouped_aspect_folds,
    is_target_review,
    sampling_weights,
    subgroup_metrics,
)
from zeroshot_classifier.data import LabeledRow
from zeroshot_classifier.labels import ASPECTS


class AspectOversamplingExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            LabeledRow('1', 'overall and food', 'food', 'positive', 0),
            LabeledRow('1', 'overall and food', 'anecdotes/miscellaneous', 'positive', 1),
            LabeledRow('2', 'food only', 'food', 'positive', 2),
            LabeledRow('3', 'service only', 'service', 'negative', 3),
            LabeledRow('4', 'misc only', 'anecdotes/miscellaneous', 'neutral', 4),
            LabeledRow('5', 'price ambience', 'price', 'positive', 5),
            LabeledRow('5', 'price ambience', 'ambience', 'positive', 6),
        ]

    def test_target_requires_miscellaneous_and_multiple_aspects(self) -> None:
        misc = ASPECTS.index('anecdotes/miscellaneous')
        food = ASPECTS.index('food')

        self.assertTrue(is_target_review((misc, food)))
        self.assertFalse(is_target_review((misc,)))
        self.assertFalse(is_target_review((food,)))

    def test_sampling_weights_only_raise_target_reviews(self) -> None:
        self.assertIsNone(sampling_weights(self.rows, 'reference'))
        self.assertEqual(
            sampling_weights(self.rows, 'sampling-control'),
            [1.0, 1.0, 1.0, 1.0, 1.0],
        )
        self.assertEqual(
            sampling_weights(self.rows, 'targeted-3x'),
            [3.0, 1.0, 1.0, 1.0, 1.0],
        )

    def test_grouped_folds_are_deterministic_disjoint_and_complete(self) -> None:
        first = grouped_aspect_folds(self.rows, 2, 42)
        second = grouped_aspect_folds(self.rows, 2, 42)

        self.assertEqual(first, second)
        self.assertFalse(first[0] & first[1])
        self.assertEqual(set().union(*first), {'1', '2', '3', '4', '5'})

    def test_subgroup_recall_counts_only_multi_miscellaneous(self) -> None:
        metrics = subgroup_metrics(
            self.rows,
            [('1', 'food'), ('1', 'anecdotes/miscellaneous'),
             ('4', 'anecdotes/miscellaneous')],
        )

        self.assertEqual(metrics, {'support': 1, 'correct': 1, 'recall': 1.0})

    def test_promotion_gate_enforces_every_guardrail(self) -> None:
        reference = self._metrics(0.88, 0.83, 0.30)
        candidate = self._metrics(0.876, 0.84, 0.50)

        self.assertTrue(eligible(candidate, reference, 0.005, 4, 5))
        candidate['aspect']['micro']['f1'] = 0.87
        self.assertFalse(eligible(candidate, reference, 0.005, 4, 5))

    def test_weighted_training_order_is_seed_reproducible(self) -> None:
        first = _training_order(
            torch, 4, torch.Generator().manual_seed(9), [4.0, 1.0, 1.0, 1.0]
        )
        second = _training_order(
            torch, 4, torch.Generator().manual_seed(9), [4.0, 1.0, 1.0, 1.0]
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    @staticmethod
    def _metrics(micro: float, misc: float, subgroup: float) -> dict[str, object]:
        return {
            'aspect': {
                'micro': {'f1': micro},
                'classes': {'anecdotes/miscellaneous': {'f1': misc}},
            },
            'multi_misc': {'recall': subgroup},
        }


if __name__ == '__main__':
    unittest.main()
