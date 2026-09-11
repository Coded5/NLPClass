from __future__ import annotations
import unittest
import torch
from zeroshot_classifier.data import LabeledRow
from zeroshot_classifier.polarity_cv_experiment import class_weights, grouped_folds, sampling_weights


class PolarityCvExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        labels = ('positive', 'negative', 'neutral', 'conflict')
        self.rows = [LabeledRow(str(index), f'text {index}', 'service', labels[index % 4], index)
                     for index in range(40)]

    def test_grouped_folds_are_deterministic_disjoint_and_complete(self) -> None:
        first = grouped_folds(self.rows, 5, 42); second = grouped_folds(self.rows, 5, 42)
        self.assertEqual(first, second)
        self.assertEqual(set().union(*first), {row.item_id for row in self.rows})
        self.assertEqual(sum(len(values) for values in first), 40)

    def test_oversampling_changes_only_minority_sampling_frequency(self) -> None:
        values = sampling_weights(self.rows[:4], 'oversampling')
        self.assertEqual(values, [1.0, 1.0, 2.0, 3.0])
        self.assertIsNone(sampling_weights(self.rows, 'reference'))

    def test_corrected_weights_remove_sampling_multiplier(self) -> None:
        reference = class_weights(torch, self.rows, 'reference')
        corrected = class_weights(torch, self.rows, 'sampling-corrected')
        self.assertTrue(torch.allclose(corrected, reference / torch.tensor((1.0, 1.0, 2.0, 3.0))))


if __name__ == '__main__':
    unittest.main()
