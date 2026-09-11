import unittest

from zeroshot_classifier.imbalance_losses import (
    LossSpec,
    class_balanced_weights,
    loss_weights,
    polarity_loss,
)
from zeroshot_classifier.data import LabeledRow


class ImbalanceLossTests(unittest.TestCase):
    def test_extra_multipliers_only_change_neutral_and_conflict(self):
        torch = self._torch()
        rows = [
            LabeledRow(str(index), 'Text', 'food', polarity, index)
            for index, polarity in enumerate(
                ['positive'] * 8 + ['negative'] * 4
                + ['neutral'] * 2 + ['conflict']
            )
        ]
        baseline = loss_weights(torch, rows, LossSpec('weighted-ce'))
        adjusted = loss_weights(
            torch, rows,
            LossSpec(
                'weighted-ce', neutral_multiplier=1.5,
                conflict_multiplier=3.0,
            ),
        )

        self.assertAlmostEqual(float(adjusted[0]), float(baseline[0]))
        self.assertAlmostEqual(float(adjusted[1]), float(baseline[1]))
        self.assertAlmostEqual(float(adjusted[2]), float(baseline[2]) * 1.5)
        self.assertAlmostEqual(float(adjusted[3]), float(baseline[3]) * 3.0)

    def test_multiplier_validation_rejects_downweighting(self):
        with self.assertRaisesRegex(ValueError, 'must be at least 1'):
            LossSpec('weighted-ce', neutral_multiplier=0.9).validate()

    def test_serialization_preserves_legacy_manifest_shape(self):
        self.assertEqual(
            LossSpec('weighted-ce').serialized(),
            {'name': 'weighted-ce', 'gamma': 0.0, 'beta': 0.999},
        )
        self.assertEqual(
            LossSpec(
                'weighted-ce', neutral_multiplier=1.5,
                conflict_multiplier=2.0,
            ).serialized(),
            {
                'name': 'weighted-ce', 'gamma': 0.0, 'beta': 0.999,
                'neutral_multiplier': 1.5, 'conflict_multiplier': 2.0,
            },
        )

    def test_class_balanced_weights_are_normalized_and_favor_rare_classes(self):
        torch = self._torch()
        weights = class_balanced_weights(torch, [100, 20, 5], 0.999)

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertLess(float(weights[1]), float(weights[2]))
        self.assertTrue(bool(torch.isfinite(weights).all()))

    def test_weighted_focal_gamma_zero_matches_weighted_cross_entropy(self):
        torch = self._torch()
        logits = torch.tensor([[2.0, -1.0], [-0.5, 0.5]])
        targets = torch.tensor([0, 1])
        weights = torch.tensor([0.7, 1.3])

        expected = polarity_loss(
            torch, logits, targets, weights, LossSpec('weighted-ce')
        )
        native = torch.nn.functional.cross_entropy(logits, targets, weight=weights)
        actual = polarity_loss(
            torch, logits, targets, weights,
            LossSpec('weighted-focal', gamma=0.0),
        )

        self.assertAlmostEqual(float(actual), float(expected), places=7)
        self.assertAlmostEqual(float(actual), float(native), places=7)

    def _torch(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is not installed')
        return torch


if __name__ == '__main__':
    unittest.main()
