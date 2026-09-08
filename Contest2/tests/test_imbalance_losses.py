import unittest

from zeroshot_classifier.imbalance_losses import (
    LossSpec,
    class_balanced_weights,
    polarity_loss,
)


class ImbalanceLossTests(unittest.TestCase):
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
