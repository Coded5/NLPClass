import unittest

from zeroshot_classifier.deberta_three_experiments import (
    SYSTEMS,
    average_backbone_logits,
    config_from_args,
    system_components,
)


class DebertaThreeExperimentTests(unittest.TestCase):
    def test_three_systems_are_independently_named(self):
        self.assertEqual(len(SYSTEMS), 3)
        self.assertEqual(len(set(SYSTEMS)), 3)
        self.assertEqual(
            system_components('roberta-aspect-mixed-polarity'),
            ('roberta', 'mixed'),
        )

    def test_mixed_ensemble_weights_backbones_equally(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is unavailable')
        left = torch.tensor([[1.0, 3.0]])
        right = torch.tensor([[3.0, 5.0]])

        self.assertTrue(torch.equal(
            average_backbone_logits(torch, left, right),
            torch.tensor([[2.0, 4.0]]),
        ))

    def test_cli_defaults_to_existing_pilot_and_batch_eight(self):
        config = config_from_args([])

        self.assertEqual(config.train_batch_size, 8)
        self.assertEqual(config.pilot_dir.name, 'deberta-v3-current-best-v1')

    def test_unknown_system_is_rejected(self):
        with self.assertRaises(ValueError):
            system_components('unknown')


if __name__ == '__main__':
    unittest.main()
