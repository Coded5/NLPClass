import unittest

from zeroshot_classifier.data import LabeledRow
from zeroshot_classifier.imbalance_losses import LossSpec
from zeroshot_classifier.absa_training import group_aspect_targets
from zeroshot_classifier.joint_absa_experiment import (
    JointCandidateDataset,
    candidate_targets,
    decode_pairs,
    masked_joint_loss,
    build_variant_parser,
    tune_pair_thresholds,
)


class JointAbsaExperimentTests(unittest.TestCase):
    def test_each_review_creates_five_candidates_with_masked_polarities(self):
        rows = [
            LabeledRow('1', 'Great food but slow service', 'food', 'positive', 0),
            LabeledRow('1', 'Great food but slow service', 'service', 'negative', 1),
        ]

        targets = candidate_targets(rows)
        dataset = JointCandidateDataset(targets, FakeTokenizer(), 128)

        self.assertEqual(len(targets), 5)
        self.assertEqual([target.present for target in targets], [1, 0, 1, 0, 0])
        self.assertEqual(targets[0].polarity_index, 0)
        self.assertEqual(targets[1].polarity_index, -100)
        self.assertEqual(dataset[2]['second'], 'aspect: service')

    def test_absent_candidate_has_no_polarity_gradient(self):
        torch = self._torch()
        aspect_logits = torch.tensor([0.2], requires_grad=True)
        polarity_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]], requires_grad=True)
        total, _, polarity = masked_joint_loss(
            torch, aspect_logits, polarity_logits,
            torch.tensor([0.0]), torch.tensor([-100]), torch.tensor([1]),
            torch.ones(5), torch.ones(4), LossSpec('weighted-ce'),
        )

        total.backward()

        self.assertEqual(float(polarity.detach()), 0.0)
        self.assertTrue(bool((polarity_logits.grad == 0).all()))

    def test_decoder_returns_multiple_pairs_and_falls_back(self):
        rows = [LabeledRow('1', 'Review', 'food', 'positive', 0)]
        reviews = group_aspect_targets(rows)

        multiple = decode_pairs(
            reviews, [[0.8, 0.1, 0.7, 0.1, 0.1]], [[0, 0, 1, 0, 0]], [0.5] * 5
        )
        fallback = decode_pairs(
            reviews, [[0.1, 0.2, 0.3, 0.4, 0.49]], [[0, 0, 0, 0, 2]], [0.5] * 5
        )

        self.assertEqual(multiple, [('1', 'food', 'positive'), ('1', 'service', 'negative')])
        self.assertEqual(fallback, [('1', 'anecdotes/miscellaneous', 'neutral')])

    def test_threshold_search_is_deterministic(self):
        rows = [
            LabeledRow('1', 'One', 'food', 'positive', 0),
            LabeledRow('2', 'Two', 'service', 'negative', 1),
        ]
        probabilities = [[0.8, 0.1, 0.2, 0.1, 0.1], [0.2, 0.1, 0.7, 0.1, 0.1]]
        polarities = [[0] * 5, [1] * 5]

        first = tune_pair_thresholds(rows, probabilities, polarities)
        second = tune_pair_thresholds(rows, probabilities, polarities)

        self.assertEqual(first, second)
        self.assertEqual(
            decode_pairs(
                group_aspect_targets(rows),
                probabilities, polarities, first,
            ),
            [('1', 'food', 'positive'), ('2', 'service', 'negative')],
        )

    def test_variant_cli_exposes_loss_controls(self):
        args = build_variant_parser().parse_args([
            '--architecture', 'joint',
            '--polarity-loss', 'class-balanced-focal',
            '--focal-gamma', '1.5',
            '--class-balance-beta', '0.99',
            '--run-dir', 'artifacts/example',
        ])

        self.assertEqual(args.architecture, 'joint')
        self.assertEqual(args.polarity_loss, 'class-balanced-focal')
        self.assertEqual(args.focal_gamma, 1.5)
        self.assertEqual(args.class_balance_beta, 0.99)

    def _torch(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is not installed')
        return torch


class FakeTokenizer:
    def __call__(self, text, second=None, **kwargs):
        return {'text': text, 'second': second}


if __name__ == '__main__':
    unittest.main()
