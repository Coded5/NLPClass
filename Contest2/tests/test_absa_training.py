import unittest

from zeroshot_classifier.absa_training import (
    AspectDataset,
    PolarityDataset,
    decode_aspects,
    group_aspect_targets,
    tune_thresholds,
)
from zeroshot_classifier.data import LabeledRow


class AbsaTrainingTests(unittest.TestCase):
    def test_groups_all_aspects_for_one_review(self):
        rows = [
            LabeledRow('1', 'Great food, slow service', 'food', 'positive', 0),
            LabeledRow('1', 'Great food, slow service', 'service', 'negative', 1),
        ]

        grouped = group_aspect_targets(rows)

        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0].aspects, (0, 2))

    def test_aspect_dataset_builds_multihot_target(self):
        grouped = group_aspect_targets([
            LabeledRow('1', 'Review', 'food', 'positive', 0),
            LabeledRow('1', 'Review', 'service', 'negative', 1),
        ])
        dataset = AspectDataset(grouped, FakeTokenizer(), 128)

        self.assertEqual(dataset[0]['labels'], [1.0, 0.0, 1.0, 0.0, 0.0])

    def test_polarity_dataset_conditions_on_aspect(self):
        dataset = PolarityDataset([
            LabeledRow('1', 'Review', 'service', 'negative', 0)
        ], FakeTokenizer(), 128)

        self.assertEqual(dataset[0]['second'], 'aspect: service')
        self.assertEqual(dataset[0]['labels'], 1)

    def test_decoder_emits_multiple_aspects_and_has_fallback(self):
        decoded = decode_aspects(
            [[0.8, 0.1, 0.7, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4, 0.49]],
            [0.5] * 5,
        )

        self.assertEqual(decoded, [[0, 2], [4]])

    def test_threshold_tuning_uses_validation_targets(self):
        probabilities = [[0.9, 0, 0, 0, 0], [0.4, 0, 0, 0, 0], [0.2, 0, 0, 0, 0]]
        targets = [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

        thresholds = tune_thresholds(probabilities, targets)

        self.assertGreater(thresholds[0], 0.2)
        self.assertLessEqual(thresholds[0], 0.4)


class FakeTokenizer:
    def __call__(self, text, second=None, **kwargs):
        return {'text': text, 'second': second}


if __name__ == '__main__':
    unittest.main()
