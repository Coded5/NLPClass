from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from zeroshot_classifier.conflict_augmentation import metrics, training_rows
from zeroshot_classifier.conflict_augmentation_multiseed import (
    CONDITIONS, SEEDS, average_records, conflict_examples, freeze_repetitions,
    matched_counts, report,
)
from zeroshot_classifier.data import LabeledRow


class ConflictAugmentationMultiseedTests(unittest.TestCase):
    def examples(self) -> list[dict]:
        aspects = ['food'] * 20 + ['service'] * 12 + ['ambience'] * 6 + ['price'] * 2
        return [{'key': str(i), 'kind': 'conflict', 'aspect': aspect,
                 'polarity': 'conflict', 'text': f'synthetic {i}'}
                for i, aspect in enumerate(aspects)]

    def fit(self) -> list[LabeledRow]:
        return [LabeledRow(f'{aspect}-{i}', f'real {aspect} conflict {i}', aspect,
                           'conflict', i) for aspect in ('food', 'service', 'ambience', 'price')
                for i in range(3)]

    def records(self, fold: int, offset: float = 0) -> list[dict]:
        labels = ('positive', 'negative', 'neutral', 'conflict')
        values = []
        for index, label in enumerate(labels):
            logits = [0.0] * 4
            logits[index] = 2 + offset
            values.append({'position': fold * 4 + index, 'id': f'{fold}-{index}',
                           'aspect': 'food', 'gold': label, 'predicted': label,
                           'text': 'text', 'logits': logits})
        return values

    def test_conflict_filter_and_exact_aspect_match(self) -> None:
        source = {'examples': self.examples() + [{'kind': 'control', 'polarity': 'positive'}]}
        examples = conflict_examples(source)
        counts = matched_counts(self.fit(), examples)
        examples = freeze_repetitions(self.fit(), examples, 50001)
        self.assertEqual(counts['requested'], Counter(e['aspect'] for e in examples))
        repeated = training_rows(self.fit(), examples, 'repetition', 50001)[len(self.fit()):]
        synthetic = training_rows(self.fit(), examples, 'synthetic', 50001)[len(self.fit()):]
        self.assertEqual(Counter(r.aspect for r in repeated), Counter(r.aspect for r in synthetic))
        self.assertTrue(all(r.polarity == 'conflict' for r in repeated + synthetic))
        self.assertEqual(examples, freeze_repetitions(self.fit(), conflict_examples(source), 50001))
        self.assertTrue(all('repetition_position' in e for e in examples))

    def test_frozen_repetition_rejects_changed_fit_data(self) -> None:
        examples = freeze_repetitions(self.fit(), self.examples(), 50001)
        position = examples[0]['repetition_position']
        changed = [row for row in self.fit() if row.position != position]
        with self.assertRaises(ValueError):
            training_rows(changed, [examples[0]], 'repetition', 50001)

    def test_missing_real_conflict_for_an_aspect_is_rejected(self) -> None:
        fit = [row for row in self.fit() if row.aspect != 'price']
        with self.assertRaises(ValueError):
            matched_counts(fit, self.examples())

    def test_average_records_uses_logits_and_validates_alignment(self) -> None:
        first = self.records(0)
        second = self.records(0, 1)
        averaged = average_records([first, list(reversed(second))])
        self.assertEqual(metrics(averaged)['macro_f1'], 1)
        self.assertEqual(averaged[0]['logits'][0], 2.5)
        with self.assertRaises(ValueError):
            average_records([first, second[:-1]])
        changed = [{**second[0], 'gold': 'negative'}, *second[1:]]
        with self.assertRaises(ValueError):
            average_records([first, changed])

    def test_report_covers_three_seeds_five_folds_and_two_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folds = [{'fold': fold, 'examples': self.examples()} for fold in range(1, 6)]
            (root / 'preparation.json').write_text(json.dumps({'folds': folds}))
            for fold in range(1, 6):
                for seed in SEEDS:
                    for condition in CONDITIONS:
                        directory = root / f'fold-{fold}' / f'seed-{seed}' / condition
                        directory.mkdir(parents=True)
                        records = self.records(fold)
                        (directory / 'result.json').write_text(json.dumps({
                            'finished': True, 'training_seconds': 1,
                            'peak_training_allocated_gib': 1,
                            'training': {'epoch': 5},
                            'metrics': {'heldout': metrics(records)},
                            'predictions': {'heldout': records},
                        }))
            value = report(root, samples=20)
            self.assertEqual(value['design']['fits'], 30)
            self.assertEqual(value['ensemble']['synthetic']['support'], 20)
            self.assertEqual(value['training']['synthetic']['seconds'], 15)
            self.assertFalse(value['decision']['synthetic_wording_supported'])
            self.assertTrue((root / 'report.md').is_file())


if __name__ == '__main__':
    unittest.main()
