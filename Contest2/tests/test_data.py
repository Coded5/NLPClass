import csv
import tempfile
import unittest
from pathlib import Path

from zeroshot_classifier.data import LabeledRow, read_labeled_csv, stratified_group_split, write_split


class DataTests(unittest.TestCase):
    def test_grouped_split_is_deterministic_and_has_no_id_leakage(self):
        rows = []
        position = 0
        labels = (
            ('food', 'positive'),
            ('food', 'negative'),
            ('service', 'positive'),
            ('service', 'negative'),
        )
        for aspect, polarity in labels:
            for index in range(10):
                item_id = f'{aspect}-{polarity}-{index}'
                rows.append(LabeledRow(item_id, f'Text {item_id}', aspect, polarity, position))
                position += 1
                if index % 3 == 0:
                    rows.append(
                        LabeledRow(
                            item_id,
                            f'Text {item_id}',
                            'ambience',
                            'neutral',
                            position,
                        )
                    )
                    position += 1

        first = stratified_group_split(rows, test_ratio=0.2, seed=7)
        second = stratified_group_split(rows, test_ratio=0.2, seed=7)

        self.assertEqual(first.test_ids, second.test_ids)
        self.assertFalse(first.train_ids & first.test_ids)
        self.assertEqual(len(first.test_ids), 8)
        all_labels = {(row.aspect, row.polarity) for row in rows}
        test_labels = {(row.aspect, row.polarity) for row in first.test_rows}
        self.assertEqual(all_labels, test_labels)

    def test_write_split_creates_artifacts_without_changing_source(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / 'source.csv'
            with source.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(('id', 'text', 'aspectCategory', 'polarity'))
                for index in range(20):
                    writer.writerow((index, f'Review {index}', 'food', 'positive'))
            original = source.read_bytes()

            manifest = write_split(source, root / 'split', seed=42)

            self.assertEqual(source.read_bytes(), original)
            self.assertTrue((root / 'split' / 'train.csv').is_file())
            self.assertTrue((root / 'split' / 'test.csv').is_file())
            self.assertEqual(manifest['ids']['test'], 2)
            self.assertEqual(len(read_labeled_csv(root / 'split' / 'test.csv')), 2)

    def test_coverage_repair_can_add_multiple_rare_joint_labels(self):
        rows = []
        labels = [
            ('food', 'positive'),
            ('price', 'neutral'),
            ('service', 'conflict'),
        ]
        for label_index, (aspect, polarity) in enumerate(labels):
            for index in range(10):
                item_id = f'{label_index}-{index}'
                rows.append(
                    LabeledRow(item_id, f'Review {item_id}', aspect, polarity, len(rows))
                )

        split = stratified_group_split(rows, test_ratio=0.1, seed=1)
        test_labels = {(row.aspect, row.polarity) for row in split.test_rows}

        self.assertEqual(test_labels, set(labels))


if __name__ == '__main__':
    unittest.main()
