from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from zeroshot_classifier.data import LabeledRow
from zeroshot_classifier.oof_error_audit import (
    _validate_partition,
    _report,
    apply_assessments,
    build_aspect_candidates,
    build_polarity_candidates,
    select_cases,
)


class OofErrorAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            LabeledRow('1', 'food and overall', 'food', 'positive', 0),
            LabeledRow('1', 'food and overall', 'anecdotes/miscellaneous', 'neutral', 1),
            LabeledRow('2', 'plain food', 'food', 'negative', 2),
            LabeledRow('3', 'general remark', 'anecdotes/miscellaneous', 'conflict', 3),
        ]

    def test_partition_rejects_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            _validate_partition([{'1', '2'}, {'2', '3'}], {'1', '2', '3'}, 'test')

    def test_partition_rejects_missing_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            _validate_partition([{'1'}, {'2'}], {'1', '2', '3'}, 'test')

    def test_multi_aspect_annotations_remain_one_review_set(self) -> None:
        predictions = {
            'reference': {'1': {'food'}, '2': {'food'}, '3': {'food'}},
            'candidate': {'1': {'food', 'anecdotes/miscellaneous'}, '2': {'food'}, '3': {'food'}},
        }
        candidates = build_aspect_candidates(self.rows, predictions, {'1': 1, '2': 1, '3': 2})

        first = next(value for value in candidates if value['id'] == '1')
        self.assertEqual(first['bucket'], 'aspect_multi_misc')
        self.assertEqual(first['gold_aspects'], ['food', 'anecdotes/miscellaneous'])
        self.assertEqual(first['error_persistence'], 1)

    def test_polarity_is_aligned_by_id_and_aspect(self) -> None:
        predictions = {
            'reference': {('1', 'food'): 'negative', ('1', 'anecdotes/miscellaneous'): 'neutral',
                          ('2', 'food'): 'negative', ('3', 'anecdotes/miscellaneous'): 'neutral'},
            'candidate': {('1', 'food'): 'positive', ('1', 'anecdotes/miscellaneous'): 'neutral',
                          ('2', 'food'): 'negative', ('3', 'anecdotes/miscellaneous'): 'positive'},
        }
        candidates = build_polarity_candidates(self.rows, predictions, {'1': 1, '2': 1, '3': 2})

        self.assertEqual({value['case_key'] for value in candidates}, {
            'polarity:1:food', 'polarity:3:anecdotes/miscellaneous'
        })
        conflict = next(value for value in candidates if value['id'] == '3')
        self.assertEqual(conflict['bucket'], 'polarity_conflict')

    def test_selection_is_deterministic_and_uses_unique_ids(self) -> None:
        values = [
            {'case_key': f'aspect:{item}', 'bucket': 'aspect_other', 'id': str(item),
             'error_persistence': persistence}
            for item, persistence in ((10, 1), (2, 3), (3, 3))
        ]
        values.append({'case_key': 'polarity:2:food', 'bucket': 'polarity_other',
                       'id': '2', 'error_persistence': 4})

        selected = select_cases(values, 3)

        self.assertEqual([value['id'] for value in selected], ['2', '3', '10'])
        self.assertEqual(len({value['id'] for value in selected}), 3)

    def test_invalid_assessment_is_rejected(self) -> None:
        cases = [{'case_key': 'aspect:1'}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'assessments.json'
            path.write_text(json.dumps([{
                'case_key': 'aspect:1', 'assessment': 'certainly wrong'
            }]))
            with self.assertRaisesRegex(ValueError, 'Invalid assessment'):
                apply_assessments(cases, path)

    def test_report_counts_partial_review_without_hardcoded_totals(self) -> None:
        cases = [{'bucket': 'aspect_other', 'assessment': 'ambiguous annotation',
                  'rationale': 'Both readings are plausible.'}]
        report = _report(cases, cases)
        self.assertIn('0 of 1 cases', report)
        self.assertIn('| ambiguous annotation | 1 |', report)
        self.assertNotIn('adjudicate these 62', report)


if __name__ == '__main__':
    unittest.main()
