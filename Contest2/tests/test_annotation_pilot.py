from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'annotation_pilot', Path(__file__).resolve().parents[1] / 'scripts/review_annotation_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class AnnotationPilotTests(unittest.TestCase):
    def test_selection_balances_strata_and_is_deterministic(self):
        cases = [{'id': str(i), 'case_key': str(i), 'bucket': bucket,
                  'assessment': ['clear model error', 'ambiguous annotation'][i % 2]}
                 for i, bucket in enumerate(
                     ['aspect_multi_misc'] * 10 + ['polarity_conflict'] * 10 +
                     ['polarity_neutral'] * 10 + ['aspect_other'] * 10)]
        first = pilot.select_pilot(cases)
        self.assertEqual(first, pilot.select_pilot(cases))
        self.assertEqual(len({v['id'] for v in first}), 20)
        for bucket in {v['bucket'] for v in cases}:
            self.assertEqual(sum(v['bucket'] == bucket for v in first), 5)

    def test_response_allows_multiple_aspects_but_rejects_duplicate_aspect(self):
        response = {'review_id': 'P01', 'status': 'complete', 'pairs': [
            {'aspect': 'food', 'polarity': 'positive', 'evidence': 'tasty'},
            {'aspect': 'service', 'polarity': 'negative', 'evidence': 'slow'}]}
        self.assertIn('P01', pilot.validate_responses([response], {'P01'}))
        response['pairs'].append(response['pairs'][0])
        with self.assertRaisesRegex(ValueError, 'one polarity'):
            pilot.validate_responses([response], {'P01'})

    def test_invalid_label_unknown_id_and_missing_evidence(self):
        for pair, allowed in [
            ({'aspect': 'other', 'polarity': 'neutral', 'evidence': 'x'}, {'P01'}),
            ({'aspect': 'food', 'polarity': 'neutral', 'evidence': ''}, {'P01'}),
            ({'aspect': 'food', 'polarity': 'neutral', 'evidence': 'x'}, {'P02'}),
        ]:
            with self.assertRaises(ValueError):
                pilot.validate_responses([{'review_id': 'P01', 'status': 'complete', 'pairs': [pair]}], allowed)

    def test_source_match_uses_text_and_complete_sets_not_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train = root / 'train.csv'
            train.write_text('id,text,aspectCategory,polarity\n1,Good food,food,positive\n')
            source = root / 'source.xml'
            source.write_text('<sentences><sentence id="different"><text>Good food</text>'
                              '<aspectCategories><aspectCategory category="food" polarity="positive"/>'
                              '</aspectCategories></sentence></sentences>')
            pilot.compare_source(root, train, source)
            result = json.loads((root / 'private/source_comparison.json').read_text())
            self.assertEqual(result['counts'], {'same_annotations': 1})

    def test_partial_review_does_not_reveal_comparisons(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'private').mkdir()
            (root / 'private/key.json').write_text(json.dumps([
                {'review_id': 'P01'}, {'review_id': 'P02'}]))
            response = root / 'incoming.json'
            response.write_text(json.dumps([{'review_id': 'P01', 'status': 'insufficient_context',
                                            'pairs': [], 'note': 'fragment'}]))
            pilot.import_responses(root, response)
            self.assertFalse((root / 'comparison.json').exists())
            self.assertEqual(json.loads((root / 'progress.json').read_text())['submitted'], 1)


if __name__ == '__main__':
    unittest.main()
