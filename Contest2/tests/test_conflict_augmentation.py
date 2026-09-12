from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

from zeroshot_classifier.conflict_augmentation import (
    CONDITIONS, RULES, digest, generate, join_clauses, metrics, paired_intervals,
    promotion, report, training_rows, validate_reviews,
)
from zeroshot_classifier.data import LabeledRow


class ConflictAugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fit = []
        self.bank = []
        for aspect in ('food', 'service'):
            for polarity in ('positive', 'negative'):
                for i in range(15):
                    key = f'{aspect}-{polarity}-{i}'
                    text = f'The {aspect} was {"good" if polarity == "positive" else "bad"} on visit {i}'
                    self.fit.append(LabeledRow(key, text, aspect, polarity, len(self.fit)))
                    self.bank.append({'key': key, 'id': key, 'aspect': aspect, 'polarity': polarity, 'span': text})
            for i in range(20):
                self.fit.append(LabeledRow(f'{aspect}-conflict-{i}', 'Mixed review', aspect, 'conflict', len(self.fit)))

    def test_generation_is_deterministic_leak_free_and_bounded(self) -> None:
        excluded = {r.item_id for r in self.fit[:4]}
        fit = [r for r in self.fit if r.item_id not in excluded]
        first = generate(self.bank, fit, 42)
        self.assertEqual(first, generate(self.bank, fit, 42))
        self.assertEqual(len(first), 80)
        uses = Counter()
        pairs = set()
        for e in first:
            a, b = e['sources']
            self.assertNotEqual(a['id'], b['id'])
            self.assertFalse({a['id'], b['id']} & excluded)
            self.assertEqual(a['aspect'] == b['aspect'], e['kind'] == 'conflict')
            if e['kind'] == 'control':
                target = a if e['polarity'] == 'positive' else b
                self.assertEqual(target['aspect'], e['aspect'])
            pair = (a['key'], b['key'])
            self.assertNotIn(pair, pairs)
            pairs.add(pair)
            uses.update(pair)
        self.assertLessEqual(max(uses.values()), 5)
        self.assertEqual(len({e['text'].lower() for e in first}), len(first))
        self.assertEqual(sum(e['template'] == 'sentences' for e in first), 16)

    def test_low_source_coverage_stops_preparation(self) -> None:
        self.assertEqual(generate(self.bank[:2], self.fit, 42), [])

    def test_templates_have_correct_grammar(self) -> None:
        self.assertEqual(join_clauses('The food is good', 'The soup is bad', 'although'),
                         'Although the food is good, the soup is bad.')
        self.assertEqual(join_clauses('The food is good', 'The soup is bad', 'however'),
                         'The food is good; however, the soup is bad.')

    def test_repetition_matches_synthetic_aspect_and_class_counts(self) -> None:
        examples = generate(self.bank, self.fit, 42)
        repeated = training_rows(self.fit, examples, 'repetition', 42)
        synthetic = training_rows(self.fit, examples, 'synthetic', 42)
        count = lambda rows: Counter((r.aspect, r.polarity) for r in rows)
        self.assertEqual(count(repeated), count(synthetic))
        self.assertEqual(len(repeated), len(synthetic))
        self.assertEqual(training_rows(self.fit, examples, 'reference', 42), self.fit)
        self.assertEqual(repeated, training_rows(self.fit, examples, 'repetition', 42))
        self.assertTrue(all(r in self.fit for r in repeated))

    def test_reviews_are_complete_and_bound_to_preparation(self) -> None:
        prepared = {'ready_for_review': True, 'folds': [{'fold': 1, 'examples': [{'key': 'a'}]}]}
        reviews = {'reviewer': 'assistant', 'preparation_sha256': digest(prepared), 'accepted': {'1:a': 'review'}}
        validate_reviews(prepared, reviews)
        for changed in ({**reviews, 'accepted': {}}, {**reviews, 'preparation_sha256': 'changed'}):
            with self.assertRaises(ValueError):
                validate_reviews(prepared, changed)

    def records(self, fold: int = 0) -> list[dict]:
        return [{'position': fold * 4 + i, 'id': f'{fold}-{i // 2}', 'text': 'Good but bad',
                 'aspect': 'food' if i % 2 == 0 else 'service', 'gold': label, 'predicted': label}
                for i, label in enumerate(('positive', 'negative', 'neutral', 'conflict'))]

    def test_duplicate_text_different_aspects_scores_perfectly(self) -> None:
        self.assertEqual(metrics(self.records())['macro_f1'], 1)
        self.assertEqual(metrics(self.records())['support'], 4)

    def test_bootstrap_aligns_positions_and_rejects_changed_gold(self) -> None:
        records = self.records()
        result = paired_intervals(records, list(reversed(records)), samples=30)
        self.assertEqual(result['conflict']['lower_95'], 0)
        bad = [{**records[0], 'gold': 'conflict'}, *records[1:]]
        with self.assertRaises(ValueError):
            paired_intervals(records, bad, samples=30)
        with self.assertRaises(ValueError):
            paired_intervals(records, records[:-1], samples=30)

    def test_promotion_requires_all_guardrails(self) -> None:
        baseline = metrics(self.records())
        baseline['classes']['conflict']['f1'] = 0.4
        synthetic = json.loads(json.dumps(baseline))
        synthetic['classes']['conflict']['f1'] = 0.5
        models = {name: {'pooled': baseline if name != 'synthetic' else synthetic} for name in CONDITIONS}
        interval = {'conflict': {'delta': 0.1, 'lower_95': 0.01}}
        self.assertTrue(promotion(models, interval, 4)['advance_to_confirmation'])
        self.assertFalse(promotion(models, interval, 3)['advance_to_confirmation'])
        synthetic['conflict_false_positive_rate'] = 0.02
        self.assertFalse(promotion(models, interval, 4)['advance_to_confirmation'])

    def test_report_aggregates_five_folds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'preparation.json').write_text(json.dumps({'folds': [{'fold': i} for i in range(5)]}))
            for i in range(5):
                for condition in CONDITIONS:
                    directory = root / f'fold-{i}' / condition
                    directory.mkdir(parents=True)
                    records = self.records(i)
                    (directory / 'result.json').write_text(json.dumps({
                        'finished': True, 'predictions': {'heldout': records},
                        'metrics': {'heldout': metrics(records), 'diagnostic': metrics(records)},
                        'training': {'epoch': 5}, 'training_seconds': 1, 'peak_training_allocated_gib': 1,
                    }))
            result = report(root, samples=10)
            self.assertEqual(result['models']['synthetic']['pooled']['support'], 20)
            self.assertFalse(result['decision']['advance_to_confirmation'])
            self.assertTrue((root / 'report.md').is_file())

    def test_training_uses_original_rows_for_class_weights(self) -> None:
        from zeroshot_classifier.absa_training import ExperimentConfig, train_stage
        torch = MagicMock()
        transformers = MagicMock()
        tokenizer = MagicMock(return_value={'input_ids': [1]})
        transformers.AutoModelForSequenceClassification.from_pretrained.side_effect = RuntimeError('stop before model construction')
        originals = self.fit
        augmented = self.fit + self.fit[:5]
        with patch('zeroshot_classifier.absa_training._polarity_weights') as weights:
            with self.assertRaisesRegex(RuntimeError, 'stop before model construction'):
                train_stage(torch, transformers, MagicMock(), tokenizer, ExperimentConfig(),
                            'polarity', augmented, originals, polarity_weight_rows=originals)
            weights.assert_called_once_with(torch, originals)

    def test_completed_fit_resumes_without_loading_a_model(self) -> None:
        from zeroshot_classifier.absa_training import ExperimentConfig
        from zeroshot_classifier.conflict_augmentation import train_one
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            saved = {'finished': True, 'condition': 'synthetic'}
            (directory / 'result.json').write_text(json.dumps(saved))
            self.assertEqual(train_one(ExperimentConfig(run_dir=directory), 'synthetic',
                                      [], [], [], [], None, None, None, None), saved)

    def test_frozen_manifest_rejects_configuration_change(self) -> None:
        from zeroshot_classifier.learning_curve_experiment import _write_or_validate
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'manifest.json'
            _write_or_validate(path, {'seed': 42})
            _write_or_validate(path, {'seed': 42})
            with self.assertRaises(RuntimeError):
                _write_or_validate(path, {'seed': 43})


if __name__ == '__main__':
    unittest.main()
