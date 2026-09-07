"""Focused tests for the n-gram contest pipeline."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from contest1 import data_policy
from contest1.contest_pipeline import (
    evaluate_dev,
    generate_test_output,
    iter_contest_rows,
    validate_output,
)
from contest1.n_gram_generator import NGramWordPredictor


class ContestPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.corpus = self.root / "corpus.txt"
        self.model = self.root / "model.sqlite3"
        self.corpus.write_text(
            "tie apricot\n"
            "tie apple\n"
            "seen alpha\n"
            "else beta\n"
            "other beta\n"
            "more beta\n"
            "punct !bang\n"
            "number 123\n"
            "upper Upper\n"
            "quoted , context apple\n"
            "a b c xz\n"
            "z b c xa\n",
            encoding="utf-8",
        )
        NGramWordPredictor.train(
            self.corpus,
            self.model,
            order=5,
            batch_size=10,
            progress_every=0,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_csv(self, name: str, header: list[str], rows: list[list[str]]) -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_tiny_corpus_counts_literal_hints_and_tie_break(self) -> None:
        with NGramWordPredictor(self.model) as predictor:
            generic = predictor.predict("tie", 2)
            self.assertEqual([item.word for item in generic], ["apple", "apricot"])
            self.assertEqual([item.count for item in generic], [1, 1])
            self.assertEqual(
                predictor.predict_with_hint("punct", "!")[0].word, "!bang"
            )
            self.assertEqual(
                predictor.predict_with_hint("number", "1")[0].word, "123"
            )
            self.assertEqual(
                predictor.predict_with_hint("upper", "U")[0].word, "Upper"
            )
            lowercase = predictor.predict_with_hint("upper", "u")
            self.assertTrue(lowercase)
            self.assertTrue(all(item.word[0] == "u" for item in lowercase))
            self.assertNotIn("Upper", [item.word for item in lowercase])

    def test_seen_context_without_matching_candidate_backs_off(self) -> None:
        with NGramWordPredictor(self.model) as predictor:
            result = predictor.predict_with_hint_result("seen", "b", 1)
        self.assertEqual(result.predictions[0].word, "beta")
        self.assertEqual(result.context_order, 0)

    def test_effective_order_reuses_higher_order_database(self) -> None:
        with NGramWordPredictor(self.model) as predictor:
            five_gram = predictor.predict_with_hint("a b c", "x", effective_order=5)
            trigram = predictor.predict_with_hint("a b c", "x", effective_order=3)
        self.assertEqual(five_gram[0].word, "xz")
        self.assertEqual(trigram[0].word, "xa")

    def test_csv_quoting_duplicates_metrics_and_output_order(self) -> None:
        dev = self.write_csv(
            "dev.csv",
            ["context", "first letter", "answer"],
            [
                ["quoted , context", "a", "apple"],
                ["seen", "b", "beta"],
                ["seen", "b", "beta"],
            ],
        )
        parsed = list(iter_contest_rows(dev, require_answers=True))
        self.assertEqual(parsed[0].context, "quoted , context")
        self.assertEqual(parsed[1], parsed[2])
        metrics = evaluate_dev(self.model, dev, effective_order=3, top_k=2)
        self.assertEqual(metrics.rows, 3)
        self.assertEqual(metrics.covered, 3)
        self.assertEqual(metrics.correct, 3)
        self.assertEqual(metrics.top_k_correct, 3)
        self.assertEqual(sum(metrics.backoff_levels.values()), 3)

        test = self.write_csv(
            "test.csv",
            ["context", "first letter"],
            [["seen", "b"], ["tie", "a"], ["seen", "b"]],
        )
        output = self.root / "predictions.txt"
        self.assertEqual(generate_test_output(self.model, test, output), 3)
        self.assertEqual(output.read_text(encoding="utf-8"), "beta\napple\nbeta\n")
        self.assertEqual(validate_output(test, output), 3)

    def test_strict_output_validation(self) -> None:
        test = self.write_csv(
            "test.csv", ["context", "first letter"], [["one", "a"], ["two", "!"]]
        )
        output = self.root / "output.txt"
        invalid_outputs = (
            "apple\n",
            "apple\ntwo words\n",
            "apple\n?wrong\n",
            "\n!bang\n",
        )
        for contents in invalid_outputs:
            with self.subTest(contents=contents):
                output.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_output(test, output)
        output.write_text("apple\n!bang\n", encoding="utf-8")
        self.assertEqual(validate_output(test, output), 2)

    def test_forbidden_evaluation_paths_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_dev(self.model, self.root / "devv_test.csv")

        forbidden = self.root / "devv_test.csv"
        forbidden.write_text("not opened through its alias", encoding="utf-8")
        alias = self.root / "renamed.csv"
        alias.symlink_to(forbidden)
        with self.assertRaises(ValueError):
            data_policy.reject_forbidden_evaluation_path(alias)

        copied = self.root / "copied.csv"
        copied.write_bytes(b"forbidden fingerprint")
        fingerprint = hashlib.sha256(copied.read_bytes()).hexdigest()
        with mock.patch.object(data_policy, "FORBIDDEN_DEV_SHA256", fingerprint):
            with self.assertRaises(ValueError):
                data_policy.reject_forbidden_evaluation_path(copied)


if __name__ == "__main__":
    unittest.main()
