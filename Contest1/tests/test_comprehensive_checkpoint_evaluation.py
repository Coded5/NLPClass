"""Unit tests for cache analysis; no checkpoint or model downloads."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from contest1 import comprehensive_checkpoint_evaluation as evaluation


class ComprehensiveEvaluationTests(unittest.TestCase):
    @staticmethod
    def fake_cache() -> SimpleNamespace:
        candidates = [
            {"token_id": 10, "word": "alpha", "frequency": 10, "bpe_length": 1},
            {"token_id": 11, "word": "atom", "frequency": 8, "bpe_length": 1},
            {"token_id": 12, "word": "apex", "frequency": 2, "bpe_length": 2},
            {"token_id": 13, "word": "amber", "frequency": 1, "bpe_length": 2},
            {"token_id": 14, "word": "ark", "frequency": 1, "bpe_length": 1},
        ]
        examples = pd.DataFrame(
            [
                {
                    "example_id": 0,
                    "context": "one two",
                    "first letter": "a",
                    "answer": "alpine",
                    "group_id": "g0",
                    "checkpoint_selection_contaminated": False,
                },
                {
                    "example_id": 1,
                    "context": "one three",
                    "first letter": "a",
                    "answer": "atom",
                    "group_id": "g1",
                    "checkpoint_selection_contaminated": False,
                },
                {
                    "example_id": 2,
                    "context": "one four",
                    "first letter": "a",
                    "answer": "apex",
                    "group_id": "g2",
                    "checkpoint_selection_contaminated": False,
                },
            ]
        )
        score_rows = [
            np.log(np.array([0.60, 0.20, 0.10, 0.07, 0.03])),
            np.log(np.array([0.55, 0.25, 0.10, 0.06, 0.04])),
            np.log(np.array([0.50, 0.25, 0.15, 0.06, 0.04])),
        ]
        cache = SimpleNamespace(
            label="fake",
            metadata={"checkpoint_step": 7},
            examples=examples,
            candidates={
                "hints": {"a": candidates},
                "word_frequencies": {
                    "alpha": 10,
                    "alpine": 20,
                    "atom": 8,
                    "apex": 2,
                    "amber": 1,
                    "ark": 1,
                },
                "word_roots": {
                    "alpha": 10,
                    "alpine": 10,
                    "atom": 11,
                    "apex": 12,
                    "amber": 13,
                    "ark": 14,
                },
            },
            word_positions={"a": {row["word"]: i for i, row in enumerate(candidates)}},
            token_positions={"a": {row["token_id"]: i for i, row in enumerate(candidates)}},
        )
        cache.row = lambda index: score_rows[index]
        return cache

    def test_forbidden_test_file_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden"):
            evaluation.reject_forbidden_dev(Path("data/devv_test.csv"))

    def test_rank_and_entropy_use_restricted_probabilities(self) -> None:
        scores = np.log(np.array([0.1, 0.6, 0.2, 0.1]))
        self.assertEqual(evaluation.ranked_positions(scores, 3).tolist(), [1, 2, 0])
        expected = -sum(probability * math.log(probability) for probability in (0.1, 0.6, 0.2, 0.1))
        self.assertAlmostEqual(evaluation.entropy_from_log_probs(scores), expected)

    def test_grouped_split_is_stable_and_keeps_duplicates_together(self) -> None:
        groups = ["same", "other", "same"]
        first = evaluation.assign_grouped_split(groups)
        second = evaluation.assign_grouped_split(groups)
        self.assertEqual(first, second)
        self.assertEqual(first[0], first[2])
        contaminated = evaluation.assign_grouped_split(groups, [True, False, True])
        self.assertEqual(contaminated[0], "contaminated")
        self.assertEqual(contaminated[2], "contaminated")

    def test_targeted_ngram_counts_only_training_text(self) -> None:
        rows = pd.DataFrame(
            [
                {
                    "previous_two_word": "a",
                    "previous_word": "b",
                    "candidate": "cat",
                },
                {
                    "previous_two_word": "x",
                    "previous_word": "b",
                    "candidate": "dog",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "train.txt"
            corpus.write_text("a b cat\na b cat\nx b dog\na b dog\n", encoding="utf-8")
            evaluation.add_targeted_ngram_counts(rows, corpus)
        self.assertEqual(rows["bigram_count"].tolist(), [2, 2])
        self.assertEqual(rows["trigram_count"].tolist(), [2, 1])

    def test_transition_metrics_reports_promotions_breakages_and_net(self) -> None:
        frame = pd.DataFrame(
            {
                "answer": ["a", "b", "c", "d"],
                "baseline_prediction": ["x", "b", "c", "x"],
                "reranked": ["a", "x", "c", "d"],
            }
        )
        metrics = evaluation.transition_metrics(frame, "reranked")
        self.assertEqual(metrics["promotion"], 2)
        self.assertEqual(metrics["breakage"], 1)
        self.assertEqual(metrics["net"], 1)
        self.assertEqual(metrics["promotion_denominator"], 2)
        self.assertEqual(metrics["promotion_rate"], 1.0)

    def test_detailed_predictions_distinguish_collision_root_and_word_rank(self) -> None:
        details = evaluation.collect_predictions(self.fake_cache())
        collision = details.iloc[0]
        rank_two = details.iloc[1]

        self.assertEqual(collision["top1_word"], "alpha")
        self.assertAlmostEqual(collision["top1_log_probability"], math.log(0.6))
        self.assertEqual(collision["true_training_frequency"], 20)
        self.assertTrue(collision["root_collision"])
        self.assertTrue(pd.isna(collision["true_word_rank"]))
        self.assertEqual(collision["true_root_rank"], 1)
        self.assertTrue(pd.isna(collision["true_score"]))
        self.assertAlmostEqual(collision["true_root_score"], math.log(0.6))
        self.assertTrue(pd.isna(collision["top1_minus_true_score_margin"]))
        self.assertEqual(rank_two["true_word_rank"], 2)
        self.assertEqual(rank_two["true_root_rank"], 2)
        self.assertEqual(rank_two["checkpoint"], "fake")
        self.assertIn("alpha", rank_two["top5_predictions"])

    def test_per_letter_uses_target_distribution_and_all_word_training_mode(self) -> None:
        cache = self.fake_cache()
        details = evaluation.collect_predictions(cache)
        metrics = evaluation.per_letter_metrics(details, cache).iloc[0]

        expected_entropy = -(3 * (1 / 3) * math.log(1 / 3))
        self.assertAlmostEqual(metrics["target_entropy"], expected_entropy)
        self.assertEqual(metrics["unique_target_words"], 3)
        self.assertEqual(metrics["most_common_target"], "alpine")
        self.assertEqual(metrics["most_common_target_count"], 1)
        self.assertAlmostEqual(metrics["top5_target_cumulative_share"], 1.0)
        self.assertEqual(metrics["training_mode"], "alpine")
        self.assertAlmostEqual(metrics["mean_target_training_frequency"], 10.0)
        self.assertEqual(metrics["median_target_training_frequency"], 8.0)

    def test_rank_diagnostics_include_distribution_letters_and_confusions(self) -> None:
        details = evaluation.collect_predictions(self.fake_cache())
        diagnostics = evaluation.rank2_to_5_diagnostics(details)

        self.assertEqual(diagnostics["total"], 2)
        self.assertAlmostEqual(diagnostics["percentage_of_all_examples"], 2 / 3)
        self.assertEqual(
            [row["rank"] for row in diagnostics["rank_distribution"]], [2, 3]
        )
        self.assertEqual(diagnostics["per_letter"][0]["rank2_to_5"], 2)
        self.assertEqual(diagnostics["confusions"].iloc[0]["prediction"], "alpha")
        self.assertLessEqual(
            diagnostics["smallest_margins"].iloc[0][
                "top1_minus_true_score_margin"
            ],
            diagnostics["largest_margins"].iloc[0][
                "top1_minus_true_score_margin"
            ],
        )

    def test_generic_weighted_scores_and_primary_comparison_rates(self) -> None:
        left = self.fake_cache()
        right = self.fake_cache()
        right_rows = [
            np.log(np.array([0.10, 0.60, 0.15, 0.10, 0.05])),
            np.log(np.array([0.10, 0.60, 0.15, 0.10, 0.05])),
            np.log(np.array([0.10, 0.20, 0.60, 0.05, 0.05])),
        ]
        right.row = lambda index: right_rows[index]
        combined = list(evaluation.weighted_score_rows([left, right], [1, 3]))
        np.testing.assert_allclose(
            combined[0], 0.25 * left.row(0) + 0.75 * right.row(0)
        )

        primary = evaluation.collect_predictions(left)
        evaluated = evaluation.collect_predictions(right)
        metrics = evaluation.comparison_metrics(evaluated, primary)
        self.assertEqual(metrics["improvements"], 2)
        self.assertEqual(metrics["breakages"], 0)
        self.assertEqual(metrics["primary_rank2_to_5_denominator"], 2)
        self.assertEqual(metrics["primary_rank2_to_5_promotions"], 2)

        identity = evaluation.comparison_metrics(primary, primary)
        self.assertEqual(identity["primary_rank2_to_5_denominator"], 2)
        self.assertEqual(identity["primary_rank2_to_5_promotions"], 0)
        self.assertEqual(metrics["primary_rank2_to_5_promotion_rate"], 1.0)

    def test_reranker_rows_include_positive_and_group_size(self) -> None:
        cache = self.fake_cache()
        details = evaluation.collect_predictions(cache)
        rows = evaluation.top5_reranker_rows(
            details, (cache.row(index) for index in range(3)), cache
        )

        collision = rows.loc[rows["example_id"] == 0]
        reachable = rows.loc[rows["example_id"] == 1]
        self.assertFalse(collision["group_has_positive"].any())
        self.assertTrue(reachable["group_has_positive"].all())
        self.assertTrue(rows["ranking_group_size"].eq(5).all())

    def test_requested_plots_are_separate_files(self) -> None:
        cache = self.fake_cache()
        details = evaluation.collect_predictions(cache)
        per_letter = evaluation.per_letter_metrics(details, cache)
        metrics = evaluation.prediction_metrics(details)
        ensemble = pd.DataFrame(
            [
                {
                    "ensemble": "alpha=0.5",
                    "top1_accuracy": metrics["top1_accuracy"],
                    "top5_accuracy": metrics["top5_accuracy"],
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = evaluation.make_plots(
                Path(directory),
                details,
                per_letter,
                {"fake": {"metrics": metrics}},
                ensemble,
            )
            names = {Path(path).name for path in paths}
            self.assertEqual(
                names,
                {
                    "top1_by_letter.png",
                    "top5_by_letter.png",
                    "target_entropy_vs_accuracy.png",
                    "examples_vs_accuracy.png",
                    "true_word_score_margin_distribution.png",
                    "checkpoint_ensemble_accuracy_comparison.png",
                },
            )
            self.assertTrue(
                all((Path(directory) / path).is_file() for path in paths)
            )

    def test_interpretation_has_requested_evidence_categories(self) -> None:
        cache = self.fake_cache()
        details = evaluation.collect_predictions(cache)
        per_letter = evaluation.per_letter_metrics(details, cache)
        rank_diagnostics = evaluation.rank2_to_5_diagnostics(details)
        interpretation, recommendations = evaluation.evidence_interpretation(
            details,
            evaluation.prediction_metrics(details),
            per_letter,
            evaluation.frequency_metrics(details, cache),
            rank_diagnostics,
            None,
            {},
        )
        self.assertEqual(
            set(interpretation),
            {"ranking", "modeling", "data", "rare", "high-entropy", "instability"},
        )
        self.assertEqual(len(recommendations), 5)


if __name__ == "__main__":
    unittest.main()
