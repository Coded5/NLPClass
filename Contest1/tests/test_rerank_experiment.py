"""Focused tests for checkpoint-fusion reranking without model downloads."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from contest1 import rerank_experiment as reranker


class CheckpointFusionTests(unittest.TestCase):
    @staticmethod
    def fake_cache(label: str, rows: list[np.ndarray]) -> SimpleNamespace:
        examples = pd.DataFrame(
            [
                {
                    "example_id": 0,
                    "first letter": "a",
                    "answer": "apricot",
                    "group_id": "g0",
                    "checkpoint_selection_contaminated": False,
                },
                {
                    "example_id": 1,
                    "first letter": "a",
                    "answer": "atom",
                    "group_id": "g1",
                    "checkpoint_selection_contaminated": True,
                },
            ]
        )
        cache = SimpleNamespace(
            label=label,
            metadata={
                "dev_sha256": "dev",
                "train_sha256": "train",
                "checkpoint_sha256": "base" if label == "step126" else label,
                "dev_path": "data/devv_eval.csv",
            },
            examples=examples,
            token_positions={"a": {10: 0, 11: 1}},
            candidates={
                "hints": {
                    "a": [
                        {"token_id": 10, "word": "apple"},
                        {"token_id": 11, "word": "atom"},
                    ]
                },
                "word_roots": {"apple": 10, "apricot": 10, "atom": 11},
                "word_frequencies": {"apple": 9, "apricot": 3, "atom": 5},
            },
        )
        cache.row = lambda index: rows[index]
        return cache

    @staticmethod
    def candidates() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "example_id": 0,
                    "group_id": "g0",
                    "split": "locked",
                    "hint": "a",
                    "answer": "apricot",
                    "baseline_prediction": "apple",
                    "candidate": "apple",
                    "candidate_correct": False,
                    "root_id": 10,
                    "root_rank": 1,
                    "within_root_rank": 1,
                },
                {
                    "example_id": 0,
                    "group_id": "g0",
                    "split": "locked",
                    "hint": "a",
                    "answer": "apricot",
                    "baseline_prediction": "apple",
                    "candidate": "apricot",
                    "candidate_correct": True,
                    "root_id": 10,
                    "root_rank": 1,
                    "within_root_rank": 2,
                },
                {
                    "example_id": 0,
                    "group_id": "g0",
                    "split": "locked",
                    "hint": "a",
                    "answer": "apricot",
                    "baseline_prediction": "apple",
                    "candidate": "atom",
                    "candidate_correct": False,
                    "root_id": 11,
                    "root_rank": 2,
                    "within_root_rank": 1,
                },
                {
                    "example_id": 1,
                    "group_id": "g1",
                    "split": "contaminated",
                    "hint": "a",
                    "answer": "atom",
                    "baseline_prediction": "atom",
                    "candidate": "atom",
                    "candidate_correct": True,
                    "root_id": 11,
                    "root_rank": 1,
                    "within_root_rank": 1,
                },
            ]
        )

    def test_checkpoint_features_map_by_example_hint_and_root(self) -> None:
        candidates = self.candidates()
        cache = self.fake_cache(
            "step258",
            [np.log(np.array([0.25, 0.75])), np.log(np.array([0.60, 0.40]))],
        )
        sidecar, names = reranker.build_checkpoint_root_features(
            candidates, {"step258": cache}
        )
        merged = reranker.merge_checkpoint_root_features(
            candidates, sidecar, list(names.values())
        )
        column = names["step258"]

        collision = merged.loc[merged["example_id"].eq(0) & merged["root_id"].eq(10)]
        self.assertEqual(collision[column].nunique(), 1)
        self.assertAlmostEqual(collision[column].iloc[0], math.log(0.25))
        atom = merged.loc[merged["example_id"].eq(0) & merged["root_id"].eq(11)]
        self.assertAlmostEqual(atom[column].iloc[0], math.log(0.75))

    def test_fusion_validation_checks_alignment_and_contamination(self) -> None:
        candidates = self.candidates()
        base = self.fake_cache(
            "step126", [np.array([-1.0, -2.0]), np.array([-2.0, -1.0])]
        )
        metadata = {
            "dev_sha256": "dev",
            "train_sha256": "train",
            "checkpoint_sha256": "base",
        }
        representatives = reranker.validate_fusion_inputs(
            candidates, metadata, {"step126": base}, "step126"
        )
        self.assertEqual(len(representatives), 2)

        broken = candidates.copy()
        broken.loc[broken["example_id"].eq(1), "split"] = "locked"
        with self.assertRaisesRegex(ValueError, "contaminated"):
            reranker.validate_fusion_inputs(
                broken, metadata, {"step126": base}, "step126"
            )

    def test_fusion_validation_checks_treatment_order_and_unions_contamination(self) -> None:
        candidates = self.candidates()
        base = self.fake_cache(
            "step126", [np.array([-1.0, -2.0]), np.array([-2.0, -1.0])]
        )
        treatment = self.fake_cache(
            "step258", [np.array([-1.5, -0.5]), np.array([-0.5, -1.5])]
        )
        treatment.examples.loc[0, "checkpoint_selection_contaminated"] = True
        metadata = {
            "dev_sha256": "dev",
            "train_sha256": "train",
            "checkpoint_sha256": "base",
        }
        representatives = reranker.validate_fusion_inputs(
            candidates,
            metadata,
            {"step126": base, "step258": treatment},
            "step126",
        )
        self.assertTrue(representatives["fusion_contaminated"].all())

        treatment.examples = treatment.examples.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "physically ordered"):
            reranker.validate_fusion_inputs(
                candidates,
                metadata,
                {"step126": base, "step258": treatment},
                "step126",
            )

    def test_ranking_metrics_include_top5_and_reciprocal_rank(self) -> None:
        candidates = self.candidates().loc[lambda frame: frame["example_id"].eq(0)]
        metrics = reranker.ranking_metrics(candidates, np.array([0.9, 0.8, 1.0]))

        self.assertEqual(metrics["candidate_oracle_coverage"], 1.0)
        self.assertEqual(metrics["top5_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mean_reciprocal_rank"], 1 / 3)

    def test_ranker_features_replace_infinite_margins(self) -> None:
        frame = pd.DataFrame({"root_margin": [np.inf, -np.inf, 1.0]})
        reranker.sanitize_ranker_features(frame, ["root_margin"])

        self.assertEqual(
            frame["root_margin"].tolist(),
            [reranker.FINITE_SCORE_SENTINEL, -reranker.FINITE_SCORE_SENTINEL, 1.0],
        )

    def test_forbidden_dev_file_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden"):
            reranker.reject_forbidden_dev(Path("data/devv_test.csv"))


if __name__ == "__main__":
    unittest.main()
