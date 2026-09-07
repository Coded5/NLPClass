"""Model-download-free tests for the neural reranker experiment."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from contest1 import neural_rerank_experiment as experiment


class NeuralRerankerTests(unittest.TestCase):
    def lexicon(self):
        return SimpleNamespace(
            hint_root_words={"a": {10: ["apple", "apricot"], 11: ["atom"]}},
            word_counts={"apple": 10, "apricot": 5, "atom": 8},
            word_bpe={"apple": (10, 20), "apricot": (10, 21), "atom": (11, 22)},
            hint_token_word={"a": {10: "apple", 11: "atom"}},
        )

    def cache(self):
        examples = pd.DataFrame([{"example_id": 0, "context": "ctx", "first letter": "a", "answer": "apricot", "group_id": "g", "checkpoint_selection_contaminated": False}])
        choices = {"hints": {"a": [{"token_id": 10, "word": "apple"}, {"token_id": 11, "word": "atom"}]}}
        cache = SimpleNamespace(examples=examples, candidates=choices)
        cache.row = lambda _: np.log(np.array([.7, .3]))
        return cache

    def test_pool_uses_top_roots_and_frequency_words(self):
        pool = experiment.build_candidate_pool(self.cache(), self.lexicon(), roots_per_example=1, words_per_root=2)
        self.assertEqual(pool.candidate.tolist(), ["apple", "apricot"])
        self.assertEqual(pool.baseline_prediction.iloc[0], "apple")
        self.assertFalse(pool.candidate_correct.iloc[0])
        self.assertTrue(pool.candidate_correct.iloc[1])
        self.assertAlmostEqual(pool.log_word_count.iloc[0], np.log1p(10))

    def test_candidate_identity_does_not_depend_on_answer(self):
        first_cache = self.cache()
        second_cache = self.cache()
        second_cache.examples.loc[0, "answer"] = "atom"
        columns = ["candidate", "root_id", "root_rank", "within_root_rank"]

        first = experiment.build_candidate_pool(first_cache, self.lexicon())[columns]
        second = experiment.build_candidate_pool(second_cache, self.lexicon())[columns]

        pd.testing.assert_frame_equal(first, second)

    def test_pool_keeps_non_alphanumeric_words_for_alphanumeric_hint(self):
        lexicon = self.lexicon()
        lexicon.hint_root_words["a"][10].append("a-b")
        lexicon.word_counts["a-b"] = 3
        lexicon.word_bpe["a-b"] = (10, 23)
        pool = experiment.build_candidate_pool(
            self.cache(), lexicon, roots_per_example=1, words_per_root=3
        )

        self.assertIn("a-b", pool.candidate.tolist())

    def test_folds_are_deterministic_and_group_based(self):
        left = experiment.deterministic_folds(["same", "same", "other"], 5)
        right = experiment.deterministic_folds(["same", "same", "other"], 5)
        self.assertTrue(np.array_equal(left, right)); self.assertEqual(left[0], left[1])

    def test_mlp_forward_and_masked_listwise_loss(self):
        model = experiment.SharedContextMLP(4, 2, hidden=8)
        scores = model(torch.randn(2, 3, 4), torch.randn(2, 3, 4), torch.randn(2, 3, 2))
        self.assertEqual(tuple(scores.shape), (2, 3))
        loss = experiment.listwise_batch_loss(scores, torch.tensor([[0., 1., 0.], [0., 0., 0.]]), torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss))

    def test_group_rows_preserves_global_dataframe_indices(self):
        frame = pd.DataFrame(
            {"example_id": [1, 1, 2]}, index=[10, 11, 20]
        )
        groups = experiment._group_rows(frame)

        self.assertEqual(groups[0][1].tolist(), [10, 11])
        self.assertEqual(groups[1][1].tolist(), [20])

    def test_paired_metrics_report_promotions_and_grouped_interval(self):
        frame = pd.DataFrame(
            {
                "group_id": ["g1", "g2", "g3"],
                "answer": ["a", "b", "c"],
                "baseline": ["x", "b", "c"],
                "prediction": ["a", "x", "c"],
            }
        )
        metrics = experiment.paired_prediction_metrics(
            frame,
            "baseline",
            "prediction",
            samples=100,
            seed=3,
        )

        self.assertEqual(metrics["wrong_to_correct"], 1)
        self.assertEqual(metrics["correct_to_wrong"], 1)
        self.assertAlmostEqual(metrics["gain"], 0.0)
        self.assertIn("gain_ci_low", metrics)

    def test_blend_combines_aligned_logits_and_prediction_bonuses(self):
        left = pd.DataFrame(
            {
                "example_id": [0, 0],
                "candidate": ["a", "b"],
                "candidate_correct": [True, False],
                "root_rank": [1, 2],
                "within_root_rank": [1, 1],
                "score": [0.0, 1.0],
            }
        )
        right = left.copy()
        right["score"] = [1.0, 0.0]
        existing = pd.DataFrame(
            {
                "example_id": [0],
                "treatment_prediction": ["a"],
                "control_prediction": ["a"],
            }
        )

        candidates, scores = experiment.blend_candidate_scores(
            left,
            right,
            existing,
            right_weight=0.5,
            fusion_bonus=0.25,
            control_bonus=0.25,
        )

        self.assertEqual(candidates.candidate.iloc[int(scores.argmax())], "a")
        self.assertTrue(np.allclose(scores, [1.0, 0.5]))

        misaligned = right.copy()
        misaligned.loc[0, "candidate"] = "c"
        with self.assertRaisesRegex(ValueError, "not identically aligned"):
            experiment.blend_candidate_scores(
                left,
                misaligned,
                existing,
                right_weight=0.5,
                fusion_bonus=0.25,
                control_bonus=0.25,
            )

    def test_scalar_normalization_uses_fit_statistics(self):
        frame = pd.DataFrame({column: [1., 3.] for column in experiment.SCALAR_COLUMNS})
        normalizer = experiment.ScalarNormalizer().fit(frame)
        transformed = normalizer.transform(frame)
        self.assertTrue(np.allclose(transformed.mean(0), 0)); self.assertTrue(np.allclose(transformed.std(0), 1))

    def test_scalar_normalization_rejects_non_finite_values(self):
        frame = pd.DataFrame(
            {column: [1.0, 3.0] for column in experiment.SCALAR_COLUMNS}
        )
        frame.loc[0, experiment.SCALAR_COLUMNS[0]] = np.inf

        with self.assertRaises(ValueError):
            experiment.ScalarNormalizer().fit(frame)

    def test_ranking_ties_are_deterministic(self):
        frame = pd.DataFrame({"example_id": [0, 0], "candidate": ["z", "a"], "candidate_correct": [False, True], "root_rank": [1, 1], "within_root_rank": [1, 1]})
        metrics = experiment.ranking_metrics(frame, np.array([1., 1.]))
        self.assertEqual(metrics["top1_accuracy"], 1.0)

    def test_forbidden_file_rejected(self):
        with self.assertRaises(ValueError): experiment.reject_forbidden_path(Path("data/devv_test.csv"))

    def test_alignment_validation_rejects_missing_mapping(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            cache_dir = directory / "cache"
            cache_dir.mkdir()
            (cache_dir / "dummy").write_text("cache", encoding="utf-8")
            group_id = experiment.group_identifier("ctx", "a")
            pd.DataFrame(
                [
                    {
                        "example_id": 0,
                        "group_id": group_id,
                        "context": "ctx",
                        "hint": "a",
                        "answer": "a",
                        "contamination": False,
                        "baseline_prediction": "a",
                        "candidate": "a",
                        "candidate_correct": True,
                        "root_id": 1,
                        "root_rank": 1,
                        "root_log_probability": 0.0,
                        "root_margin": 0.0,
                        "root_entropy": 0.0,
                        "within_root_rank": 1,
                        "word_count": 1,
                        "log_word_count": np.log(2),
                        "root_candidate_count": 1,
                        "bpe_length": 1,
                        "char_length": 1,
                        "is_numeric": False,
                        "is_alpha": True,
                        "is_representative": True,
                        "candidate_miss": "reachable",
                    }
                ]
            ).to_parquet(directory / "candidate_pool.parquet")
            pd.DataFrame({"context_key": ["ctx"], "context_id": [0]}).to_csv(
                directory / "contexts.csv", index=False
            )
            pd.DataFrame(
                {"candidate": ["other"], "candidate_id": [0]}
            ).to_csv(directory / "words.csv", index=False)
            np.save(
                directory / "context_embeddings.npy",
                np.zeros((1, 2), dtype=np.float16),
            )
            np.save(
                directory / "candidate_embeddings.npy",
                np.zeros((1, 2), dtype=np.float16),
            )
            metadata = {
                "schema_version": experiment.SCHEMA_VERSION,
                "dev_path": "data/devv_eval.csv",
                "rows": 1,
                "examples": 1,
                "embedding_dim": 2,
                "cache_dir": str(cache_dir),
                "cache_files_sha256": {
                    "dummy": experiment.file_sha256(cache_dir / "dummy")
                },
                "prepared_files_sha256": {
                    name: experiment.file_sha256(directory / name)
                    for name in experiment.PREPARED_FILES
                },
            }
            (directory / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "absent from embedding mappings"):
                experiment._load_prepared(directory)


if __name__ == "__main__":
    unittest.main()
