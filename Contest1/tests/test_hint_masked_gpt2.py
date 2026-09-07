"""Focused tests for the hint-masked GPT-2 helpers without model downloads."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
from typing import cast
import unittest

import pandas as pd
import torch

from contest1 import hint_masked_lib as hml
from contest1 import rerank_experiment as reranker
from contest1 import train_hint_masked as trainer_module


MLFLOW_AVAILABLE = importlib.util.find_spec("mlflow") is not None


class FakeBatch(dict):
    def to(self, device):
        return FakeBatch({key: value.to(device) for key, value in self.items()})


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 19

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return {
            " apple": [10, 11],
            " apricot": [10, 12],
            " amazing": [10, 12, 11],
            " atom": [13],
        }[text]

    def __len__(self):
        return 20

    def decode(self, token_ids, **_kwargs):
        return " " if token_ids[0] in (10, 13, 19) else "x"

    def __call__(self, contexts, **_kwargs):
        encoded = {"short": [1], "long": [2, 1]}
        rows = [encoded[context] for context in contexts]
        width = max(map(len, rows))
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, values in enumerate(rows):
            input_ids[row, -len(values) :] = torch.tensor(values)
            attention_mask[row, -len(values) :] = 1
        return FakeBatch(input_ids=input_ids, attention_mask=attention_mask)


class FakeLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.position_ids_seen: list[torch.Tensor] = []

    def forward(self, input_ids, attention_mask, position_ids, **_kwargs):
        del attention_mask
        self.position_ids_seen.append(position_ids.detach().cpu())
        logits = torch.zeros((*input_ids.shape, 20), device=input_ids.device)
        logits[input_ids == 1, 10] = 8
        logits[input_ids == 10, 11] = 1
        logits[input_ids == 10, 12] = 7
        logits[input_ids == 12, 11] = 5
        logits[input_ids == 11, 14] = 6
        logits[input_ids == 12, 14] = 2
        return SimpleNamespace(logits=logits + self.anchor * 0)


class HintMaskedGpt2Tests(unittest.TestCase):
    def test_lexicon_preserves_words_that_share_a_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            corpus.write_text(
                "apple apple apricot\napricot atom\n", encoding="utf-8"
            )
            lexicon = hml.build_training_lexicon(
                FakeTokenizer(), corpus, progress_every=0
            )

        self.assertEqual(lexicon.hint_token_word["a"][10], "apple")
        self.assertEqual(
            lexicon.hint_root_words["a"][10], ["apple", "apricot"]
        )
        self.assertEqual(lexicon.predictable_words, {"apple", "atom"})
        self.assertEqual(
            lexicon.sequence_predictable_words, {"apple", "apricot", "atom"}
        )

    def test_hidden_loss_matches_full_logits_loss(self) -> None:
        torch.manual_seed(3)
        hidden = torch.randn(2, 3, 4)
        weight = torch.randn(7, 4)
        labels = torch.tensor([[-100, 1, 2], [3, -100, 1]])
        hints = torch.tensor([[-1, 0, 0], [1, -1, 0]])
        candidates, lookup = hml.build_hint_candidate_tables(
            ["a", "b"], {"a": [1, 2], "b": [3, 4]}, 7, "cpu"
        )
        logits = hidden @ weight.transpose(0, 1)

        full = hml.hint_masked_loss(logits, labels, hints, candidates, lookup)
        restricted = hml.hint_masked_hidden_loss(
            hidden, labels, hints, candidates, lookup, weight
        )

        self.assertTrue(torch.allclose(full[0], restricted[0], atol=1e-6))
        self.assertEqual(full[1:], restricted[1:])

    def test_sequence_reranking_resolves_a_root_collision(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeLanguageModel()
        common = {
            "candidate_ids_by_hint": {"a": [10]},
            "hint_token_word": {"a": {10: "apple"}},
            "fallback_by_hint": {"a": "apple"},
            "top_k": 1,
        }

        root_only = hml.predict_top_k(
            model, tokenizer, ["short"], ["a"], **common
        )
        reranked = hml.predict_top_k(
            model,
            tokenizer,
            ["short"],
            ["a"],
            hint_root_words={"a": {10: ["apple", "apricot"]}},
            word_bpe={"apple": (10, 11), "apricot": (10, 12)},
            rerank_words_per_root=2,
            **common,
        )

        self.assertEqual(root_only, [["apple"]])
        self.assertEqual(reranked, [["apricot"]])

    def test_candidate_scoring_exposes_separate_suffix_features(self) -> None:
        candidates = hml.score_candidates(
            FakeLanguageModel(),
            FakeTokenizer(),
            ["short"],
            ["a"],
            candidate_ids_by_hint={"a": [10, 13]},
            hint_root_words={"a": {10: ["apple", "apricot"], 13: ["atom"]}},
            word_bpe={"apple": (10, 11), "apricot": (10, 12), "atom": (13,)},
            word_counts={"apple": 3, "apricot": 2, "atom": 1},
            root_beam=2,
            words_per_root=2,
        )[0]
        by_word = {candidate.word: candidate for candidate in candidates}

        self.assertEqual(by_word["apple"].root_rank, 1)
        self.assertEqual(by_word["apricot"].within_root_rank, 2)
        self.assertGreater(
            by_word["apricot"].suffix_log_probability,
            by_word["apple"].suffix_log_probability,
        )
        self.assertEqual(by_word["atom"].suffix_token_count, 0)
        self.assertEqual(by_word["apple"].word_count, 3)

    def test_batched_suffix_and_boundary_positions_handle_different_lengths(self) -> None:
        common = {
            "root_id": 10,
            "root_rank": 1,
            "root_logit": 8.0,
            "root_log_probability": -0.1,
            "root_margin": 2.0,
            "root_entropy": 0.1,
            "suffix_log_probability": 0.0,
            "boundary_log_probability": None,
            "word_count": 1,
        }
        candidates = [
            hml.CandidateScore(
                word="apple", token_ids=(10, 11), within_root_rank=1, **common
            ),
            hml.CandidateScore(
                word="amazing", token_ids=(10, 12, 12), within_root_rank=2, **common
            ),
        ]

        scored = hml.score_candidate_batch(
            FakeLanguageModel(),
            FakeTokenizer(),
            [([1], candidate) for candidate in candidates],
            batch_size=2,
            boundary_token_ids=[14],
        )

        self.assertEqual([candidate.word for candidate in scored], ["apple", "amazing"])
        self.assertEqual(scored[1].suffix_token_count, 2)
        self.assertTrue(math.isfinite(scored[1].suffix_log_probability))
        first_boundary = cast(float, scored[0].boundary_log_probability)
        second_boundary = cast(float, scored[1].boundary_log_probability)
        self.assertGreater(first_boundary, second_boundary)

    def test_rerank_score_can_apply_frequency_without_suffix_scoring(self) -> None:
        common = {
            "token_ids": (10,),
            "root_id": 10,
            "root_rank": 1,
            "root_logit": 1.0,
            "root_log_probability": -0.5,
            "root_margin": 1.0,
            "root_entropy": 0.2,
            "suffix_log_probability": 0.0,
            "boundary_log_probability": None,
        }
        rare = hml.CandidateScore(
            word="rare", within_root_rank=2, word_count=1, **common
        )
        frequent = hml.CandidateScore(
            word="frequent", within_root_rank=1, word_count=100, **common
        )

        ranked = hml.rerank_candidate_scores(
            [rare, frequent], suffix_weight=0.0, frequency_weight=0.1
        )

        self.assertEqual([candidate.word for candidate in ranked], ["frequent", "rare"])

    def test_reranked_coverage_uses_expanded_words(self) -> None:
        metrics, _, _ = hml.evaluate_model(
            FakeLanguageModel(),
            FakeTokenizer(),
            pd.DataFrame(
                [{"context": "short", "first letter": "a", "answer": "apricot"}]
            ),
            candidate_ids_by_hint={"a": [10]},
            hint_token_word={"a": {10: "apple"}},
            fallback_by_hint={"a": "apple"},
            predictable_words={"apple"},
            hint_root_words={"a": {10: ["apple", "apricot"]}},
            word_bpe={"apple": (10, 11), "apricot": (10, 12)},
            rerank_words_per_root=2,
            top_k=1,
        )

        self.assertEqual(metrics["candidate_coverage"], 1.0)

    def test_grouped_split_keeps_duplicates_and_selection_leakage_together(self) -> None:
        frame = pd.DataFrame(
            [
                {"context": "same", "first letter": "a", "answer": "apple"},
                {"context": "same", "first letter": "a", "answer": "apple"},
                {"context": "same", "first letter": "a", "answer": "apricot"},
                {"context": "other", "first letter": "a", "answer": "atom"},
                {"context": "more", "first letter": "b", "answer": "ball"},
                {"context": "last", "first letter": "b", "answer": "book"},
            ]
        )
        split = reranker.assign_grouped_splits(
            frame, selection_size=1, tune_fraction=0.5, seed=7
        )

        duplicate_splits = split.loc[split["context"] == "same", "split"].unique()
        self.assertEqual(len(duplicate_splits), 1)
        self.assertTrue(
            split.groupby("group_id")["split"].nunique().eq(1).all()
        )

    def test_heuristic_predictions_keep_collision_and_confidence_paths_separate(self) -> None:
        rows = []
        for example_id, confidence in ((1, 0.9), (2, 0.2)):
            for candidate, root_rank, within_rank, suffix in (
                ("apple", 1, 1, -5.0),
                ("apricot", 1, 2, -1.0),
                ("atom", 2, 1, -0.1),
            ):
                rows.append(
                    {
                        "example_id": example_id,
                        "group_id": str(example_id),
                        "answer": "apricot" if example_id == 1 else "atom",
                        "baseline_prediction": "apple",
                        "candidate": candidate,
                        "root_rank": root_rank,
                        "within_root_rank": within_rank,
                        "root_log_probability": math.log(confidence)
                        if root_rank == 1
                        else math.log(1 - confidence),
                        "suffix_log_probability": suffix,
                        "suffix_token_count": int(within_rank == 2),
                        "log_word_count": 0.0,
                        "boundary_log_probability": 0.0,
                    }
                )
        predictions = reranker.choose_predictions(
            pd.DataFrame(rows),
            words_per_root=2,
            suffix_weight=1.0,
            length_penalty=1.0,
            frequency_weight=0.0,
            boundary_weight=0.0,
            confidence_gate=0.5,
        )

        self.assertEqual(predictions.loc[1, "prediction"], "apricot")
        self.assertEqual(predictions.loc[2, "prediction"], "atom")

    def test_prediction_resources_restore_integer_roots_and_token_tuples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resources.json"
            path.write_text(
                json.dumps(
                    {
                        "candidate_ids_by_hint": {"a": [10]},
                        "hint_token_word": {"a": {"10": "apple"}},
                        "hint_root_words": {
                            "a": {"10": ["apple", "apricot"]}
                        },
                        "word_bpe": {
                            "apple": [10, 11],
                            "apricot": [10, 12],
                        },
                        "fallback_by_hint": {"a": "apple"},
                    }
                ),
                encoding="utf-8",
            )

            resources = hml.load_prediction_resources(path)

        self.assertEqual(resources["hint_token_word"]["a"][10], "apple")
        self.assertEqual(resources["hint_root_words"]["a"][10][1], "apricot")
        self.assertEqual(resources["word_bpe"]["apricot"], (10, 12))

    def test_empty_metric_subgroup_is_none(self) -> None:
        import pandas as pd

        metrics, _, _ = hml.evaluate_model(
            FakeLanguageModel(),
            FakeTokenizer(),
            pd.DataFrame(
                [{"context": "short", "first letter": "a", "answer": "apple"}]
            ),
            candidate_ids_by_hint={"a": [10]},
            hint_token_word={"a": {10: "apple"}},
            fallback_by_hint={"a": "apple"},
            predictable_words={"apple"},
            top_k=1,
        )
        self.assertEqual(metrics["alphanumeric_top_1_accuracy"], 1.0)
        self.assertIsNone(metrics["non_alphanumeric_top_1_accuracy"])

    def test_left_padding_uses_unshifted_position_ids(self) -> None:
        model = FakeLanguageModel()
        hml.predict_top_k(
            model,
            FakeTokenizer(),
            ["short", "long"],
            ["a", "a"],
            candidate_ids_by_hint={"a": [10]},
            hint_token_word={"a": {10: "apple"}},
            fallback_by_hint={"a": "apple"},
            top_k=1,
        )
        self.assertTrue(
            torch.equal(model.position_ids_seen[0], torch.tensor([[0, 0], [0, 1]]))
        )

    def test_rng_state_round_trip(self) -> None:
        random.seed(17)
        torch.manual_seed(17)
        state = trainer_module.capture_rng_state()
        expected_python = random.random()
        expected_torch = torch.rand(3)

        random.seed(99)
        torch.manual_seed(99)
        trainer_module.restore_rng_state(state)

        self.assertEqual(random.random(), expected_python)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))

    def test_best_checkpoint_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 2)
            best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }
            path = Path(directory) / "best_dev.pt"
            torch.save(
                {
                    "model": best_state,
                    "metrics": {"top_1_accuracy": 0.75},
                    "step": 12,
                },
                path,
            )
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10)

            trainer = trainer_module.Trainer.__new__(trainer_module.Trainer)
            trainer.args = SimpleNamespace(output_dir=Path(directory))
            trainer.device = torch.device("cpu")
            trainer.model = model
            trainer.best_dev_accuracy = 0.75
            trainer._load_best_model()

            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, best_state[name]))

    @unittest.skipUnless(MLFLOW_AVAILABLE, "MLflow is not installed")
    def test_existing_mlflow_run_is_reopened_and_logged(self) -> None:
        import mlflow
        from mlflow import MlflowClient

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracking_uri = "sqlite:///" + str(root / "mlflow.db")
            client = MlflowClient(tracking_uri=tracking_uri)
            experiment_id = client.create_experiment("resume-test")
            run_id = client.create_run(experiment_id).info.run_id
            client.set_terminated(run_id, status="FINISHED")

            trainer = trainer_module.Trainer.__new__(trainer_module.Trainer)
            trainer.args = SimpleNamespace(
                mlflow=True,
                tracking_uri=tracking_uri,
                output_dir=root,
                experiment_name="resume-test",
                run_name=None,
            )
            trainer.global_step = 7
            trainer.mlflow_failed = False
            trainer.mlflow_run_id = run_id
            trainer._start_mlflow()
            trainer._log_step_metrics(
                {
                    "step": 7,
                    "loss": 1.25,
                    "root_accuracy": 0.5,
                    "learning_rate": 1e-5,
                    "gradient_norm": 0.75,
                }
            )
            trainer._end_mlflow("KILLED")

            self.assertIsNone(mlflow.active_run())
            self.assertEqual(client.get_run(run_id).info.status, "KILLED")
            history = client.get_metric_history(run_id, "train_loss")
            self.assertEqual([(metric.step, metric.value) for metric in history], [(7, 1.25)])
            metadata = json.loads(
                (root / "mlflow_resume.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["run_id"], run_id)


if __name__ == "__main__":
    unittest.main()
