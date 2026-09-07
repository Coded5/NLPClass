"""Focused tests for the neural language model; no dependency downloads needed."""

from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import random
import tempfile
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed; neural tests skipped")
class NeuralLanguageModelTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch

        torch.manual_seed(7)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.corpus = self.root / "corpus.txt"
        self.corpus.write_text(
            "Alpha apple 123 !bang [UNK]\n"
            "apple Alpha 123 ?question\n"
            "Alpha apricot 123 !bang\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _model(self):
        from contest1.neural_language_model import ModelConfig, NeuralLanguageModel, Vocabulary

        vocabulary = Vocabulary.build(self.corpus, min_count=1, progress_every=0)
        model = NeuralLanguageModel(
            ModelConfig(context_size=3, embedding_dim=8, hidden_dim=12, hint_dim=4),
            vocabulary,
        )
        return model, vocabulary

    def test_vocabulary_is_deterministic_and_literal_unk_is_lexical(self) -> None:
        from contest1.neural_language_model import UNK_ID, Vocabulary

        first = Vocabulary.build(self.corpus, min_count=1, progress_every=0)
        second = Vocabulary.build(self.corpus, min_count=1, progress_every=0)
        self.assertEqual(first.id_to_token, second.id_to_token)
        self.assertNotEqual(first.encode("[UNK]"), UNK_ID)
        self.assertEqual(first.encode("missing"), UNK_ID)
        self.assertIn(first.encode("Alpha"), first.candidate_ids["A"])
        self.assertIn(first.encode("123"), first.candidate_ids["1"])
        self.assertIn(first.encode("!bang"), first.candidate_ids["!"])
        self.assertEqual(first.corpus_tokens, 13)
        self.assertEqual(first.retained_tokens, 13)

    def test_windows_preserve_exact_hints_and_bound_context(self) -> None:
        from contest1.neural_language_model import iter_sentence_windows

        windows = list(iter_sentence_windows(["one", "TWO", "?", "3rd"], 2))
        self.assertEqual(windows[0], ([], "o", "one"))
        self.assertEqual(windows[-1], (["TWO", "?"], "3", "3rd"))

    def test_global_max_examples_with_multiple_workers(self) -> None:
        from torch.utils.data import DataLoader
        from contest1.neural_language_model import CorpusWindowDataset, Vocabulary, collate_windows

        vocabulary = Vocabulary.build(self.corpus, min_count=1, progress_every=0)

        def collect(workers: int) -> Counter[tuple[tuple[int, ...], int, str]]:
            dataset = CorpusWindowDataset(
                self.corpus,
                vocabulary,
                3,
                seed=19,
                shuffle_buffer=2,
                max_examples=7,
            )
            loader = DataLoader(
                dataset,
                batch_size=3,
                num_workers=workers,
                collate_fn=collate_windows,
            )
            examples: Counter[tuple[tuple[int, ...], int, str]] = Counter()
            for contexts, lengths, targets, hints in loader:
                for row, hint in enumerate(hints):
                    length = int(lengths[row])
                    context = tuple(int(item) for item in contexts[row, :length])
                    examples[(context, int(targets[row]), hint)] += 1
            return examples

        single_worker = collect(0)
        multiple_workers = collect(2)
        self.assertEqual(sum(single_worker.values()), 7)
        self.assertEqual(multiple_workers, single_worker)

    def test_cpu_backward_update_restricted_cache_and_invalid_hint(self) -> None:
        import torch
        from contest1.neural_language_model import CorpusWindowDataset, collate_windows

        model, vocabulary = self._model()
        dataset = CorpusWindowDataset(
            self.corpus, vocabulary, 3, shuffle_buffer=2, max_examples=6
        )
        contexts, lengths, targets, hints = collate_windows(list(dataset))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        before = model.fusion.weight.detach().clone()
        loss, correct = model.loss(contexts, lengths, targets, hints)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(correct, 0)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.fusion.weight.detach()))

        state = model.condition(model.encode_context(contexts[:1], lengths[:1]), [hints[0]])
        model.restricted_logits(state, hints[0])
        first = model.candidate_tensor(hints[0], state.device)
        model.restricted_logits(state, hints[0])
        second = model.candidate_tensor(hints[0], state.device)
        self.assertEqual(first.data_ptr(), second.data_ptr())

        wrong_hint = "A" if hints[0] != "A" else "a"
        with self.assertRaisesRegex(ValueError, "target/hint mismatch"):
            model.loss(contexts[:1], lengths[:1], targets[:1], [wrong_hint])

    def test_checkpoint_exact_round_trip_and_resume_state(self) -> None:
        import torch
        from contest1.neural_language_model import (
            CorpusWindowDataset,
            NeuralWordPredictor,
            collate_windows,
            load_checkpoint,
            model_from_checkpoint,
            restore_rng_state,
            save_checkpoint,
        )

        model, vocabulary = self._model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch = list(
            CorpusWindowDataset(
                self.corpus, vocabulary, 3, shuffle_buffer=2, max_examples=4
            )
        )
        contexts, lengths, targets, hints = collate_windows(batch)
        loss, _ = model.loss(contexts, lengths, targets, hints)
        loss.backward()
        optimizer.step()
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        state = {
            "completed_epochs": 1,
            "epoch_examples": 6,
            "global_step": 3,
            "best_dev_accuracy": 0.25,
            "seed": 7,
            "data_config": {"batch_size": 2},
        }
        random.seed(31)
        checkpoint_path = self.root / "state" / "model.last.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            scaler=scaler,
            training_state=state,
        )
        checkpoint = load_checkpoint(checkpoint_path)
        restored = model_from_checkpoint(checkpoint)
        self.assertEqual(checkpoint["training_state"], state)
        self.assertIsNotNone(checkpoint["optimizer_state"])
        self.assertTrue(checkpoint["optimizer_state"]["state"])
        self.assertIsNotNone(checkpoint["scaler_state"])
        self.assertIn("python", checkpoint["rng_state"])
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, restored.state_dict()[name]), name)
        expected_random = random.random()
        restore_rng_state(checkpoint["rng_state"])
        self.assertEqual(random.random(), expected_random)
        predictor = NeuralWordPredictor.load(checkpoint_path)
        predictions = predictor.predict("Alpha apple", "1", top_k=2)
        self.assertTrue(predictions)
        self.assertTrue(all(item.word.startswith("1") for item in predictions))
        self.assertEqual(predictor.predict("Alpha", "Z"), [])
        self.assertEqual(predictor.vocabulary.id_to_token, vocabulary.id_to_token)

    def test_csv_evaluation_prediction_and_atomic_failure(self) -> None:
        import torch
        from contest1.neural_language_model import (
            NeuralWordPredictor,
            evaluate_csv,
            predict_csv,
        )

        model, vocabulary = self._model()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.output_bias[vocabulary.encode("Alpha")] = 10
            model.output_bias[vocabulary.encode("apple")] = 10
        predictor = NeuralWordPredictor(model, "cpu")
        dev = self.root / "dev.csv"
        dev.write_text(
            "context,first letter,answer\n"
            "apple,A,Alpha\n"
            "Alpha,a,absent\n",
            encoding="utf-8",
        )
        result = evaluate_csv(predictor, dev, batch_size=1)
        self.assertEqual(result.examples, 2)
        self.assertEqual(result.correct, 1)
        self.assertEqual(result.answer_oov, 1)

        test_csv = self.root / "test.csv"
        test_csv.write_text(
            "context,first letter\n"
            "apple,A\n"
            "Alpha,Z\n",
            encoding="utf-8",
        )
        output = self.root / "nested" / "predictions.txt"
        self.assertEqual(predict_csv(predictor, test_csv, output, batch_size=1), 2)
        self.assertEqual(output.read_text(encoding="utf-8"), "Alpha\nZ\n")

        invalid = self.root / "invalid.csv"
        invalid.write_text(
            "context,first letter\n"
            "apple,A\n"
            "Alpha,too-long\n",
            encoding="utf-8",
        )
        output.write_text("preserve me\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "CSV row 3"):
            predict_csv(predictor, invalid, output, batch_size=1)
        self.assertEqual(output.read_text(encoding="utf-8"), "preserve me\n")

    def test_csv_validation_and_invalid_answer_hint(self) -> None:
        from contest1.neural_language_model import NeuralWordPredictor, evaluate_csv, predict_csv

        model, _ = self._model()
        predictor = NeuralWordPredictor(model, "cpu")
        invalid = self.root / "bad_dev.csv"
        invalid.write_text(
            "context,first letter,answer\nctx,A,apple\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "CSV row 2.*does not match"):
            evaluate_csv(predictor, invalid)
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            predict_csv(predictor, invalid, self.root / "out.txt", batch_size=0)
        with self.assertRaisesRegex(ValueError, "limit"):
            evaluate_csv(predictor, invalid, limit=0)
        duplicate = self.root / "duplicate.csv"
        duplicate.write_text(
            "context,first letter,first letter\nctx,A,A\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "CSV row 1: duplicate"):
            predict_csv(predictor, duplicate, self.root / "duplicate.txt")


if __name__ == "__main__":
    unittest.main()
