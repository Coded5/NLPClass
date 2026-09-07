#!/usr/bin/env python3
"""Train and use a first-character-conditioned GRU word predictor.

The corpus is expected to contain one whitespace-tokenized sentence per line.
Training examples are generated lazily, so corpus-sized lists of windows are
never held in memory. Output softmaxes contain only words matching the literal
first-character hint.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
from typing import Any, Iterator, Sequence

from .data_policy import reject_forbidden_evaluation_path
from .paths import DATA_DIR, PROJECT_ROOT, TRAIN_DIR

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader, IterableDataset, get_worker_info
except ImportError as error:  # Give CLI users a useful error while retaining the cause.
    raise ImportError(
        "neural_language_model requires PyTorch; install requirements.txt in "
        "the project virtualenv"
    ) from error


ROOT = PROJECT_ROOT
DEFAULT_CORPUS = TRAIN_DIR / "train.src.tok"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs/neural_model.pt"
DEFAULT_DEV_CSV = DATA_DIR / "devv_eval.csv"
DEFAULT_TEST_CSV = DATA_DIR / "test_set_no_answer.csv"

# NUL cannot occur in a whitespace token read by the supported text tooling.
# In particular, the literal corpus token "[UNK]" is an ordinary lexical word.
PAD_TOKEN = "\0NEURAL_PAD"
UNK_TOKEN = "\0NEURAL_UNK"
PAD_ID = 0
UNK_ID = 1
CHECKPOINT_VERSION = 2


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python and Torch, optionally requesting deterministic algorithms."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class ModelConfig:
    context_size: int = 24
    embedding_dim: int = 256
    hidden_dim: int = 384
    hint_dim: int = 32
    gru_layers: int = 1
    dropout: float = 0.1

    def validate(self) -> None:
        for name in (
            "context_size",
            "embedding_dim",
            "hidden_dim",
            "hint_dim",
            "gru_layers",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class Prediction:
    word: str
    log_probability: float


@dataclass(frozen=True)
class Evaluation:
    examples: int
    correct: int
    no_candidates: int
    answer_oov: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.examples if self.examples else 0.0


class Vocabulary:
    """Deterministic word IDs and exact first-character candidate groups."""

    def __init__(
        self,
        id_to_token: Sequence[str],
        *,
        corpus_tokens: int | None = None,
        retained_tokens: int | None = None,
    ) -> None:
        self.id_to_token = list(id_to_token)
        if self.id_to_token[:2] != [PAD_TOKEN, UNK_TOKEN]:
            raise ValueError("vocabulary must begin with the internal PAD and UNK tokens")
        if len(set(self.id_to_token)) != len(self.id_to_token):
            raise ValueError("vocabulary contains duplicate tokens")
        self.token_to_id = {token: index for index, token in enumerate(self.id_to_token)}
        groups: dict[str, list[int]] = defaultdict(list)
        for word_id, token in enumerate(self.id_to_token[2:], start=2):
            if not token:
                raise ValueError("empty lexical tokens are not supported")
            groups[token[0]].append(word_id)
        self.candidate_ids = dict(groups)
        self.output_position = [0] * len(self.id_to_token)
        for candidates in self.candidate_ids.values():
            for position, word_id in enumerate(candidates):
                self.output_position[word_id] = position
        self.hints = sorted(groups)
        self.hint_to_id = {hint: index for index, hint in enumerate(self.hints)}
        self.corpus_tokens = corpus_tokens
        self.retained_tokens = retained_tokens

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, token: str) -> int:
        return self.token_to_id.get(token, UNK_ID)

    def decode(self, word_id: int) -> str:
        return self.id_to_token[word_id]

    @classmethod
    def build(
        cls,
        corpus_path: str | Path,
        *,
        min_count: int = 2,
        max_vocab: int | None = 250_000,
        max_lines: int | None = None,
        progress_every: int = 100_000,
    ) -> Vocabulary:
        """Count tokens in one pass and assign deterministic frequency-ranked IDs."""
        if min_count < 1:
            raise ValueError("min_count must be at least 1")
        if max_vocab is not None and max_vocab < 1:
            raise ValueError("max_vocab must be at least 1")
        if max_lines is not None and max_lines < 1:
            raise ValueError("max_lines must be at least 1")
        path = Path(corpus_path)
        counts: Counter[str] = Counter()
        with path.open("r", encoding="utf-8") as corpus:
            for line_number, line in enumerate(corpus, start=1):
                if max_lines is not None and line_number > max_lines:
                    break
                tokens = line.split()
                if PAD_TOKEN in tokens or UNK_TOKEN in tokens:
                    raise ValueError(f"reserved internal token found on line {line_number}")
                counts.update(tokens)
                if progress_every and line_number % progress_every == 0:
                    print(f"vocabulary: counted {line_number:,} lines", flush=True)

        words = [item for item in counts.items() if item[1] >= min_count]
        words.sort(key=lambda item: (-item[1], item[0]))
        if max_vocab is not None:
            words = words[:max_vocab]
        if not words:
            raise ValueError("no corpus tokens satisfy the vocabulary settings")
        retained_tokens = sum(count for _, count in words)
        return cls(
            [PAD_TOKEN, UNK_TOKEN, *(word for word, _ in words)],
            corpus_tokens=sum(counts.values()),
            retained_tokens=retained_tokens,
        )


def iter_sentence_windows(
    tokens: Sequence[str], context_size: int
) -> Iterator[tuple[list[str], str, str]]:
    """Yield (context, exact first character, target) for every sentence token."""
    if context_size < 1:
        raise ValueError("context_size must be at least 1")
    for target_index, target in enumerate(tokens):
        if not target:
            continue
        start = max(0, target_index - context_size)
        yield list(tokens[start:target_index]), target[0], target


class CorpusWindowDataset(IterableDataset[tuple[list[int], int, str]]):
    """Lazily read the corpus and optionally shuffle within a bounded buffer."""

    def __init__(
        self,
        corpus_path: str | Path,
        vocabulary: Vocabulary,
        context_size: int,
        *,
        seed: int = 42,
        shuffle_buffer: int = 20_000,
        max_lines: int | None = None,
        max_examples: int | None = None,
    ) -> None:
        super().__init__()
        if shuffle_buffer < 1:
            raise ValueError("shuffle_buffer must be at least 1")
        self.corpus_path = Path(corpus_path)
        self.vocabulary = vocabulary
        self.context_size = context_size
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.max_lines = max_lines
        self.max_examples = max_examples

    def _ordered_examples(self) -> Iterator[tuple[list[int], int, str]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        global_example = 0
        with self.corpus_path.open("r", encoding="utf-8") as corpus:
            for zero_line_number, line in enumerate(corpus):
                if self.max_lines is not None and zero_line_number >= self.max_lines:
                    break
                # Without a cap, line sharding avoids duplicate tokenization. With a
                # cap, every worker walks the same prefix and emits its disjoint
                # share, making the global first-N examples independent of workers.
                if (
                    self.max_examples is None
                    and zero_line_number % worker_count != worker_id
                ):
                    continue
                for context, hint, target in iter_sentence_windows(
                    line.split(), self.context_size
                ):
                    target_id = self.vocabulary.encode(target)
                    # UNK has no literal surface form and cannot be a valid output class.
                    if target_id == UNK_ID:
                        continue
                    if self.max_examples is not None and global_example >= self.max_examples:
                        return
                    if self.max_examples is None or global_example % worker_count == worker_id:
                        yield (
                            [self.vocabulary.encode(token) for token in context],
                            target_id,
                            hint,
                        )
                    global_example += 1

    def __iter__(self) -> Iterator[tuple[list[int], int, str]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        rng = random.Random(self.seed + worker_id)
        buffer: list[tuple[list[int], int, str]] = []
        for example in self._ordered_examples():
            if len(buffer) < self.shuffle_buffer:
                buffer.append(example)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = example
        rng.shuffle(buffer)
        yield from buffer


def collate_windows(
    batch: Sequence[tuple[list[int], int, str]],
) -> tuple[Tensor, Tensor, Tensor, list[str]]:
    """Right-pad contexts; represent an empty context by one PAD time step."""
    lengths = torch.tensor([max(1, len(item[0])) for item in batch], dtype=torch.long)
    width = int(lengths.max().item())
    contexts = torch.full((len(batch), width), PAD_ID, dtype=torch.long)
    for row, (context, _, _) in enumerate(batch):
        if context:
            contexts[row, : len(context)] = torch.tensor(context, dtype=torch.long)
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    hints = [item[2] for item in batch]
    return contexts, lengths, targets, hints


class NeuralLanguageModel(nn.Module):
    """GRU context encoder with hint-conditioned, restricted output scoring."""

    def __init__(self, config: ModelConfig, vocabulary: Vocabulary) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.vocabulary = vocabulary
        self.word_embedding = nn.Embedding(
            len(vocabulary), config.embedding_dim, padding_idx=PAD_ID
        )
        self.gru = nn.GRU(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )
        self.hint_embedding = nn.Embedding(len(vocabulary.hints), config.hint_dim)
        self.fusion = nn.Linear(config.hidden_dim + config.hint_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.output_embedding = nn.Embedding(len(vocabulary), config.hidden_dim)
        self.output_bias = nn.Parameter(torch.zeros(len(vocabulary)))
        self.register_buffer(
            "output_position",
            torch.tensor(vocabulary.output_position, dtype=torch.long),
            persistent=False,
        )
        word_hint_ids = [-1] * len(vocabulary)
        for hint, candidates in vocabulary.candidate_ids.items():
            for word_id in candidates:
                word_hint_ids[word_id] = vocabulary.hint_to_id[hint]
        self.register_buffer(
            "word_hint_ids",
            torch.tensor(word_hint_ids, dtype=torch.long),
            persistent=False,
        )
        self._candidate_tensor_cache: dict[tuple[str, str], Tensor] = {}

    def encode_context(self, contexts: Tensor, lengths: Tensor) -> Tensor:
        embedded = self.word_embedding(contexts)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden[-1]

    def _hint_tensor(self, hints: Sequence[str], device: torch.device) -> Tensor:
        try:
            hint_ids = [self.vocabulary.hint_to_id[hint] for hint in hints]
        except KeyError as error:
            raise ValueError(f"hint has no vocabulary candidates: {error.args[0]!r}") from error
        return torch.tensor(hint_ids, dtype=torch.long, device=device)

    def condition(self, context_states: Tensor, hints: Sequence[str]) -> Tensor:
        hint_tensor = self._hint_tensor(hints, context_states.device)
        features = torch.cat((context_states, self.hint_embedding(hint_tensor)), dim=1)
        return self.dropout(torch.tanh(self.fusion(features)))

    def candidate_tensor(self, hint: str, device: torch.device) -> Tensor:
        key = (hint, str(device))
        cached = self._candidate_tensor_cache.get(key)
        if cached is None:
            candidates = self.vocabulary.candidate_ids.get(hint, [])
            cached = torch.tensor(candidates, dtype=torch.long, device=device)
            self._candidate_tensor_cache[key] = cached
        return cached

    def restricted_logits(self, states: Tensor, hint: str) -> tuple[Tensor, list[int]]:
        candidate_ids = self.vocabulary.candidate_ids.get(hint, [])
        if not candidate_ids:
            return states.new_empty((states.shape[0], 0)), []
        ids = self.candidate_tensor(hint, states.device)
        logits = states @ self.output_embedding(ids).T + self.output_bias[ids]
        return logits, candidate_ids

    def loss(
        self, contexts: Tensor, lengths: Tensor, targets: Tensor, hints: Sequence[str]
    ) -> tuple[Tensor, int]:
        batch_size = contexts.shape[0]
        if not (
            lengths.numel() == batch_size
            and targets.numel() == batch_size
            and len(hints) == batch_size
        ):
            raise ValueError("contexts, lengths, targets, and hints must have equal batch size")
        if not hints:
            raise ValueError("loss requires a non-empty batch")
        if int(targets.min().item()) < 2 or int(targets.max().item()) >= len(self.vocabulary):
            raise ValueError("loss targets must be lexical vocabulary IDs")
        hint_ids = self._hint_tensor(hints, targets.device)
        target_hint_ids = self.word_hint_ids[targets]
        mismatches = torch.nonzero(target_hint_ids != hint_ids, as_tuple=False)
        if mismatches.numel():
            row = int(mismatches[0].item())
            word = self.vocabulary.decode(int(targets[row].item()))
            raise ValueError(
                f"target/hint mismatch at batch row {row}: target={word!r}, hint={hints[row]!r}"
            )
        context_states = self.encode_context(contexts, lengths)
        states = self.condition(context_states, hints)
        groups: dict[str, list[int]] = defaultdict(list)
        for row, hint in enumerate(hints):
            groups[hint].append(row)

        total_loss = states.new_zeros(())
        correct = 0
        for hint, rows in groups.items():
            row_ids = torch.tensor(rows, dtype=torch.long, device=states.device)
            logits, _ = self.restricted_logits(states[row_ids], hint)
            local_targets = self.output_position[targets[row_ids]]
            total_loss = total_loss + F.cross_entropy(
                logits, local_targets, reduction="sum"
            )
            correct += int((logits.argmax(dim=1) == local_targets).sum().item())
        return total_loss / len(hints), correct


class NeuralWordPredictor:
    """Checkpoint-backed prediction API intended for CLI and later ensembles."""

    def __init__(self, model: NeuralLanguageModel, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vocabulary = model.vocabulary
        self.config = model.config

    @classmethod
    def load(
        cls, checkpoint_path: str | Path, device: str | torch.device = "cpu"
    ) -> NeuralWordPredictor:
        checkpoint = load_checkpoint(checkpoint_path)
        model = model_from_checkpoint(checkpoint)
        return cls(model, device)

    def predict(
        self,
        context: str | Sequence[str],
        first_character: str,
        *,
        top_k: int = 1,
    ) -> list[Prediction]:
        return self.predict_many([(context, first_character)], top_k=top_k)[0]

    @torch.inference_mode()
    def predict_many(
        self,
        examples: Sequence[tuple[str | Sequence[str], str]],
        *,
        top_k: int = 1,
        batch_size: int = 512,
    ) -> list[list[Prediction]]:
        if top_k < 1 or batch_size < 1:
            raise ValueError("top_k and batch_size must be at least 1")
        results: list[list[Prediction]] = []
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            encoded: list[tuple[list[int], int, str]] = []
            for context, hint in chunk:
                if len(hint) != 1:
                    raise ValueError(f"first-character hint must have length 1: {hint!r}")
                tokens = context.split() if isinstance(context, str) else list(context)
                tokens = tokens[-self.config.context_size :]
                encoded.append(
                    ([self.vocabulary.encode(token) for token in tokens], UNK_ID, hint)
                )
            contexts, lengths, _, hints = collate_windows(encoded)
            contexts = contexts.to(self.device)
            lengths = lengths.to(self.device)
            context_states = self.model.encode_context(contexts, lengths)

            chunk_results: list[list[Prediction]] = [[] for _ in chunk]
            groups: dict[str, list[int]] = defaultdict(list)
            for row, hint in enumerate(hints):
                groups[hint].append(row)
            for hint, rows in groups.items():
                if hint not in self.vocabulary.candidate_ids:
                    continue
                row_ids = torch.tensor(rows, device=self.device)
                conditioned = self.model.condition(
                    context_states[row_ids], [hint] * len(rows)
                )
                logits, candidates = self.model.restricted_logits(conditioned, hint)
                count = min(top_k, len(candidates))
                values, positions = torch.topk(
                    F.log_softmax(logits.float(), dim=1), count, dim=1
                )
                for group_row, original_row in enumerate(rows):
                    chunk_results[original_row] = [
                        Prediction(
                            self.vocabulary.decode(candidates[int(position)]),
                            float(value),
                        )
                        for value, position in zip(
                            values[group_row].cpu(), positions[group_row].cpu()
                        )
                    ]
            results.extend(chunk_results)
        return results


def _safe_torch_load(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Use restricted loading when supported; never silently fall back on unsafe data."""
    try:
        loaded = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch versions predating weights_only.
        loaded = torch.load(path, map_location=map_location)
    if not isinstance(loaded, dict):
        raise ValueError("checkpoint root must be a dictionary")
    return loaded


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = _safe_torch_load(path)
    required = {
        "format_version",
        "config",
        "id_to_token",
        "hint_to_id",
        "model_state",
        "training_state",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["format_version"] != CHECKPOINT_VERSION:
        raise ValueError("unsupported neural checkpoint format")
    if not isinstance(checkpoint["training_state"], dict):
        raise ValueError("checkpoint training_state must be a dictionary")
    return checkpoint


def model_from_checkpoint(checkpoint: dict[str, Any]) -> NeuralLanguageModel:
    vocabulary = Vocabulary(
        checkpoint["id_to_token"],
        corpus_tokens=checkpoint.get("corpus_tokens"),
        retained_tokens=checkpoint.get("retained_tokens"),
    )
    if checkpoint.get("hint_to_id") != vocabulary.hint_to_id:
        raise ValueError("checkpoint hint mapping is inconsistent with its vocabulary")
    config = ModelConfig(**checkpoint["config"])
    model = NeuralLanguageModel(config, vocabulary)
    model.load_state_dict(checkpoint["model_state"])
    return model


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict) or "python" not in state or "torch" not in state:
        raise ValueError("checkpoint contains incomplete RNG state")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def save_checkpoint(
    path: str | Path,
    model: NeuralLanguageModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    """Atomically save weights plus every mapping needed for inference."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "config": asdict(model.config),
        "id_to_token": model.vocabulary.id_to_token,
        "hint_to_id": model.vocabulary.hint_to_id,
        "corpus_tokens": model.vocabulary.corpus_tokens,
        "retained_tokens": model.vocabulary.retained_tokens,
        "model_state": model.state_dict(),
        "training_state": training_state or {},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def train(args: argparse.Namespace) -> None:
    reject_forbidden_evaluation_path(args.dev_csv)
    seed_everything(args.seed, not args.allow_nondeterministic)
    device = torch.device(_resolve_device(args.device))
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume:
        resume_checkpoint = load_checkpoint(args.resume)
        model = model_from_checkpoint(resume_checkpoint).to(device)
        vocabulary = model.vocabulary
        config = model.config
    else:
        vocabulary = Vocabulary.build(
            args.corpus,
            min_count=args.min_count,
            max_vocab=args.max_vocab,
            max_lines=args.max_lines,
            progress_every=args.progress_every,
        )
        config = ModelConfig(
            context_size=args.context_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            hint_dim=args.hint_dim,
            gru_layers=args.gru_layers,
            dropout=args.dropout,
        )
        model = NeuralLanguageModel(config, vocabulary).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    data_config = {
        "corpus": str(args.corpus.resolve()),
        "batch_size": args.batch_size,
        "max_lines": args.max_lines,
        "max_examples": args.max_examples,
        "shuffle_buffer": args.shuffle_buffer,
        "num_workers": args.num_workers,
    }
    completed_epochs = 0
    resume_epoch_examples = 0
    global_step = 0
    best_dev_accuracy = -1.0
    if resume_checkpoint is not None:
        state = resume_checkpoint["training_state"]
        required = {
            "completed_epochs",
            "epoch_examples",
            "global_step",
            "best_dev_accuracy",
            "seed",
            "data_config",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"resume checkpoint is missing training fields: {sorted(missing)}")
        if state["seed"] != args.seed:
            raise ValueError(
                f"resume seed {state['seed']} does not match --seed {args.seed}"
            )
        if state["data_config"] != data_config:
            raise ValueError(
                "resume data options differ from the checkpoint; corpus, batch size, "
                "limits, shuffle buffer, and workers must match"
            )
        if resume_checkpoint.get("optimizer_state") is None:
            raise ValueError("resume checkpoint has no optimizer state")
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        _optimizer_to(optimizer, device)
        if resume_checkpoint.get("scaler_state") is not None:
            scaler.load_state_dict(resume_checkpoint["scaler_state"])
        if "rng_state" not in resume_checkpoint:
            raise ValueError("resume checkpoint has no RNG state")
        restore_rng_state(resume_checkpoint["rng_state"])
        completed_epochs = int(state["completed_epochs"])
        resume_epoch_examples = int(state["epoch_examples"])
        global_step = int(state["global_step"])
        best_dev_accuracy = float(state["best_dev_accuracy"])
        if min(completed_epochs, resume_epoch_examples, global_step) < 0:
            raise ValueError("resume counters cannot be negative")
        if not math.isfinite(best_dev_accuracy) or not -1 <= best_dev_accuracy <= 1:
            raise ValueError("resume best_dev_accuracy must be in [-1, 1]")

    last_checkpoint = args.last_checkpoint or args.checkpoint.with_name(
        f"{args.checkpoint.stem}.last{args.checkpoint.suffix}"
    )
    corpus_tokens = vocabulary.corpus_tokens
    retained_tokens = vocabulary.retained_tokens
    if corpus_tokens is not None and retained_tokens is not None:
        skipped = corpus_tokens - retained_tokens
        coverage = retained_tokens / corpus_tokens if corpus_tokens else 0.0
        coverage_text = (
            f" retained_targets={retained_tokens:,} skipped_oov_targets={skipped:,} "
            f"coverage={coverage:.2%}"
        )
    else:
        coverage_text = " retained_targets=unknown skipped_oov_targets=unknown"
    print(
        f"training on {device}; vocabulary={len(vocabulary):,}; "
        f"hints={len(vocabulary.hints):,};{coverage_text}",
        flush=True,
    )
    dev_examples = 0
    dev_answer_oov = 0
    for rows in _csv_chunks(
        args.dev_csv, args.eval_batch_size, args.eval_limit, require_answer=True
    ):
        dev_examples += len(rows)
        dev_answer_oov += sum(
            vocabulary.encode(row.answer or "") == UNK_ID for row in rows
        )
    print(
        f"dev prerequisites: examples={dev_examples:,} answer_oov={dev_answer_oov:,}",
        flush=True,
    )
    if completed_epochs >= args.epochs:
        raise ValueError(
            f"checkpoint already completed {completed_epochs} epochs; "
            f"--epochs must be greater"
        )

    def checkpoint_state(done_epochs: int, epoch_examples: int) -> dict[str, Any]:
        return {
            "completed_epochs": done_epochs,
            "epoch_examples": epoch_examples,
            "global_step": global_step,
            "best_dev_accuracy": best_dev_accuracy,
            "seed": args.seed,
            "data_config": data_config,
        }

    for epoch in range(completed_epochs, args.epochs):
        dataset = CorpusWindowDataset(
            args.corpus,
            vocabulary,
            config.context_size,
            seed=args.seed + epoch,
            shuffle_buffer=args.shuffle_buffer,
            max_lines=args.max_lines,
            max_examples=args.max_examples,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_windows,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0
        epoch_examples = resume_epoch_examples if epoch == completed_epochs else 0
        remaining_skip = epoch_examples
        interval_start = time.monotonic()
        for contexts, lengths, targets, hints in loader:
            if remaining_skip:
                if len(hints) > remaining_skip:
                    raise ValueError(
                        "resume checkpoint epoch_examples does not align to a batch boundary"
                    )
                remaining_skip -= len(hints)
                if remaining_skip == 0:
                    interval_start = time.monotonic()
                continue
            contexts = contexts.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda" and args.amp,
            ):
                loss, correct = model.loss(contexts, lengths, targets, hints)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            count = len(hints)
            running_loss += float(loss.detach()) * count
            running_correct += correct
            running_examples += count
            epoch_examples += count
            global_step += 1
            if args.log_every and global_step % args.log_every == 0:
                elapsed = max(time.monotonic() - interval_start, 1e-9)
                rate = running_examples / elapsed
                expected = vocabulary.retained_tokens
                if args.max_examples is not None:
                    expected = min(expected, args.max_examples) if expected else args.max_examples
                remaining = max(0, expected - epoch_examples) if expected else None
                eta = f" eta={remaining / rate:.0f}s" if remaining is not None and rate else ""
                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"loss={running_loss / running_examples:.4f} "
                    f"accuracy={running_correct / running_examples:.4%} "
                    f"examples_per_second={rate:,.0f}{eta}",
                    flush=True,
                )
                running_loss = 0.0
                running_correct = 0
                running_examples = 0
                interval_start = time.monotonic()
            if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                save_checkpoint(
                    last_checkpoint,
                    model,
                    optimizer=optimizer,
                    scaler=scaler,
                    training_state=checkpoint_state(epoch, epoch_examples),
                )
        if remaining_skip:
            raise ValueError("resume checkpoint exceeds the available epoch examples")
        result = evaluate_csv(
            NeuralWordPredictor(model, device),
            args.dev_csv,
            batch_size=args.eval_batch_size,
            limit=args.eval_limit,
        )
        print(
            f"dev examples={result.examples:,} accuracy={result.accuracy:.4%} "
            f"no_candidates={result.no_candidates:,} answer_oov={result.answer_oov:,}",
            flush=True,
        )
        improved = result.accuracy > best_dev_accuracy
        if improved:
            best_dev_accuracy = result.accuracy
        final_state = checkpoint_state(epoch + 1, 0)
        save_checkpoint(
            last_checkpoint,
            model,
            optimizer=optimizer,
            scaler=scaler,
            training_state=final_state,
        )
        if improved:
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer=optimizer,
                scaler=scaler,
                training_state=final_state,
            )
            print(f"promoted best checkpoint to {args.checkpoint}", flush=True)
        resume_epoch_examples = 0


@dataclass(frozen=True)
class CsvExample:
    context: str
    hint: str
    answer: str | None
    row_number: int


def _csv_chunks(
    path: str | Path,
    chunk_size: int,
    limit: int | None = None,
    *,
    require_answer: bool = False,
) -> Iterator[list[CsvExample]]:
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("CSV chunk_size must be a positive integer")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
    ):
        raise ValueError("CSV limit must be a positive integer or None")
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"context", "first letter"}
        if require_answer:
            required.add("answer")
        if not reader.fieldnames:
            raise ValueError("CSV row 1: missing header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV row 1: duplicate column names")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV row 1: missing columns {sorted(missing)}")
        chunk: list[CsvExample] = []
        rows_read = 0
        for row in reader:
            if limit is not None and rows_read >= limit:
                break
            row_number = reader.line_num
            if None in row:
                raise ValueError(f"CSV row {row_number}: too many fields")
            if any(value is None for value in row.values()):
                raise ValueError(f"CSV row {row_number}: missing field value")
            context = row["context"]
            hint = row["first letter"]
            answer = row.get("answer")
            if len(hint) != 1:
                raise ValueError(
                    f"CSV row {row_number}: first letter must be exactly one character"
                )
            if require_answer:
                if not answer or answer.split() != [answer]:
                    raise ValueError(
                        f"CSV row {row_number}: answer must be one non-empty token"
                    )
                if answer[0] != hint:
                    raise ValueError(
                        f"CSV row {row_number}: answer {answer!r} does not match "
                        f"first letter {hint!r}"
                    )
            chunk.append(CsvExample(context, hint, answer, row_number))
            rows_read += 1
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
        if rows_read == 0:
            raise ValueError("CSV row 2: no data rows")


def evaluate_csv(
    predictor: NeuralWordPredictor,
    csv_path: str | Path,
    *,
    batch_size: int = 512,
    limit: int | None = None,
) -> Evaluation:
    reject_forbidden_evaluation_path(csv_path)
    examples = correct = no_candidates = 0
    answer_oov = 0
    for rows in _csv_chunks(csv_path, batch_size, limit, require_answer=True):
        predictions = predictor.predict_many(
            [(row.context, row.hint) for row in rows],
            batch_size=batch_size,
        )
        for row, candidates in zip(rows, predictions):
            examples += 1
            assert row.answer is not None
            if predictor.vocabulary.encode(row.answer) == UNK_ID:
                answer_oov += 1
            if not candidates:
                no_candidates += 1
            elif candidates[0].word == row.answer:
                correct += 1
    return Evaluation(examples, correct, no_candidates, answer_oov)


def predict_csv(
    predictor: NeuralWordPredictor,
    csv_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 512,
    limit: int | None = None,
) -> int:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    written = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for rows in _csv_chunks(csv_path, batch_size, limit):
                predictions = predictor.predict_many(
                    [(row.context, row.hint) for row in rows],
                    batch_size=batch_size,
                )
                for row, candidates in zip(rows, predictions):
                    # A one-character word is always a legal exact-hint fallback.
                    output.write(
                        (candidates[0].word if candidates else row.hint) + "\n"
                    )
                    written += 1
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return written


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _probability(value: str) -> float:
    parsed = _nonnegative_float(value)
    if parsed >= 1:
        raise argparse.ArgumentTypeError("must be less than 1")
    return parsed


def _device_arg(value: str) -> str:
    if value not in {"auto", "cpu", "cuda"} and not re.fullmatch(r"cuda:\d+", value):
        raise argparse.ArgumentTypeError("must be auto, cpu, cuda, or cuda:N")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train and save a checkpoint")
    train_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    train_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    train_parser.add_argument("--last-checkpoint", type=Path)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--dev-csv", type=Path, default=DEFAULT_DEV_CSV)
    train_parser.add_argument("--context-size", type=_positive_int, default=24)
    train_parser.add_argument("--embedding-dim", type=_positive_int, default=256)
    train_parser.add_argument("--hidden-dim", type=_positive_int, default=384)
    train_parser.add_argument("--hint-dim", type=_positive_int, default=32)
    train_parser.add_argument("--gru-layers", type=_positive_int, default=1)
    train_parser.add_argument("--dropout", type=_probability, default=0.1)
    train_parser.add_argument("--min-count", type=_positive_int, default=2)
    train_parser.add_argument("--max-vocab", type=_positive_int, default=250_000)
    train_parser.add_argument("--batch-size", type=_positive_int, default=512)
    train_parser.add_argument("--epochs", type=_positive_int, default=1)
    train_parser.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=_nonnegative_float, default=0.01)
    train_parser.add_argument("--clip-grad-norm", type=_positive_float, default=1.0)
    train_parser.add_argument("--shuffle-buffer", type=_positive_int, default=20_000)
    train_parser.add_argument("--num-workers", type=_nonnegative_int, default=2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--device", type=_device_arg, default="auto", help="auto, cpu, cuda, or cuda:N"
    )
    train_parser.add_argument("--no-amp", action="store_false", dest="amp")
    train_parser.add_argument("--allow-nondeterministic", action="store_true")
    train_parser.add_argument("--max-lines", type=_positive_int)
    train_parser.add_argument("--max-examples", type=_positive_int)
    train_parser.add_argument("--eval-limit", type=_positive_int)
    train_parser.add_argument("--eval-batch-size", type=_positive_int, default=512)
    train_parser.add_argument("--progress-every", type=_nonnegative_int, default=100_000)
    train_parser.add_argument("--log-every", type=_nonnegative_int, default=100)
    train_parser.add_argument("--checkpoint-every", type=_nonnegative_int, default=2_000)
    train_parser.set_defaults(func=train)

    evaluate_parser = subparsers.add_parser("evaluate", help="exact accuracy on dev CSV")
    evaluate_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    evaluate_parser.add_argument("--csv", type=Path, default=DEFAULT_DEV_CSV)
    evaluate_parser.add_argument("--device", type=_device_arg, default="auto")
    evaluate_parser.add_argument("--batch-size", type=_positive_int, default=512)
    evaluate_parser.add_argument("--limit", type=_positive_int)

    predict_parser = subparsers.add_parser("predict", help="write one test prediction per line")
    predict_parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    predict_parser.add_argument("--csv", type=Path, default=DEFAULT_TEST_CSV)
    predict_parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/neural_predictions.txt"
    )
    predict_parser.add_argument("--device", type=_device_arg, default="auto")
    predict_parser.add_argument("--batch-size", type=_positive_int, default=512)
    predict_parser.add_argument("--limit", type=_positive_int)
    return parser


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"requested device {value!r}, but CUDA is unavailable")
        if ":" in value:
            index = int(value.partition(":")[2])
            if index >= torch.cuda.device_count():
                raise ValueError(
                    f"requested device {value!r}, but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are visible"
                )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        args.func(args)
    elif args.command == "evaluate":
        predictor = NeuralWordPredictor.load(args.checkpoint, _resolve_device(args.device))
        result = evaluate_csv(predictor, args.csv, batch_size=args.batch_size, limit=args.limit)
        print(
            f"examples={result.examples:,} correct={result.correct:,} "
            f"accuracy={result.accuracy:.6%} no_candidates={result.no_candidates:,} "
            f"answer_oov={result.answer_oov:,}"
        )
    else:
        predictor = NeuralWordPredictor.load(args.checkpoint, _resolve_device(args.device))
        count = predict_csv(
            predictor,
            args.csv,
            args.output,
            batch_size=args.batch_size,
            limit=args.limit,
        )
        print(f"wrote {count:,} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
