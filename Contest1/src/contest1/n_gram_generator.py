#!/usr/bin/env python3
"""Train and use a backoff n-gram word predictor.

The corpus is read one sentence per line and is expected to be tokenized
already. Models are stored in SQLite so training does not require the whole
703 MB training set to fit in memory.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
from itertools import chain, islice
import math
import os
from pathlib import Path
import random
import sqlite3
import struct
import sys
import tempfile
from typing import Iterable, Sequence

from .paths import PROJECT_ROOT, TRAIN_DIR


DEFAULT_CORPUS = TRAIN_DIR / "train.src.tok"
DEFAULT_MODEL = PROJECT_ROOT / "outputs/ngram_model.sqlite3"
START_TOKEN = "<s>"
END_TOKEN = "</s>"


@dataclass(frozen=True)
class Prediction:
    word: str
    count: int
    probability: float


@dataclass(frozen=True)
class ConstrainedPrediction:
    predictions: tuple[Prediction, ...]
    context_order: int | None


@dataclass(frozen=True)
class Evaluation:
    sentences: int
    tokens: int
    correct: int
    out_of_vocabulary: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.tokens


def _encode_context(context: Sequence[int]) -> bytes:
    if not context:
        return b""
    return struct.pack(f"<{len(context)}I", *context)


def _validate_positive(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def _validate_holdout_fraction(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("holdout_fraction must be finite and in [0, 1)")


def _is_held_out(line: str, fraction: float, seed: int) -> bool:
    if fraction == 0:
        return False
    digest = hashlib.blake2b(digest_size=8, person=b"ngram-holdout")
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(line.encode("utf-8"))
    value = int.from_bytes(digest.digest(), byteorder="big")
    return value < int(fraction * (1 << 64))


class NGramWordPredictor:
    """A thread-confined, disk-backed model with shorter-context backoff."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model does not exist: {self.model_path}")

        model_uri = f"{self.model_path.resolve().as_uri()}?mode=ro"
        self._connection = sqlite3.connect(model_uri, uri=True)
        try:
            metadata = dict(
                self._connection.execute("SELECT key, value FROM metadata")
            )
            if metadata.get("ready") != "1":
                raise ValueError(f"model is incomplete: {self.model_path}")
            self.order = int(metadata["order"])
            self.start_id = int(metadata["start_id"])
            self.end_id = int(metadata["end_id"])
        except (sqlite3.Error, KeyError, ValueError):
            self._connection.close()
            raise

        self.metadata = metadata

        self._token_cache: OrderedDict[str, int | None] = OrderedDict(
            ((START_TOKEN, self.start_id), (END_TOKEN, self.end_id))
        )
        self._candidate_cache: OrderedDict[
            tuple[bytes, str | None, bool, int], tuple[tuple[str, int], ...]
        ] = OrderedDict()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> NGramWordPredictor:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @classmethod
    def train(
        cls,
        corpus_path: str | Path,
        model_path: str | Path,
        *,
        order: int = 3,
        min_count: int = 1,
        batch_size: int = 250_000,
        max_lines: int | None = None,
        progress_every: int = 100_000,
        holdout_fraction: float = 0,
        holdout_seed: int = 42,
    ) -> None:
        """Train a model, replacing model_path only after training succeeds."""
        _validate_positive(order, "order")
        _validate_positive(min_count, "min_count")
        _validate_positive(batch_size, "batch_size")
        if max_lines is not None:
            _validate_positive(max_lines, "max_lines")
        if progress_every < 0:
            raise ValueError("progress_every cannot be negative")
        _validate_holdout_fraction(holdout_fraction)

        corpus_path = Path(corpus_path)
        model_path = Path(model_path)
        if not corpus_path.is_file():
            raise FileNotFoundError(f"training corpus does not exist: {corpus_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if corpus_path.resolve() == model_path.resolve():
            raise ValueError("the model path cannot be the training corpus")

        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{model_path.name}.",
            suffix=".tmp",
            dir=model_path.parent,
        )
        os.close(temporary_descriptor)
        temporary_path = Path(temporary_name)
        temporary_files = [
            temporary_path,
            Path(f"{temporary_path}-wal"),
            Path(f"{temporary_path}-shm"),
        ]

        connection = sqlite3.connect(temporary_path)
        try:
            cls._create_schema(connection)
            vocabulary = {START_TOKEN: 1, END_TOKEN: 2}
            connection.executemany(
                "INSERT INTO vocabulary(id, token) VALUES (?, ?)",
                ((1, START_TOKEN), (2, END_TOKEN)),
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("ready", "0"),
                    ("order", str(order)),
                    ("start_id", "1"),
                    ("end_id", "2"),
                ),
            )
            connection.commit()

            counts: Counter[tuple[tuple[int, ...], int]] = Counter()
            pending_events = 0
            sentence_count = 0
            token_count = 0
            next_word_id = 3

            with corpus_path.open("r", encoding="utf-8") as corpus:
                lines: Iterable[str] = (
                    islice(corpus, max_lines) if max_lines is not None else corpus
                )
                for line_number, line in enumerate(lines, start=1):
                    if _is_held_out(line, holdout_fraction, holdout_seed):
                        continue

                    word_ids: list[int] = []
                    new_words: list[tuple[int, str]] = []
                    for token in line.split():
                        if token in (START_TOKEN, END_TOKEN):
                            raise ValueError(
                                f"reserved token {token!r} found on line {line_number}"
                            )
                        word_id = vocabulary.get(token)
                        if word_id is None:
                            word_id = next_word_id
                            next_word_id += 1
                            vocabulary[token] = word_id
                            new_words.append((word_id, token))
                        word_ids.append(word_id)

                    if not word_ids:
                        continue

                    sentence_count += 1
                    if new_words:
                        connection.executemany(
                            "INSERT INTO vocabulary(id, token) VALUES (?, ?)",
                            new_words,
                        )

                    token_count += len(word_ids)
                    history = [1] * (order - 1)
                    for target_id in chain(word_ids, (2,)):
                        max_context = min(order - 1, len(history))
                        for context_size in range(max_context + 1):
                            context = (
                                tuple(history[-context_size:])
                                if context_size
                                else ()
                            )
                            counts[(context, target_id)] += 1
                        pending_events += max_context + 1
                        history.append(target_id)

                        if pending_events >= batch_size:
                            cls._flush_counts(connection, counts)
                            counts.clear()
                            pending_events = 0

                    if progress_every and line_number % progress_every == 0:
                        print(
                            f"Processed {sentence_count:,} sentences and "
                            f"{token_count:,} tokens",
                            file=sys.stderr,
                        )

            if counts:
                cls._flush_counts(connection, counts)

            if sentence_count == 0:
                raise ValueError("the training corpus contains no sentences")

            if min_count > 1:
                connection.execute(
                    "DELETE FROM ngrams WHERE length(context) > 0 AND count < ?",
                    (min_count,),
                )

            connection.execute(
                "CREATE INDEX ngrams_by_count "
                "ON ngrams(context, count DESC, word_id)"
            )
            metadata = {
                "ready": "1",
                "sentences": str(sentence_count),
                "tokens": str(token_count),
                "vocabulary_size": str(len(vocabulary) - 2),
                "min_count": str(min_count),
                "holdout_fraction": repr(holdout_fraction),
                "holdout_seed": str(holdout_seed),
            }
            connection.executemany(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                metadata.items(),
            )
            connection.commit()
            checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is not None and checkpoint[0] != 0:
                raise sqlite3.DatabaseError("could not checkpoint the trained model")
            journal_mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
            if journal_mode is None or journal_mode[0].lower() != "delete":
                raise sqlite3.DatabaseError("could not finalize the trained model")
            if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise sqlite3.DatabaseError("the trained model failed its integrity check")
        except Exception:
            connection.close()
            for path in temporary_files:
                path.unlink(missing_ok=True)
            raise
        else:
            connection.close()
            os.replace(temporary_path, model_path)

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-65536;
            PRAGMA temp_store=FILE;

            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE vocabulary (
                id INTEGER PRIMARY KEY,
                token TEXT NOT NULL UNIQUE
            );

            CREATE TABLE ngrams (
                context BLOB NOT NULL,
                word_id INTEGER NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (context, word_id)
            ) WITHOUT ROWID;
            """
        )

    @staticmethod
    def _flush_counts(
        connection: sqlite3.Connection,
        counts: Counter[tuple[tuple[int, ...], int]],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO ngrams(context, word_id, count) VALUES (?, ?, ?)
            ON CONFLICT(context, word_id)
            DO UPDATE SET count = count + excluded.count
            """,
            (
                (_encode_context(context), word_id, count)
                for (context, word_id), count in counts.items()
            ),
        )
        connection.commit()

    @staticmethod
    def _tokens(value: str | Sequence[str]) -> list[str]:
        return value.split() if isinstance(value, str) else list(value)

    def _token_id(self, token: str) -> int | None:
        if token in self._token_cache:
            self._token_cache.move_to_end(token)
            return self._token_cache[token]

        row = self._connection.execute(
            "SELECT id FROM vocabulary WHERE token = ?", (token,)
        ).fetchone()
        word_id = row[0] if row is not None else None
        self._token_cache[token] = word_id
        if len(self._token_cache) > 100_000:
            self._token_cache.popitem(last=False)
        return word_id

    def _contexts(
        self,
        tokens: Sequence[str],
        effective_order: int | None = None,
    ) -> Iterable[bytes]:
        order = self.order if effective_order is None else effective_order
        _validate_positive(order, "effective_order")
        if order > self.order:
            raise ValueError(
                f"effective_order ({order}) exceeds model order ({self.order})"
            )
        if order == 1:
            yield b""
            return

        context_ids: list[int | None] = [
            self._token_id(token) for token in tokens[-(order - 1) :]
        ]
        if len(context_ids) < order - 1:
            context_ids[:0] = [self.start_id] * (
                order - 1 - len(context_ids)
            )

        for size in range(order - 1, -1, -1):
            suffix = context_ids[-size:] if size else []
            if all(word_id is not None for word_id in suffix):
                yield _encode_context(
                    tuple(word_id for word_id in suffix if word_id is not None)
                )

    def _candidate_rows(
        self,
        tokens: Sequence[str],
        limit: int,
        *,
        include_end: bool,
        hint: str | None = None,
        effective_order: int | None = None,
    ) -> tuple[bytes, list[tuple[str, int]]]:
        excluded_ids = (self.start_id,) if include_end else (
            self.start_id,
            self.end_id,
        )
        placeholders = ", ".join("?" for _ in excluded_ids)
        hint_clause = "AND substr(vocabulary.token, 1, 1) = ?" if hint is not None else ""
        query = f"""
            SELECT vocabulary.token, ngrams.count
            FROM ngrams
            JOIN vocabulary ON vocabulary.id = ngrams.word_id
            WHERE ngrams.context = ?
              AND ngrams.word_id NOT IN ({placeholders})
              {hint_clause}
            ORDER BY ngrams.count DESC, vocabulary.token ASC, ngrams.word_id ASC
            LIMIT ?
        """
        for context in self._contexts(tokens, effective_order):
            cache_key = (context, hint, include_end, limit)
            cached = self._candidate_cache.get(cache_key)
            if cached is None:
                parameters: tuple[object, ...] = (context, *excluded_ids)
                if hint is not None:
                    parameters += (hint,)
                rows = tuple(
                    self._connection.execute(query, (*parameters, limit))
                )
                self._candidate_cache[cache_key] = rows
                if len(self._candidate_cache) > 20_000:
                    self._candidate_cache.popitem(last=False)
            else:
                self._candidate_cache.move_to_end(cache_key)
                rows = cached

            if rows:
                return context, list(rows)
            if hint is not None:
                continue

            # Generic prediction historically stops at the first seen context,
            # even when its only continuation is filtered out.
            exists = self._connection.execute(
                """
                SELECT 1 FROM ngrams
                WHERE context = ? AND word_id != ?
                LIMIT 1
                """,
                (context, self.start_id),
            ).fetchone()
            if exists is not None:
                return context, []
        return b"", []

    def predict_with_hint_result(
        self,
        context: str | Sequence[str],
        hint: str,
        top_k: int = 5,
        *,
        effective_order: int | None = None,
    ) -> ConstrainedPrediction:
        """Predict words whose first character is exactly ``hint``."""
        _validate_positive(top_k, "top_k")
        if len(hint) != 1 or hint.isspace():
            raise ValueError("hint must be exactly one non-whitespace character")
        context_key, rows = self._candidate_rows(
            self._tokens(context),
            top_k,
            include_end=False,
            hint=hint,
            effective_order=effective_order,
        )
        if not rows:
            return ConstrainedPrediction((), None)

        total = self._connection.execute(
            """
            SELECT COALESCE(SUM(count), 0)
            FROM ngrams
            WHERE context = ? AND word_id != ?
            """,
            (context_key, self.start_id),
        ).fetchone()[0]
        predictions = tuple(
            Prediction(word=word, count=count, probability=count / total)
            for word, count in rows
        )
        return ConstrainedPrediction(predictions, len(context_key) // 4)

    def predict_with_hint(
        self,
        context: str | Sequence[str],
        hint: str,
        top_k: int = 5,
        *,
        effective_order: int | None = None,
    ) -> list[Prediction]:
        """Return literal first-character-constrained predictions."""
        return list(
            self.predict_with_hint_result(
                context, hint, top_k, effective_order=effective_order
            ).predictions
        )

    predict_constrained = predict_with_hint

    def predict(
        self,
        context: str | Sequence[str],
        top_k: int = 5,
    ) -> list[Prediction]:
        """Return the most frequent next words and their backoff probabilities."""
        _validate_positive(top_k, "top_k")
        context_key, rows = self._candidate_rows(
            self._tokens(context), top_k, include_end=False
        )
        if not rows:
            return []

        total = self._connection.execute(
            """
            SELECT COALESCE(SUM(count), 0)
            FROM ngrams
            WHERE context = ? AND word_id != ?
            """,
            (context_key, self.start_id),
        ).fetchone()[0]
        return [
            Prediction(word=word, count=count, probability=count / total)
            for word, count in rows
        ]

    def evaluate(
        self,
        corpus_path: str | Path,
        *,
        holdout_fraction: float | None = None,
        holdout_seed: int | None = None,
        max_lines: int | None = None,
        progress_every: int = 100_000,
    ) -> Evaluation:
        """Measure top-1 next-word accuracy on a deterministic holdout."""
        corpus_path = Path(corpus_path)
        if not corpus_path.is_file():
            raise FileNotFoundError(f"evaluation corpus does not exist: {corpus_path}")
        if max_lines is not None:
            _validate_positive(max_lines, "max_lines")
        if progress_every < 0:
            raise ValueError("progress_every cannot be negative")

        if holdout_fraction is None:
            holdout_fraction = float(self.metadata.get("holdout_fraction", "0"))
        if holdout_seed is None:
            holdout_seed = int(self.metadata.get("holdout_seed", "42"))
        _validate_holdout_fraction(holdout_fraction)

        sentence_count = 0
        token_count = 0
        correct_count = 0
        out_of_vocabulary = 0
        with corpus_path.open("r", encoding="utf-8") as corpus:
            lines: Iterable[str] = (
                islice(corpus, max_lines) if max_lines is not None else corpus
            )
            for line_number, line in enumerate(lines, start=1):
                if holdout_fraction and not _is_held_out(
                    line, holdout_fraction, holdout_seed
                ):
                    continue

                sentence = line.split()
                if not sentence:
                    continue
                sentence_count += 1
                history: list[str] = []
                for target in sentence:
                    _, candidates = self._candidate_rows(
                        history, 1, include_end=False
                    )
                    if candidates and candidates[0][0] == target:
                        correct_count += 1
                    if self._token_id(target) is None:
                        out_of_vocabulary += 1
                    token_count += 1
                    history.append(target)

                if progress_every and line_number % progress_every == 0:
                    print(
                        f"Evaluated {sentence_count:,} held-out sentences and "
                        f"{token_count:,} tokens",
                        file=sys.stderr,
                    )

        if token_count == 0:
            raise ValueError("the evaluation selection contains no tokens")
        return Evaluation(
            sentences=sentence_count,
            tokens=token_count,
            correct=correct_count,
            out_of_vocabulary=out_of_vocabulary,
        )

    def generate(
        self,
        seed_text: str | Sequence[str] = "",
        *,
        max_new_words: int = 30,
        temperature: float = 1.0,
        candidate_limit: int = 100,
        random_seed: int | None = None,
    ) -> str:
        """Generate text with temperature-controlled top-k sampling."""
        if max_new_words < 0:
            raise ValueError("max_new_words cannot be negative")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and cannot be negative")
        _validate_positive(candidate_limit, "candidate_limit")

        tokens = self._tokens(seed_text)
        rng = random.Random(random_seed)
        for _ in range(max_new_words):
            _, candidates = self._candidate_rows(
                tokens, candidate_limit, include_end=True
            )
            if not candidates:
                break

            if temperature == 0:
                next_word = candidates[0][0]
            else:
                scaled_logs = [math.log(count) / temperature for _, count in candidates]
                largest_log = max(scaled_logs)
                weights = [math.exp(value - largest_log) for value in scaled_logs]
                next_word = rng.choices(
                    [word for word, _ in candidates],
                    weights=weights,
                    k=1,
                )[0]

            if next_word == END_TOKEN:
                break
            tokens.append(next_word)

        return " ".join(tokens)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run a disk-backed n-gram word predictor.",
        epilog=(
            "Examples:\n"
            "  contest1-ngram train --order 3 --min-count 2\n"
            "  contest1-ngram predict 'the united states' --top-k 10\n"
            "  contest1-ngram generate 'the united states' --words 30"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a new model")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_CORPUS)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    train_parser.add_argument("-n", "--order", type=int, default=3)
    train_parser.add_argument("--min-count", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=250_000)
    train_parser.add_argument(
        "--holdout",
        type=float,
        default=0,
        help="exclude a deterministic fraction of sentences for evaluation",
    )
    train_parser.add_argument("--holdout-seed", type=int, default=42)
    train_parser.add_argument(
        "--max-lines",
        type=int,
        help="train on only the first N lines (useful for experiments)",
    )
    train_parser.add_argument(
        "--quiet", action="store_true", help="disable training progress output"
    )

    predict_parser = subparsers.add_parser(
        "predict", help="show likely next words"
    )
    predict_parser.add_argument("context", help="tokenized context text")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict_parser.add_argument("-k", "--top-k", type=int, default=5)
    predict_parser.add_argument(
        "--hint", help="restrict predictions to this literal first character"
    )
    predict_parser.add_argument(
        "--effective-order", type=int, help="maximum order to use for prediction"
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="measure held-out top-1 next-word accuracy"
    )
    evaluate_parser.add_argument("--data", type=Path, default=DEFAULT_CORPUS)
    evaluate_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    evaluate_parser.add_argument(
        "--holdout",
        type=float,
        help="fraction selected for evaluation (defaults to the model setting)",
    )
    evaluate_parser.add_argument("--holdout-seed", type=int)
    evaluate_parser.add_argument("--max-lines", type=int)
    evaluate_parser.add_argument(
        "--quiet", action="store_true", help="disable evaluation progress output"
    )

    generate_parser = subparsers.add_parser("generate", help="generate text")
    generate_parser.add_argument("seed_text", nargs="?", default="")
    generate_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    generate_parser.add_argument("-w", "--words", type=int, default=30)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    generate_parser.add_argument(
        "--candidate-limit",
        type=int,
        default=100,
        help="sample from only the top N next-token candidates",
    )
    generate_parser.add_argument("--seed", type=int, help="random number seed")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            NGramWordPredictor.train(
                args.data,
                args.model,
                order=args.order,
                min_count=args.min_count,
                batch_size=args.batch_size,
                max_lines=args.max_lines,
                progress_every=0 if args.quiet else 100_000,
                holdout_fraction=args.holdout,
                holdout_seed=args.holdout_seed,
            )
            print(f"Model saved to {args.model}")
            return 0

        with NGramWordPredictor(args.model) as predictor:
            if args.command == "predict":
                predictions = (
                    predictor.predict_with_hint(
                        args.context,
                        args.hint,
                        args.top_k,
                        effective_order=args.effective_order,
                    )
                    if args.hint is not None
                    else predictor.predict(args.context, args.top_k)
                )
                for prediction in predictions:
                    print(
                        f"{prediction.word}\t{prediction.probability:.6f}"
                        f"\t({prediction.count})"
                    )
            elif args.command == "generate":
                print(
                    predictor.generate(
                        args.seed_text,
                        max_new_words=args.words,
                        temperature=args.temperature,
                        candidate_limit=args.candidate_limit,
                        random_seed=args.seed,
                    )
                )
            else:
                evaluation = predictor.evaluate(
                    args.data,
                    holdout_fraction=args.holdout,
                    holdout_seed=args.holdout_seed,
                    max_lines=args.max_lines,
                    progress_every=0 if args.quiet else 100_000,
                )
                print(f"Sentences: {evaluation.sentences:,}")
                print(f"Tokens: {evaluation.tokens:,}")
                print(f"Correct: {evaluation.correct:,}")
                print(f"Out-of-vocabulary: {evaluation.out_of_vocabulary:,}")
                print(f"Top-1 accuracy: {evaluation.accuracy:.4%}")
        return 0
    except (FileNotFoundError, OSError, sqlite3.Error, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
