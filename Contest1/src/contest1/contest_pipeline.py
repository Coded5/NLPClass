"""CSV evaluation and submission helpers for the n-gram predictor."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .data_policy import reject_forbidden_evaluation_path
from .n_gram_generator import NGramWordPredictor
from .paths import DATA_DIR, PROJECT_ROOT


@dataclass(frozen=True)
class ContestRow:
    context: str
    hint: str
    answer: str | None = None


@dataclass(frozen=True)
class ContestMetrics:
    rows: int
    covered: int
    correct: int
    top_k_correct: int
    backoff_levels: dict[int | None, int]

    @property
    def accuracy(self) -> float:
        return self.correct / self.rows if self.rows else 0.0


def iter_contest_rows(
    path: str | Path, *, require_answers: bool
) -> Iterator[ContestRow]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"context", "first letter"}
        if require_answers:
            required.add("answer")
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, 2):
            context = row["context"]
            hint = row["first letter"]
            answer = row.get("answer") if require_answers else None
            if len(hint) != 1 or hint.isspace():
                raise ValueError(f"invalid hint on CSV line {line_number}")
            if require_answers and not answer:
                raise ValueError(f"missing answer on CSV line {line_number}")
            yield ContestRow(context=context, hint=hint, answer=answer)


def evaluate_dev(
    model_path: str | Path,
    dev_path: str | Path,
    *,
    effective_order: int | None = None,
    top_k: int = 5,
) -> ContestMetrics:
    reject_forbidden_evaluation_path(dev_path)
    rows = covered = correct = top_k_correct = 0
    backoff_levels: Counter[int | None] = Counter()
    with NGramWordPredictor(model_path) as predictor:
        for row in iter_contest_rows(dev_path, require_answers=True):
            result = predictor.predict_with_hint_result(
                row.context,
                row.hint,
                top_k,
                effective_order=effective_order,
            )
            predictions = [prediction.word for prediction in result.predictions]
            rows += 1
            covered += bool(predictions)
            correct += bool(predictions and predictions[0] == row.answer)
            top_k_correct += row.answer in predictions
            backoff_levels[result.context_order] += 1
    return ContestMetrics(
        rows=rows,
        covered=covered,
        correct=correct,
        top_k_correct=top_k_correct,
        backoff_levels=dict(backoff_levels),
    )


def generate_test_output(
    model_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
    *,
    effective_order: int | None = None,
) -> int:
    predictions: list[str] = []
    with NGramWordPredictor(model_path) as predictor:
        for row in iter_contest_rows(test_path, require_answers=False):
            result = predictor.predict_with_hint_result(
                row.context,
                row.hint,
                1,
                effective_order=effective_order,
            )
            if not result.predictions:
                raise ValueError(f"no candidate available for hint {row.hint!r}")
            predictions.append(result.predictions[0].word)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    validate_output(test_path, output_path)
    return len(predictions)


def validate_output(test_path: str | Path, output_path: str | Path) -> int:
    rows = list(iter_contest_rows(test_path, require_answers=False))
    predictions = Path(output_path).read_text(encoding="utf-8").splitlines()
    if len(predictions) != len(rows):
        raise ValueError(
            f"expected {len(rows)} predictions, found {len(predictions)}"
        )
    for line_number, (row, prediction) in enumerate(zip(rows, predictions), 1):
        if (
            not prediction
            or any(character.isspace() for character in prediction)
            or not prediction.startswith(row.hint)
        ):
            raise ValueError(f"invalid prediction on line {line_number}")
    return len(predictions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--dev", type=Path, default=DATA_DIR / "devv_eval.csv")
    evaluate.add_argument("--effective-order", type=int)
    evaluate.add_argument("--top-k", type=int, default=5)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--model", type=Path, required=True)
    generate.add_argument(
        "--test", type=Path, default=DATA_DIR / "test_set_no_answer.csv"
    )
    generate.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs/test_set_pred.txt"
    )
    generate.add_argument("--effective-order", type=int)
    validate = subparsers.add_parser("validate")
    validate.add_argument(
        "--test", type=Path, default=DATA_DIR / "test_set_no_answer.csv"
    )
    validate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate":
        result = evaluate_dev(
            args.model,
            args.dev,
            effective_order=args.effective_order,
            top_k=args.top_k,
        )
        print(
            f"rows={result.rows:,} covered={result.covered:,} "
            f"top1={result.accuracy:.4%} top{args.top_k}="
            f"{result.top_k_correct / result.rows:.4%}"
        )
    elif args.command == "generate":
        count = generate_test_output(
            args.model,
            args.test,
            args.output,
            effective_order=args.effective_order,
        )
        print(f"wrote {count:,} predictions to {args.output}")
    else:
        print(f"validated {validate_output(args.test, args.output):,} predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
