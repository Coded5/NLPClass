import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import ollama
from sklearn.metrics import classification_report


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "financial-news-data" / "financial-news-test.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
PROMPT_PATH = SCRIPT_DIR / "prompt.txt"

LABELS = (
    "Analyst Update",
    "Company | Product News",
    "Currencies",
    "Dividend",
    "Earnings",
    "Energy | Oil",
    "Fed | Central Banks",
    "Financials",
    "General News | Opinion",
    "Gold | Metals | Materials",
    "IPO",
    "Legal | Regulation",
    "M&A | Investments",
    "Macro",
    "Markets",
    "Personnel Change",
    "Politics",
    "Stock Commentary",
    "Stock Movement",
    "Treasuries | Corporate Debt",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot financial-news classification with a local Ollama model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--host", help="Ollama host URL; defaults to OLLAMA_HOST or localhost.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument(
        "--limit",
        type=int,
        help="Classify only the first N rows. Intended for smoke testing.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard any compatible checkpoint and classify from the beginning.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_retries < 1:
        parser.error("--max-retries must be at least 1")
    if args.retry_delay < 0:
        parser.error("--retry-delay cannot be negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    return args


def load_rows(path, limit=None):
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not {"text", "label"}.issubset(reader.fieldnames):
            raise ValueError("Input CSV must contain 'text' and 'label' columns")

        rows = []
        for row_id, row in enumerate(reader):
            if limit is not None and len(rows) >= limit:
                break

            text = row["text"].strip()
            label = row["label"].strip()
            if not text:
                raise ValueError(f"Row {row_id} has empty text")
            if label not in LABELS:
                raise ValueError(f"Row {row_id} has unknown label: {label!r}")
            rows.append({"id": row_id, "text": text, "label": label})

    if not rows:
        raise ValueError("Input CSV contains no data rows")
    return rows


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def response_schema(batch_size):
    return {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "minItems": batch_size,
                "maxItems": batch_size,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "label": {"type": "string", "enum": list(LABELS)},
                    },
                    "required": ["id", "label"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["predictions"],
        "additionalProperties": False,
    }


def parse_predictions(content, batch):
    parsed = json.loads(content)
    items = parsed.get("predictions") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        raise ValueError("Response does not contain a predictions list")

    expected_ids = {row["id"] for row in batch}
    predictions = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("A prediction is not an object")
        row_id = item.get("id")
        label = item.get("label")
        if isinstance(row_id, bool) or not isinstance(row_id, int):
            raise ValueError(f"Invalid prediction id: {row_id!r}")
        if row_id in predictions:
            raise ValueError(f"Duplicate prediction id: {row_id}")
        if row_id not in expected_ids:
            raise ValueError(f"Unexpected prediction id: {row_id}")
        if label not in LABELS:
            raise ValueError(f"Invalid prediction label: {label!r}")
        predictions[row_id] = label

    if set(predictions) != expected_ids:
        missing = sorted(expected_ids - set(predictions))
        raise ValueError(f"Response is missing prediction ids: {missing}")
    return predictions


class OllamaZeroShotClassifier:
    def __init__(self, model, prompt, host=None, max_retries=3, retry_delay=2.0):
        self.model = model
        self.prompt = prompt
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def classify(self, batch):
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = [
                    {"id": row["id"], "text": row["text"]}
                    for row in batch
                ]
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {
                            "role": "user",
                            "content": "Classify every item in this JSON array:\n"
                            + json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                    format=response_schema(len(batch)),
                    options={
                        "temperature": 0,
                        "seed": 42,
                        "num_ctx": 8192,
                        "num_predict": max(256, len(batch) * 40),
                    },
                    keep_alive="30m",
                )
                return parse_predictions(response.message.content, batch)
            except Exception as error:
                last_error = error
                print(
                    f"Batch attempt {attempt}/{self.max_retries} failed: {error}",
                    file=sys.stderr,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        if len(batch) == 1:
            raise RuntimeError(
                f"Could not classify row {batch[0]['id']} after retries"
            ) from last_error

        midpoint = len(batch) // 2
        print(
            f"Splitting failed batch of {len(batch)} rows into smaller batches.",
            file=sys.stderr,
        )
        return self.classify(batch[:midpoint]) | self.classify(batch[midpoint:])


def checkpoint_metadata(args, rows, input_hash, prompt_hash):
    return {
        "model": args.model,
        "input_sha256": input_hash,
        "prompt_sha256": prompt_hash,
        "row_count": len(rows),
    }


def load_checkpoint(path, metadata, restart):
    if restart or not path.exists():
        return {}

    with path.open(encoding="utf-8") as file:
        checkpoint = json.load(file)
    if checkpoint.get("metadata") != metadata:
        raise ValueError(
            f"Checkpoint {path} was created with different input, prompt, or model. "
            "Use --restart to replace it."
        )

    predictions = checkpoint.get("predictions", {})
    if not isinstance(predictions, dict):
        raise ValueError(f"Checkpoint {path} has invalid predictions")

    restored = {}
    for row_id, label in predictions.items():
        numeric_id = int(row_id)
        if numeric_id < 0 or numeric_id >= metadata["row_count"]:
            raise ValueError(f"Checkpoint contains invalid row id: {row_id}")
        if label not in LABELS:
            raise ValueError(f"Checkpoint contains invalid label: {label!r}")
        restored[numeric_id] = label
    return restored


def atomic_write_json(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def save_checkpoint(path, metadata, predictions):
    atomic_write_json(
        path,
        {
            "metadata": metadata,
            "predictions": {
                str(row_id): predictions[row_id] for row_id in sorted(predictions)
            },
        },
    )


def write_predictions(path, rows, predictions):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["id", "text", "label", "predicted_label", "correct"],
        )
        writer.writeheader()
        for row in rows:
            prediction = predictions[row["id"]]
            writer.writerow(
                {
                    **row,
                    "predicted_label": prediction,
                    "correct": prediction == row["label"],
                }
            )
    temporary_path.replace(path)


def write_report(output_dir, rows, predictions):
    y_true = [row["label"] for row in rows]
    y_pred = [predictions[row["id"]] for row in rows]
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(LABELS),
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )

    (output_dir / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    atomic_write_json(output_dir / "classification_report.json", report_dict)
    return report_text


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    rows = load_rows(input_path, args.limit)
    metadata = checkpoint_metadata(
        args,
        rows,
        file_sha256(input_path),
        hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )
    checkpoint_path = output_dir / "checkpoint.json"
    predictions = load_checkpoint(checkpoint_path, metadata, args.restart)
    if predictions:
        print(f"Restored {len(predictions)} predictions from {checkpoint_path}")

    classifier = OllamaZeroShotClassifier(
        model=args.model,
        prompt=prompt,
        host=args.host,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    pending_rows = [row for row in rows if row["id"] not in predictions]
    for start in range(0, len(pending_rows), args.batch_size):
        batch = pending_rows[start : start + args.batch_size]
        predictions.update(classifier.classify(batch))
        save_checkpoint(checkpoint_path, metadata, predictions)
        print(f"Classified {len(predictions)}/{len(rows)} rows", flush=True)

    if len(predictions) != len(rows):
        raise RuntimeError("Prediction count does not match input row count")

    predictions_path = output_dir / "predictions.csv"
    write_predictions(predictions_path, rows, predictions)
    report = write_report(output_dir, rows, predictions)
    print(f"\nPredictions: {predictions_path}")
    print(f"Reports: {output_dir / 'classification_report.txt'}")
    print(f"         {output_dir / 'classification_report.json'}\n")
    print(report)


if __name__ == "__main__":
    main()
