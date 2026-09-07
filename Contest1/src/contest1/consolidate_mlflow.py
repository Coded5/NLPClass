#!/usr/bin/env python3
"""Create one canonical MLflow run from cumulative local training logs."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Metric, Param

from .paths import PROJECT_ROOT


HERE = PROJECT_ROOT
DEFAULT_OUTPUT_DIR = HERE / "artifacts" / "gpt2-keyboard-script"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-name", default="hint-masked-gpt2")
    parser.add_argument("--run-name", default="gpt2-keyboard-consolidated")
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="default: sqlite:///<output-dir>/mlflow.db",
    )
    return parser.parse_args()


def atomic_text_write(text: str, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def backup_database(database: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = database.with_name(f"{database.stem}.pre-consolidation-{stamp}.db")
    source_connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    backup_connection = sqlite3.connect(backup)
    try:
        source_connection.backup(backup_connection)
    finally:
        backup_connection.close()
        source_connection.close()
    return backup


def parameter_value(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return "None"
    return str(value)


def log_metric_batches(
    client: MlflowClient, run_id: str, metrics: list[Metric], batch_size: int = 800
) -> None:
    for start in range(0, len(metrics), batch_size):
        client.log_batch(run_id, metrics=metrics[start : start + batch_size])


def history_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def latest_metric_step(client: MlflowClient, run_id: str, key: str) -> int:
    history = client.get_metric_history(run_id, key)
    return max((metric.step for metric in history), default=-1)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    history_path = output_dir / "training_history.json"
    dev_path = output_dir / "dev_curve.json"
    config_path = output_dir / "run_config.json"
    for path in (history_path, dev_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    training_history = json.loads(history_path.read_text(encoding="utf-8"))
    dev_curve = json.loads(dev_path.read_text(encoding="utf-8"))
    run_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not training_history:
        raise ValueError("training history is empty")
    digest = history_digest((history_path, dev_path, config_path))

    tracking_uri = args.tracking_uri or (
        "sqlite:///" + str((output_dir / "mlflow.db").resolve())
    )
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(args.experiment_name)
        source_runs = []
    else:
        experiment_id = experiment.experiment_id
        source_runs = [
            run
            for run in client.search_runs([experiment_id])
            if run.data.tags.get("consolidated") != "true"
        ]

    final_step = int(training_history[-1]["step"])
    source_run_ids = [run.info.run_id for run in source_runs]
    manifest_path = output_dir / "consolidated_mlflow.json"
    previous_step = -1
    existing_manifest = None
    run_id = ""
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = existing_manifest["run_id"]
        existing_run = client.get_run(run_id)
        if existing_run.data.tags.get("consolidated") != "true":
            raise ValueError("manifest points to a non-consolidated MLflow run")
        previous_step = int(existing_manifest["consolidated_through_step"])
        if previous_step > final_step:
            raise ValueError("consolidated run is ahead of local training history")
        if previous_step == final_step and existing_manifest.get("history_sha256") == digest:
            resume_metadata = {
                "run_id": run_id,
                "tracking_uri": tracking_uri,
                "experiment_name": args.experiment_name,
                "checkpoint_step": final_step,
            }
            atomic_text_write(
                json.dumps(resume_metadata, indent=2) + "\n",
                output_dir / "mlflow_resume.json",
            )
            print(json.dumps(existing_manifest, indent=2))
            return

    backup = None
    if args.tracking_uri is None:
        backup = backup_database(output_dir / "mlflow.db")

    if existing_manifest is None:
        tags = {
            "mlflow.runName": args.run_name,
            "consolidated": "true",
            "consolidated_through_step": str(final_step),
            "source_run_ids": ",".join(source_run_ids),
        }
        run = client.create_run(experiment_id, tags=tags)
        run_id = run.info.run_id
    else:
        client.set_tag(run_id, "consolidated_through_step", str(final_step))
        client.set_tag(run_id, "source_run_ids", ",".join(source_run_ids))

    try:
        if existing_manifest is None:
            parameters = [
                Param(key, parameter_value(value)) for key, value in run_config.items()
            ]
            for start in range(0, len(parameters), 100):
                client.log_batch(run_id, params=parameters[start : start + 100])

        timestamp = int(time.time() * 1000)
        cadence = max(1, int(run_config.get("log_every_steps", 10)))
        latest_train_step = latest_metric_step(client, run_id, "train_loss")
        selected_records = [
            record
            for record in training_history
            if int(record["step"]) > latest_train_step
            and (
                int(record["step"]) % cadence == 0
                or int(record["step"]) == final_step
            )
        ]
        metrics: list[Metric] = []
        for record in selected_records:
            step = int(record["step"])
            metrics.extend(
                [
                    Metric("train_loss", float(record["loss"]), timestamp, step),
                    Metric(
                        "train_root_accuracy",
                        float(record["root_accuracy"]),
                        timestamp,
                        step,
                    ),
                    Metric(
                        "learning_rate",
                        float(record["learning_rate"]),
                        timestamp,
                        step,
                    ),
                    Metric(
                        "gradient_norm",
                        float(record["gradient_norm"]),
                        timestamp,
                        step,
                    ),
                ]
            )
        log_metric_batches(client, run_id, metrics)

        dev_metrics: list[Metric] = []
        top_k = int(run_config.get("top_k", 5))
        latest_dev_step = latest_metric_step(client, run_id, "dev_top_1_accuracy")
        for record in dev_curve:
            step = int(record["step"])
            if step <= latest_dev_step:
                continue
            values = {
                "dev_top_1_accuracy": record["top_1_accuracy"],
                f"dev_top_{top_k}_accuracy": record[f"top_{top_k}_accuracy"],
                "dev_alphanumeric_top_1": record.get(
                    "alphanumeric_top_1_accuracy"
                ),
                "dev_non_alphanumeric_top_1": record.get(
                    "non_alphanumeric_top_1_accuracy"
                ),
                "dev_candidate_coverage": record.get("candidate_coverage"),
                "epoch": record.get("epoch"),
            }
            dev_metrics.extend(
                Metric(key, float(value), timestamp, step)
                for key, value in values.items()
                if value is not None
            )
        log_metric_batches(client, run_id, dev_metrics)

        manifest = {
            "run_id": run_id,
            "experiment_name": args.experiment_name,
            "tracking_uri": tracking_uri,
            "consolidated_through_step": final_step,
            "history_sha256": digest,
            "training_metric_records_added": len(selected_records),
            "dev_metric_records_total": len(dev_curve),
            "source_run_ids": source_run_ids,
            "database_backup": str(backup) if backup else None,
        }
        atomic_text_write(
            json.dumps(manifest, indent=2) + "\n", manifest_path
        )
        resume_metadata = {
            "run_id": run_id,
            "tracking_uri": tracking_uri,
            "experiment_name": args.experiment_name,
            "checkpoint_step": final_step,
        }
        atomic_text_write(
            json.dumps(resume_metadata, indent=2) + "\n",
            output_dir / "mlflow_resume.json",
        )
        for artifact in (config_path, history_path, dev_path, manifest_path):
            client.log_artifact(run_id, str(artifact))
        client.set_terminated(run_id, status="FINISHED")
    except Exception:
        client.set_terminated(run_id, status="FAILED")
        raise

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
