#!/usr/bin/env python3
"""Resilient, monitorable training for the hint-masked GPT-2 keyboard model.

Example
-------
Smoke run::

    contest1-train-gpt2 --max-train-steps 12 --max-train-lines 6000

Full single-epoch run::

    contest1-train-gpt2 --max-train-steps -1

Double-descent experiment (cosine schedule, multiple epochs)::

    contest1-train-gpt2 --double-descent --total-epochs 8 --max-train-steps -1

Resilience
----------
* Atomic checkpointing (temp file + ``os.replace``); ``--resume`` continues from
  the latest checkpoint by rewinding the deterministic data stream.
* ``SIGINT``/``SIGTERM`` finish the current optimizer update, save a checkpoint,
  and exit cleanly, so Ctrl-C never loses more than a few updates.

Monitoring
----------
* ``tqdm`` progress bar with loss / masked-root-accuracy / LR / dev accuracy.
* Optional MLflow tracking (``--mlflow``; default on when importable) plus local
  JSON/CSV logs that are always written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from . import hint_masked_lib as hml
from .data_policy import reject_forbidden_evaluation_path
from .paths import PROJECT_ROOT

try:
    import mlflow
    from mlflow import MlflowClient
except ImportError:  # MLflow is optional; local logs always work.
    mlflow = None
    MlflowClient = None


HERE = PROJECT_ROOT
DEFAULT_TRAIN_PATH = HERE / "train" / "train.src.tok"
DEFAULT_DEV_PATH = HERE / "data" / "devv_eval.csv"
DEFAULT_TEST_PATH = HERE / "data" / "test_set_no_answer.csv"
DEFAULT_OUTPUT_DIR = HERE / "artifacts" / "gpt2-keyboard-script"
EVALUATION_VERSION = 2
CHECKPOINT_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--dev-path", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_TEST_PATH,
        help="contest test CSV (context, first letter) for test_set_pred.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="gpt2")

    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--double-descent",
        action="store_true",
        help="cosine LR over --total-epochs epochs (epoch-wise double-descent probe)",
    )
    parser.add_argument("--total-epochs", type=int, default=8)
    parser.add_argument("--max-train-lines", type=int, default=None)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=2000,
        help="optimizer updates to run; -1 uses the full schedule "
        "(steps per epoch x epochs or total_epochs)",
    )

    parser.add_argument("--eval-every-steps", type=int, default=2000)
    parser.add_argument("--eval-subset-size", type=int, default=2000)
    parser.add_argument(
        "--eval-at-epoch-end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also evaluate at exact optimizer-step epoch boundaries",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--rerank-words-per-root",
        type=int,
        default=1,
        help="full-word candidates scored per selected BPE root; 1 keeps fast root decoding",
    )
    parser.add_argument(
        "--rerank-root-beam",
        type=int,
        default=None,
        help="number of BPE roots considered by full-word reranking (default: top-k)",
    )
    parser.add_argument("--rerank-batch-size", type=int, default=64)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-legacy-resume",
        action="store_true",
        help="resume a pre-v2 checkpoint without exact RNG/config guarantees",
    )
    parser.add_argument(
        "--allow-schedule-change",
        action="store_true",
        help="allow changing only the planned step/epoch horizon when resuming",
    )
    parser.add_argument("--resume-every-steps", type=int, default=1000)
    parser.add_argument(
        "--keep-last-n",
        type=int,
        default=1,
        help="keep this many numbered resume checkpoints (1 keeps only the "
        "latest atomic snapshot)",
    )
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help="ignore an incompatible resume checkpoint and start over",
    )

    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-fp16", action="store_true", help="disable mixed-precision autocast"
    )

    parser.add_argument(
        "--no-final-dev-eval",
        dest="final_dev_eval",
        action="store_false",
        default=True,
        help="skip the full dev evaluation at the end (handy for smoke tests)",
    )
    parser.add_argument("--mlflow", dest="mlflow", action="store_true", default=None)
    parser.add_argument("--no-mlflow", dest="mlflow", action="store_false")
    parser.set_defaults(mlflow=True)
    parser.add_argument("--experiment-name", default="hint-masked-gpt2")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--mlflow-run-id",
        default=None,
        help="resume metrics in an existing MLflow run (also read from output-dir/mlflow_run_id.txt)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: sqlite:///<output-dir>/mlflow.db)",
    )
    parser.add_argument(
        "--mlflow-log-artifacts",
        action="store_true",
        help="log the entire output dir (including model weights) to MLflow",
    )
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_dev_frame(frame: pd.DataFrame) -> None:
    required = ["context", "first letter", "answer"]
    if list(frame.columns) != required:
        raise ValueError(f"Expected dev columns {required}, got {list(frame.columns)}")
    if frame[required].eq("").to_numpy().any():
        raise ValueError("Validation data contains empty fields")
    if not frame["first letter"].str.len().eq(1).all():
        raise ValueError("Every first-character hint must contain exactly one character")
    answer_matches_hint = frame.apply(
        lambda row: row["answer"].startswith(row["first letter"]), axis=1
    )
    if not bool(answer_matches_hint.all()):
        raise ValueError("At least one validation answer does not match its hint")


def atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def atomic_text_write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class _StateHolder:
    trainer: "Trainer | None" = None


def _request_stop(signum, frame) -> None:  # noqa: ARG001
    trainer = _StateHolder.trainer
    if trainer is not None:
        trainer.stop_requested = True
        print("\nStop requested; finishing the current update and checkpointing...", flush=True)


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop_requested = False
        reject_forbidden_evaluation_path(args.dev_path)
        if args.rerank_words_per_root < 1 or args.rerank_batch_size < 1:
            raise ValueError("reranking sizes must be positive")
        if args.rerank_root_beam is not None and args.rerank_root_beam < 1:
            raise ValueError("--rerank-root-beam must be positive")
        for path in (args.train_path, args.dev_path):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        if args.num_workers:
            raise ValueError("--num-workers must be 0 (deterministic single-process stream)")
        args.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = bool(self.device.type == "cuda") and not args.no_fp16
        self.model_dtype = torch.float32

        print(f"Loading tokenizer/model {args.model_name} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=self.model_dtype
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.model.to(self.device)
        print(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M",
            flush=True,
        )

        print(f"Building candidate lexicon from {args.train_path} ...", flush=True)
        self.lexicon = hml.build_training_lexicon(
            self.tokenizer, args.train_path, max_lines=args.max_train_lines
        )
        print(
            f"Corpus: {self.lexicon.training_lines:,} lines, "
            f"{self.lexicon.training_tokens:,} words, {len(self.lexicon.word_counts):,} types",
            flush=True,
        )

        self.dev_df = pd.read_csv(args.dev_path, keep_default_na=False)
        validate_dev_frame(self.dev_df)
        if args.rerank_words_per_root > 1:
            self.decoder_predictable_words = {
                word
                for mapping in self.lexicon.hint_root_words.values()
                for words in mapping.values()
                for word in words[: args.rerank_words_per_root]
            }
        else:
            self.decoder_predictable_words = self.lexicon.predictable_words
        self.dev_coverage = hml.dev_answer_coverage(
            self.dev_df["answer"].tolist(),
            self.dev_df["first letter"].tolist(),
            self.decoder_predictable_words,
            self.lexicon.fallback_by_hint,
        )
        print(f"Decoder answer coverage on dev: {self.dev_coverage:.2%}", flush=True)

        self.used_bpe, self.used_lines = hml.count_pack_tokens(
            args.train_path, self.lexicon.word_bpe, args.max_train_lines
        )
        packed_blocks = self.used_bpe // args.block_size
        self.micro_batches_per_epoch = packed_blocks // args.train_batch_size
        self.steps_per_epoch = (
            self.micro_batches_per_epoch // args.gradient_accumulation_steps
        )
        if self.steps_per_epoch < 1:
            raise ValueError(
                "Training stream is too small to produce one optimizer update; "
                "raise --max-train-lines or lower block/batch sizes"
            )
        run_epochs = args.total_epochs if args.double_descent else args.epochs
        full_steps = self.steps_per_epoch * run_epochs
        if args.max_train_steps is None or args.max_train_steps < 0:
            self.planned_steps = full_steps
        else:
            self.planned_steps = args.max_train_steps
        if self.planned_steps < 1:
            raise ValueError("--max-train-steps must be positive or -1")

        self.cand_tensors, self.local_lookup = hml.build_hint_candidate_tables(
            self.lexicon.hint_chars,
            self.lexicon.candidate_ids_by_hint,
            self.tokenizer.vocab_size,
            self.device,
        )

    # ------------------------------------------------------------------ setup
    def _optimizer_and_schedule(self) -> None:
        args = self.args
        decay = [
            p for p in self.model.parameters() if p.requires_grad and p.ndim >= 2
        ]
        no_decay = [
            p for p in self.model.parameters() if p.requires_grad and p.ndim < 2
        ]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": args.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=args.learning_rate,
        )
        warmup = min(args.warmup_steps, max(1, self.planned_steps // 10))
        if args.double_descent:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup,
                num_training_steps=self.planned_steps,
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup,
                num_training_steps=self.planned_steps,
            )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_fp16)

    def _dataset(self) -> None:
        args = self.args
        dataset = hml.HintMaskedPackedDataset(
            args.train_path,
            self.lexicon.word_bpe,
            self.lexicon.hint_to_idx,
            self.tokenizer.eos_token_id,
            args.block_size,
            args.max_train_lines,
        )
        self.train_loader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            num_workers=0,
            pin_memory=self.use_fp16,
            drop_last=True,
            generator=torch.Generator().manual_seed(args.seed),
            collate_fn=hml.collate_hint_blocks,
        )

    # ------------------------------------------------------------ checkpoint
    def _data_fingerprint(self) -> dict:
        args = self.args
        return {
            "train_path": str(Path(args.train_path).resolve()),
            "model_name": args.model_name,
            "block_size": args.block_size,
            "train_batch_size": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_train_lines": args.max_train_lines,
        }

    def _resume_fingerprint(self) -> dict:
        args = self.args
        train_stat = Path(args.train_path).stat()
        return {
            **self._data_fingerprint(),
            "train_size": train_stat.st_size,
            "train_mtime_ns": train_stat.st_mtime_ns,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "max_grad_norm": args.max_grad_norm,
            "fp16": self.use_fp16,
            "double_descent": args.double_descent,
            "effective_epochs": (
                args.total_epochs if args.double_descent else args.epochs
            ),
            "planned_steps": self.planned_steps,
        }

    def _resume_state(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "overflow_steps": self.overflow_steps,
            "epoch_index": self.current_epoch,
            "micro_batch_index": self.micro_batch_index,
            "training_history": self.training_history,
            "dev_curve": self.dev_curve,
            "best_dev_accuracy": self.best_dev_accuracy,
            "fingerprint": self._data_fingerprint(),
            "resume_fingerprint": self._resume_fingerprint(),
            "rng_state": capture_rng_state(),
            "evaluation_version": EVALUATION_VERSION,
            "checkpoint_version": CHECKPOINT_VERSION,
            "mlflow_run_id": self.mlflow_run_id,
        }

    def save_resume(self) -> None:
        args = self.args
        path = args.output_dir / "resume_state.pt"
        atomic_torch_save(self._resume_state(), path)
        self._write_mlflow_resume_metadata()
        if args.keep_last_n > 1:
            numbered = args.output_dir / f"checkpoint_step_{self.global_step:08d}.pt"
            atomic_torch_save(self._resume_state(), numbered)
            checkpoints = sorted(
                args.output_dir.glob("checkpoint_step_*.pt"),
                key=lambda p: int(p.stem.rsplit("_", 1)[1]),
            )
            for stale in checkpoints[:-args.keep_last_n]:
                stale.unlink(missing_ok=True)

    def load_resume(self) -> bool:
        args = self.args
        path = args.output_dir / "resume_state.pt"
        if not args.resume or not path.is_file():
            if args.resume:
                print("No resume checkpoint found; starting fresh.", flush=True)
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["fingerprint"] != self._data_fingerprint():
            if args.force_fresh:
                print("Ignoring incompatible resume checkpoint (--force-fresh).", flush=True)
                return False
            raise ValueError(
                "Resume checkpoint was trained with a different data configuration "
                f"({state['fingerprint']}). Pass --force-fresh to discard it."
            )
        saved_resume_fingerprint = state.get("resume_fingerprint")
        current_resume_fingerprint = self._resume_fingerprint()
        if saved_resume_fingerprint is None:
            if args.force_fresh:
                print("Ignoring legacy resume checkpoint (--force-fresh).", flush=True)
                return False
            if not args.allow_legacy_resume:
                raise ValueError(
                    "Legacy checkpoint lacks exact RNG/config metadata. Pass "
                    "--allow-legacy-resume to continue it non-deterministically, "
                    "or --force-fresh to start over."
                )
        elif saved_resume_fingerprint != current_resume_fingerprint:
            differing = {
                key
                for key in saved_resume_fingerprint.keys() | current_resume_fingerprint.keys()
                if saved_resume_fingerprint.get(key) != current_resume_fingerprint.get(key)
            }
            schedule_only = differing <= {"effective_epochs", "planned_steps"}
            if args.allow_schedule_change and schedule_only:
                print("Resuming with an explicitly changed schedule horizon.", flush=True)
            elif args.force_fresh:
                print("Ignoring incompatible resume checkpoint (--force-fresh).", flush=True)
                return False
            else:
                raise ValueError(
                    "Resume checkpoint has incompatible settings "
                    f"({sorted(differing)}). Pass --force-fresh to start over."
                )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step = int(state["global_step"])
        self.overflow_steps = int(state["overflow_steps"])
        legacy_completed_groups = self.global_step + self.overflow_steps
        self.current_epoch = int(
            state.get(
                "epoch_index", legacy_completed_groups // self.steps_per_epoch
            )
        )
        self.micro_batch_index = int(
            state.get(
                "micro_batch_index",
                (legacy_completed_groups % self.steps_per_epoch)
                * args.gradient_accumulation_steps,
            )
        )
        self.training_history = list(state["training_history"])
        if state.get("evaluation_version") == EVALUATION_VERSION:
            self.dev_curve = list(state.get("dev_curve", []))
            self.best_dev_accuracy = state.get("best_dev_accuracy")
        else:
            self.dev_curve = []
            self.best_dev_accuracy = None
            print(
                "Resetting legacy dev metrics because corrected left-padding "
                "position IDs change evaluation accuracy.",
                flush=True,
            )
        best_path = args.output_dir / "best_dev.pt"
        if self.best_dev_accuracy is not None and best_path.is_file():
            best_checkpoint = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            best_file_accuracy = float(
                best_checkpoint["metrics"]["top_1_accuracy"]
            )
            self.best_dev_accuracy = max(
                float(self.best_dev_accuracy), best_file_accuracy
            )
        if state.get("rng_state") is None:
            print(
                "WARNING: legacy checkpoint has no RNG state; its first resumed "
                "segment cannot exactly match uninterrupted training.",
                flush=True,
            )
        restore_rng_state(state.get("rng_state"))
        if self.mlflow_run_id is None:
            self.mlflow_run_id = state.get("mlflow_run_id")
        print(f"Resumed from global step {self.global_step}", flush=True)
        return True

    # --------------------------------------------------------------- logging
    def _write_local_logs(self) -> None:
        args = self.args
        atomic_text_write(
            json.dumps(self.training_history, indent=2),
            args.output_dir / "training_history.json",
        )
        atomic_text_write(
            json.dumps(self.dev_curve, indent=2),
            args.output_dir / "dev_curve.json",
        )
        atomic_text_write(
            json.dumps(self.config_snapshot(), indent=2),
            args.output_dir / "run_config.json",
        )
        dev_frame = pd.DataFrame(self.dev_curve)
        if len(dev_frame):
            dev_frame.to_csv(args.output_dir / "dev_curve.csv", index=False)

    def config_snapshot(self) -> dict:
        args = self.args
        snapshot = vars(args).copy()
        for key in ("train_path", "dev_path", "test_path", "output_dir", "tracking_uri"):
            value = snapshot.get(key)
            if isinstance(value, Path):
                snapshot[key] = str(value)
            elif value is None:
                snapshot[key] = None
        snapshot.update(
            {
                "device": str(self.device),
                "fp16": self.use_fp16,
                "used_lines": self.used_lines,
                "used_bpe_tokens": self.used_bpe,
                "steps_per_epoch": self.steps_per_epoch,
                "micro_batches_per_epoch": self.micro_batches_per_epoch,
                "planned_steps": self.planned_steps,
                "dev_candidate_coverage": self.dev_coverage,
            }
        )
        return snapshot

    # ------------------------------------------------------------------- run
    def run(self) -> None:
        args = self.args
        start_time = time.time()
        self._optimizer_and_schedule()
        self._dataset()

        self.global_step = 0
        self.overflow_steps = 0
        self.training_history = []
        self.dev_curve = []
        self.best_dev_accuracy: float | None = None
        self.current_epoch = 0
        self.micro_batch_index = 0
        self.mlflow_failed = False
        self.mlflow_run_id = args.mlflow_run_id
        if self.mlflow_run_id is not None and not args.resume:
            raise ValueError("--mlflow-run-id requires --resume")
        mlflow_resume_path = args.output_dir / "mlflow_resume.json"
        mlflow_resume = None
        if args.resume and mlflow_resume_path.is_file():
            mlflow_resume = json.loads(
                mlflow_resume_path.read_text(encoding="utf-8")
            )
        self.load_resume()
        if mlflow_resume is not None:
            sidecar_run_id = mlflow_resume["run_id"]
            if self.mlflow_run_id is not None and self.mlflow_run_id != sidecar_run_id:
                raise ValueError(
                    "MLflow run ID conflict between checkpoint/CLI and mlflow_resume.json"
                )
            if mlflow_resume.get("experiment_name") != args.experiment_name:
                raise ValueError("MLflow experiment does not match mlflow_resume.json")
            if mlflow_resume.get("tracking_uri") != self._tracking_uri():
                raise ValueError("MLflow tracking URI does not match mlflow_resume.json")
            if int(mlflow_resume.get("checkpoint_step", 0)) > self.global_step:
                raise ValueError("MLflow sidecar is ahead of the model checkpoint")
            self.mlflow_run_id = sidecar_run_id
        args.mlflow_run_id = self.mlflow_run_id

        if args.eval_subset_size and args.eval_subset_size < len(self.dev_df):
            self.dev_subset = self.dev_df.sample(
                n=args.eval_subset_size, random_state=args.seed
            )
        else:
            self.dev_subset = self.dev_df

        _StateHolder.trainer = self
        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)

        self._start_mlflow()

        if self.global_step >= self.planned_steps:
            print("Checkpoint already at the requested number of steps.", flush=True)
        else:
            self._train_loop()

        if not self.stop_requested:
            self.save_resume()
        self._write_local_logs()
        self._finalize()
        elapsed = time.time() - start_time
        print(f"\nTotal wall time: {elapsed / 60:.1f} minutes", flush=True)

    def _train_loop(self) -> None:
        args = self.args
        self.model.train()
        self.model.config.use_cache = False
        self.optimizer.zero_grad(set_to_none=True)
        accumulated_batches = 0
        accumulated_loss_sum = 0.0
        accumulated_correct = 0
        accumulated_active = 0
        progress = tqdm(
            total=self.planned_steps,
            initial=self.global_step,
            desc="Hint-masked fine-tuning",
            unit=" update",
            dynamic_ncols=True,
        )

        while self.global_step < self.planned_steps and not self.stop_requested:
            completed_data_epoch = False
            resume_batch_index = self.micro_batch_index
            for batch_index, (input_ids, label_root_ids, label_hint_ids) in enumerate(
                self.train_loader
            ):
                if batch_index < resume_batch_index:
                    continue
                self.micro_batch_index = batch_index + 1
                input_ids = input_ids.to(self.device, non_blocking=True)
                label_root_ids = label_root_ids.to(self.device, non_blocking=True)
                label_hint_ids = label_hint_ids.to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_fp16,
                ):
                    hidden_states = self.model.base_model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        use_cache=False,
                    ).last_hidden_state
                    output_embeddings = self.model.get_output_embeddings()
                    if output_embeddings is None:
                        raise RuntimeError("language model has no output embedding layer")
                    loss, batch_correct, batch_active = hml.hint_masked_hidden_loss(
                        hidden_states,
                        label_root_ids,
                        label_hint_ids,
                        self.cand_tensors,
                        self.local_lookup,
                        output_embeddings.weight,
                        getattr(output_embeddings, "bias", None),
                    )
                    scaled_loss = loss / args.gradient_accumulation_steps
                self.scaler.scale(scaled_loss).backward()
                accumulated_batches += 1
                accumulated_loss_sum += float(loss.detach()) * batch_active
                accumulated_correct += batch_correct
                accumulated_active += batch_active
                if accumulated_batches < args.gradient_accumulation_steps:
                    continue

                self.scaler.unscale_(self.optimizer)
                grad_norm = float(
                    clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                )
                overflow = not math.isfinite(grad_norm)
                if not overflow:
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                update_loss = accumulated_loss_sum / accumulated_active
                update_accuracy = accumulated_correct / accumulated_active
                update_active = accumulated_active
                accumulated_loss_sum = 0.0
                accumulated_correct = 0
                accumulated_active = 0
                completed_data_epoch = (
                    self.micro_batch_index // args.gradient_accumulation_steps
                    >= self.steps_per_epoch
                )
                if overflow:
                    self.overflow_steps += 1
                    if self.stop_requested or completed_data_epoch:
                        break
                    continue

                self.scheduler.step()
                self.global_step += 1
                record = {
                    "step": self.global_step,
                    "loss": update_loss,
                    "root_accuracy": update_accuracy,
                    "active_targets": update_active,
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "gradient_norm": grad_norm,
                }
                self.training_history.append(record)
                progress.update(1)
                progress.set_postfix(
                    loss=f"{record['loss']:.3f}",
                    root_acc=f"{record['root_accuracy']:.3f}",
                    best_dev=f"{self.best_dev_accuracy:.4f}"
                    if self.best_dev_accuracy is not None
                    else "-",
                )

                if args.log_every_steps and self.global_step % args.log_every_steps == 0:
                    self._log_step_metrics(record)
                if (
                    args.eval_every_steps
                    and args.eval_subset_size
                    and self.global_step % args.eval_every_steps == 0
                ):
                    self._run_dev_eval()
                if (
                    args.resume_every_steps
                    and self.global_step % args.resume_every_steps == 0
                    and not completed_data_epoch
                ):
                    self.save_resume()
                    self._write_local_logs()

                if (
                    self.global_step >= self.planned_steps
                    or self.stop_requested
                    or completed_data_epoch
                ):
                    break
            else:
                completed_data_epoch = True

            if completed_data_epoch:
                self.optimizer.zero_grad(set_to_none=True)
                accumulated_batches = 0
                accumulated_loss_sum = 0.0
                accumulated_correct = 0
                accumulated_active = 0
                self.current_epoch += 1
                self.micro_batch_index = 0
                if (
                    args.eval_at_epoch_end
                    and args.eval_subset_size
                    and (
                        not self.dev_curve
                        or self.dev_curve[-1].get("step") != self.global_step
                    )
                ):
                    self._run_dev_eval()
                self.save_resume()
                self._write_local_logs()

        progress.close()
        if self.stop_requested:
            print("Stop requested: checkpointing at the current step.", flush=True)
            self.save_resume()
        print(
            f"Finished {self.global_step:,} optimizer updates "
            f"({self.overflow_steps:,} overflow-skipped)",
            flush=True,
        )

    # ------------------------------------------------------------ MLflow etc.
    def _tracking_uri(self) -> str:
        return self.args.tracking_uri or (
            "sqlite:///" + str((self.args.output_dir / "mlflow.db").resolve())
        )

    def _write_mlflow_resume_metadata(self) -> None:
        if not self.mlflow_run_id:
            return
        metadata = {
            "run_id": self.mlflow_run_id,
            "tracking_uri": self._tracking_uri(),
            "experiment_name": self.args.experiment_name,
            "checkpoint_step": self.global_step,
        }
        atomic_text_write(
            json.dumps(metadata, indent=2) + "\n",
            self.args.output_dir / "mlflow_resume.json",
        )

    def _mlflow_available(self) -> bool:
        return bool(self.args.mlflow and mlflow is not None and not self.mlflow_failed)

    def _mlflow_try(self, action: str, callback):
        if not self._mlflow_available():
            return None
        error = None
        for attempt in range(3):
            try:
                return callback()
            except Exception as caught:
                error = caught
                if attempt < 2:
                    time.sleep(0.5 * (2**attempt))
        self.mlflow_failed = True
        print(
            f"WARNING: MLflow {action} failed after retries ({error!r}); "
            "continuing with local JSON/CSV logging.",
            flush=True,
        )
        return None

    def _end_mlflow(self, status: str) -> None:
        if mlflow is None or not self.mlflow_run_id:
            return
        try:
            mlflow.end_run(status=status)
        except Exception as error:
            print(f"WARNING: MLflow run finalization failed ({error!r}).", flush=True)

    def _start_mlflow(self) -> None:
        args = self.args
        if not self._mlflow_available():
            if args.mlflow and mlflow is None:
                print(
                    "WARNING: mlflow is not installed; continuing with local logs only.",
                    flush=True,
                )
            return
        tracking_uri = self._tracking_uri()
        try:
            mlflow.set_tracking_uri(tracking_uri)
            if self.mlflow_run_id:
                client = MlflowClient(tracking_uri=tracking_uri)
                existing = client.get_run(self.mlflow_run_id)
                experiment = client.get_experiment(existing.info.experiment_id)
                if experiment.name != args.experiment_name:
                    raise ValueError("existing MLflow run belongs to another experiment")
                metric_history = client.get_metric_history(
                    self.mlflow_run_id, "train_loss"
                )
                if metric_history and max(metric.step for metric in metric_history) > self.global_step:
                    raise ValueError("MLflow run is ahead of the model checkpoint")
                self.mlflow_run = mlflow.start_run(run_id=self.mlflow_run_id)
                experiment_name = args.experiment_name
            else:
                experiment = mlflow.set_experiment(args.experiment_name)
                self.mlflow_run = mlflow.start_run(
                    run_name=args.run_name or Path(args.output_dir).name
                )
                self.mlflow_run_id = self.mlflow_run.info.run_id
                experiment_name = experiment.name
                mlflow.log_params(
                    {
                        key: (
                            str(value)
                            if isinstance(value, (Path, list, dict))
                            else value
                        )
                        for key, value in self.config_snapshot().items()
                    }
                )
            self._write_mlflow_resume_metadata()
        except Exception as error:
            self.mlflow_failed = True
            print(
                f"WARNING: MLflow startup failed ({error!r}); continuing with "
                "local JSON/CSV logging.",
                flush=True,
            )
            return
        print(
            f"MLflow run {experiment_name}: {self.mlflow_run.info.run_id} "
            f"(tracking: {tracking_uri})",
            flush=True,
        )

    def _log_step_metrics(self, record: dict) -> None:
        if self._mlflow_available():
            self._mlflow_try(
                "training metric logging",
                lambda: mlflow.log_metrics(
                    {
                        "train_loss": record["loss"],
                        "train_root_accuracy": record["root_accuracy"],
                        "learning_rate": record["learning_rate"],
                        "gradient_norm": record["gradient_norm"],
                    },
                    step=record["step"],
                ),
            )

    def _run_dev_eval(self) -> dict:
        args = self.args
        self.model.eval()
        try:
            metrics, _, _ = hml.evaluate_model(
                self.model,
                self.tokenizer,
                self.dev_subset,
                candidate_ids_by_hint=self.lexicon.candidate_ids_by_hint,
                hint_token_word=self.lexicon.hint_token_word,
                fallback_by_hint=self.lexicon.fallback_by_hint,
                predictable_words=self.decoder_predictable_words,
                hint_root_words=self.lexicon.hint_root_words,
                word_bpe=self.lexicon.word_bpe,
                rerank_words_per_root=args.rerank_words_per_root,
                rerank_root_beam=args.rerank_root_beam,
                rerank_batch_size=args.rerank_batch_size,
                label="dev",
                top_k=args.top_k,
                eval_batch_size=64,
                max_context_tokens=args.max_context_tokens,
            )
        finally:
            self.model.train()
        metrics["step"] = self.global_step
        metrics["epoch"] = self.global_step / self.steps_per_epoch
        self.dev_curve.append(metrics)
        dev_accuracy = float(metrics["top_1_accuracy"])
        if self.best_dev_accuracy is None or dev_accuracy > self.best_dev_accuracy:
            self.best_dev_accuracy = dev_accuracy
            atomic_torch_save(
                {
                    "model": self.model.state_dict(),
                    "metrics": metrics,
                    "step": self.global_step,
                },
                args.output_dir / "best_dev.pt",
            )
        if self._mlflow_available():
            mlflow_metrics = {
                "dev_top_1_accuracy": metrics["top_1_accuracy"],
                f"dev_top_{args.top_k}_accuracy": metrics[
                    f"top_{args.top_k}_accuracy"
                ],
                "dev_alphanumeric_top_1": metrics[
                    "alphanumeric_top_1_accuracy"
                ],
            }
            self._mlflow_try(
                "validation metric logging",
                lambda: mlflow.log_metrics(
                    {
                        key: value
                        for key, value in mlflow_metrics.items()
                        if value is not None
                    },
                    step=self.global_step,
                ),
            )
        print(
            f"  step {self.global_step}: dev top-1={dev_accuracy:.4f} "
            f"(best={self.best_dev_accuracy:.4f})",
            flush=True,
        )
        return metrics

    # -------------------------------------------------------------- finalize
    def _save_model_and_resources(self) -> None:
        args = self.args
        self.model.save_pretrained(args.output_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(args.output_dir)
        resources = {
            "candidate_ids_by_hint": self.lexicon.candidate_ids_by_hint,
            "hint_token_word": {
                hint: {str(root): word for root, word in mapping.items()}
                for hint, mapping in self.lexicon.hint_token_word.items()
            },
            "fallback_by_hint": self.lexicon.fallback_by_hint,
            "hint_chars": self.lexicon.hint_chars,
        }
        if args.rerank_words_per_root > 1:
            rerank_words = {
                hint: {
                    str(root): words[: args.rerank_words_per_root]
                    for root, words in mapping.items()
                }
                for hint, mapping in self.lexicon.hint_root_words.items()
            }
            retained_words = {
                word
                for mapping in rerank_words.values()
                for words in mapping.values()
                for word in words
            }
            resources["hint_root_words"] = rerank_words
            resources["word_bpe"] = {
                word: self.lexicon.word_bpe[word] for word in retained_words
            }
        atomic_text_write(
            json.dumps(resources, ensure_ascii=False),
            args.output_dir / "prediction_resources.json",
        )

    def _load_best_model(self) -> None:
        if self.best_dev_accuracy is None:
            return
        path = self.args.output_dir / "best_dev.pt"
        if not path.is_file():
            raise FileNotFoundError(f"best checkpoint is missing: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        print(
            f"Selected best checkpoint from step {checkpoint['step']} "
            f"(dev top-1={checkpoint['metrics']['top_1_accuracy']:.4f}).",
            flush=True,
        )

    def _finalize(self) -> None:
        args = self.args
        self._write_local_logs()

        if self.stop_requested:
            print(
                "Interrupt received: saving model/resources and exiting without "
                "full dev evaluation or test predictions.",
                flush=True,
            )
            self._save_model_and_resources()
            self._end_mlflow("KILLED")
            print(f"Artifacts saved to {args.output_dir}", flush=True)
            return

        self._load_best_model()

        if args.final_dev_eval:
            print("Evaluating final model on the full dev set...", flush=True)
            final_metrics, final_details, final_per_hint = hml.evaluate_model(
                self.model,
                self.tokenizer,
                self.dev_df,
                candidate_ids_by_hint=self.lexicon.candidate_ids_by_hint,
                hint_token_word=self.lexicon.hint_token_word,
                fallback_by_hint=self.lexicon.fallback_by_hint,
                predictable_words=self.decoder_predictable_words,
                hint_root_words=self.lexicon.hint_root_words,
                word_bpe=self.lexicon.word_bpe,
                rerank_words_per_root=args.rerank_words_per_root,
                rerank_root_beam=args.rerank_root_beam,
                rerank_batch_size=args.rerank_batch_size,
                label="Hint-masked GPT-2 (final)",
                top_k=args.top_k,
                eval_batch_size=64,
                max_context_tokens=args.max_context_tokens,
            )
            print(json.dumps(final_metrics, indent=2), flush=True)
            pd.DataFrame([final_metrics]).to_csv(
                args.output_dir / "validation_summary.csv", index=False
            )
            final_details.to_csv(
                args.output_dir / "validation_predictions.csv", index=False
            )
            final_per_hint.to_csv(
                args.output_dir / "validation_per_hint.csv", index=False
            )
        else:
            print("Skipping full dev evaluation (--no-final-dev-eval).", flush=True)

        self._save_model_and_resources()

        if Path(args.test_path).is_file():
            print(f"Writing predictions for {args.test_path} ...", flush=True)
            rows = hml.write_test_predictions(
                self.model,
                self.tokenizer,
                args.test_path,
                args.output_dir / "test_set_pred.txt",
                candidate_ids_by_hint=self.lexicon.candidate_ids_by_hint,
                hint_token_word=self.lexicon.hint_token_word,
                fallback_by_hint=self.lexicon.fallback_by_hint,
                hint_root_words=self.lexicon.hint_root_words,
                word_bpe=self.lexicon.word_bpe,
                rerank_words_per_root=args.rerank_words_per_root,
                rerank_root_beam=args.rerank_root_beam,
                rerank_batch_size=args.rerank_batch_size,
                max_context_tokens=args.max_context_tokens,
                batch_size=128,
            )
            print(f"Wrote {rows} predictions to {args.output_dir / 'test_set_pred.txt'}", flush=True)
        else:
            print(
                f"Test file {args.test_path} not found; skipping test predictions.",
                flush=True,
            )

        if self._mlflow_available():
            for name in (
                "run_config.json",
                "training_history.json",
                "dev_curve.csv",
                "validation_summary.csv",
                "validation_per_hint.csv",
            ):
                path = args.output_dir / name
                if path.is_file():
                    self._mlflow_try(
                        f"artifact logging ({name})",
                        lambda path=path: mlflow.log_artifact(str(path)),
                    )
            if args.mlflow_log_artifacts:
                self._mlflow_try(
                    "artifact directory logging",
                    lambda: mlflow.log_artifacts(str(args.output_dir)),
                )
            self._end_mlflow("FINISHED")

        print(f"Artifacts saved to {args.output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    print(
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )
    try:
        trainer = Trainer(args)
    except KeyboardInterrupt:
        print("Interrupted before training started.", flush=True)
        sys.exit(130)
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_resume()
        trainer._end_mlflow("KILLED")
        print("Interrupted; checkpoint saved.", flush=True)
        sys.exit(130)
    except Exception:
        trainer._end_mlflow("FAILED")
        raise
    finally:
        _StateHolder.trainer = None
    if trainer.stop_requested:
        print("Stopped after interrupt; checkpoint saved.", flush=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
