# DeBERTa-v3-large LoRA polarity pilot

## Queue and scope

User authorized this experiment on 2026-09-13, following the three-seed
class-weight confirmation. The dependency is the specific attempt
`artifacts/experiments/f1-roadmap-v2/attempts/confirmation-ZKT9qXKc`.
The dependency finished successfully. The tested direct runner was launched in
tmux window `NLP:5`, pane `%53`, PID `3624266`. It runs the feasibility
benchmark and pilot itself, writes a durable exit status, and sends a completion
message to the originating Codex thread.

Train `microsoft/deberta-v3-large` with LoRA for aspect-conditioned four-class
polarity, starting with seed 42. Keep aspect models frozen. Use the established
training data, exclusions, and frozen grouped folds; select checkpoints only
on each fold's inner validation partition. Keep the historical test out of
selection. Report both matched seed-42 and existing three-seed comparisons.
Do not add base-LoRA training or additional seeds without a subsequent decision.

## Question

Can the larger pretrained backbone with limited trainable adaptation improve
polarity and complete-pair F1 over the current fully fine-tuned base model?
This changes both model size and adaptation method; a negative result does
not isolate either factor as the cause.

## Procedure

Inspect the completed weighting confirmation before fixing the pilot loss
recipe. Preserve the existing inner-selected fold recipes if using that
experiment's challengers; never select a recipe using outer-fold labels.
Record the exact recipe and baseline before training.

Implement explicit attention query/value LoRA adapters, initially rank 16,
alpha 32, dropout 0.05, with trainable classification/pooling heads. Verify
the actual module names and trainable parameter inventory. Freeze the backbone,
including embeddings. Start with mixed precision, gradient checkpointing,
microbatch 2 and gradient accumulation to preserve the baseline effective batch.
Treat adapter learning rate separately from the full-fine-tuning baseline;
predeclare a starting adapter rate of 1e-4, with no outer-fold tuning.

Run a ten-minute GPU feasibility benchmark before committing to all five
seed-42 folds. Probe microbatches `16, 8, 4, 2, 1` using a full-length
forward/backward step and select the largest configuration that fits while
holding the effective batch at 16 through gradient accumulation. Measure
steady-state throughput, peak allocated/reserved memory,
finite loss, adapter/head updates, and save/reload prediction equivalence.
Persist model revision, tokenizer fingerprint, splits and source hashes,
adapter configuration, timing, and a runtime projection. Preserve DeBERTa's
unmodified tokenizer behavior (`fix_mistral_regex=False`). If the benchmark
fits and updates correctly, continue the pilot; report a feasibility failure
otherwise. Do not claim this benchmark establishes model quality.

## Evaluation and interpretation

Report pooled OOF pair F1, aspect and official polarity F1, exact-set accuracy,
gold-aspect polarity accuracy/macro-F1/per-class metrics, fold deltas, and paired
normalized-text-group bootstrap intervals. Compare with the same folds and
seed before comparing against the three-seed incumbent. A pilot gain remains
unconfirmed until additional seeds reproduce it. Do not promote based solely
on a single validation score or add blending searches automatically.

## Completion notifications

Use a detached window in the current tmux session with visible tee output.
Record exit status before invoking `codex queue` to the originating thread.
Notify after the pilot completes or fails, including log and result paths;
the window must close automatically. Stop after this experiment.
