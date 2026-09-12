# Synthetic conflict augmentation pilot

## Frozen design

Three RoBERTa-base conditions across five grouped outer folds: original data,
matched repetition, and synthetic augmentation. Only the original training
partition is used. Reuse learning-curve-v1 fitting/selection/heldout review IDs.
Both source clauses must belong to the fitting IDs for the particular fold.

Each fold adds 40 conflict examples and 40 cross-aspect controls (20 positive,
20 negative). Repetition adds the same counts by aspect and polarity using
original fitting annotations. Each clause is used at most five times per fold;
source pairs and generated texts are unique within a fold. Clause order varies;
20% use separate sentences and the remainder use but/though/however/although.
All 400 examples were inspected by the assistant before training. Judgments
remain provisional. Exact source spans, IDs, labels and rationale are retained.
No anecdotes/miscellaneous examples are synthesized because its target is less
specific. Consequently, augmentation is not evenly distributed across aspects.

The source bank is curated from original training annotations without reference
to per-example model errors. Fold filtering takes place before pairing. No
source clause can enter training from that fold's selection or heldout reviews.
Original labels are unchanged. The separately authored 12-row diagnostic is
frozen in preparation.json and never used for checkpoint or condition selection.

## Training

Aspect-conditioned four-class polarity; weighted cross-entropy with weights
computed only from original fitting annotations. All conditions use learning
rate 2e-5, physical batch 8, accumulation 2, maximum length 256, weight decay
0.01, warmup 0.1, maximum 50 epochs, evaluation every five epochs, patience
three checks, and minimum improvement 0.001. Selection metric is natural
inner-selection polarity macro-F1. Seeds are 42 + fold number. Repetition and
augmentation have identical maximum update budgets, though early stopping may
yield different actual training lengths. Original-only reference has fewer rows.

A one-epoch smoke check for all three conditions in fold 1 precedes the full
15 fits. Smoke results are isolated and do not influence parameters or promotion.
GPU training and evaluation remain resident across epochs. Best model weights
and logits are retained; a completed fit's own optimizer snapshot is removed.
Unfinished fits remain resumable. Interrupted runs reuse the existing trainer's
recovery behavior; bitwise equivalence to uninterrupted training is not claimed.

## Primary decision

Primary contrast: synthetic minus matched repetition, pooled natural conflict F1.
Use 10,000 paired bootstrap resamples of review IDs, keeping annotations together.
Proceed to a separate ensemble-confirmation experiment only if all apply:

- Conflict F1 improves at least 0.03 over repetition and its 95% interval excludes zero positively.
- Conflict F1 improves in at least four of five folds and exceeds original reference pooled F1.
- Macro-F1 and neutral F1 each fall by no more than 0.01 versus either control.
- Conflict false-positive rate on non-conflict annotations rises by no more than 0.01 versus either control.

Full class precision/recall/F1/support, confusion matrices, fold mean/SD, timing,
peak allocated memory, selected epochs and diagnostic predictions are retained.
Natural conjunction-subset metrics are descriptive; conjunction presence does
not establish same-aspect conflict. Intervals condition on the fitted models
and exclude training-seed uncertainty. This pilot cannot establish a new
complete-system winner or an unbiased improvement on a fresh test set.

## Commands

```bash
.venv/bin/python scripts/run_conflict_augmentation.py prepare
.venv/bin/python scripts/run_conflict_augmentation.py run --smoke-only
.venv/bin/python scripts/run_conflict_augmentation.py run
.venv/bin/python scripts/run_conflict_augmentation.py report
```

Training requires a complete assistant review matched to the preparation hash.
Configuration, source hashes and reviews are frozen in the run manifest. The
tmux wrapper streams run.log, writes exit_code.txt, queues completion back to
the originating thread, and closes its window automatically.
