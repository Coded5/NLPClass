# ABSA Model Development Report

**Status date:** 9 September 2026  
**Status:** Experiment complete. Validation selection, locked test evaluation,
three-seed ensembling, and bootstrap comparison have finished.

## Objective

The original RoBERTa implementation treated each annotation row as an independent
single-label example. A review containing multiple aspect rows therefore received
the same single aspect prediction for every copy of its text. This prevented the
system from recovering the complete set of aspect-polarity pairs.

The work so far has replaced that assumption, analyzed the remaining errors, and
started controlled experiments on class imbalance and a joint ABSA architecture.
The primary selection metric is micro-F1 over complete
`(id, aspectCategory, polarity)` tuples.

## Dataset and Evaluation Integrity

The labeled dataset contains 3,156 rows and 2,584 review IDs. Analysis of repeated
text found:

| Property | Count |
|---|---:|
| Repeated-text groups | 494 |
| Valid multi-aspect repeated-text groups | 490 |
| Exact duplicate label triplets | 4 |
| Conflicting polarities for the same text and aspect | 0 |
| Repeated-text groups spanning multiple IDs | 2 |

Most repeated text is expected ABSA structure: one review has one row for every
annotated aspect. These rows must not be deduplicated by text or ID during
evaluation. The evaluator instead uses sets of `(id, aspect, polarity)` tuples,
which preserves valid multi-aspect labels and naturally collapses exact duplicate
triplets.

Splits are grouped by review ID to keep all annotations for a review together. The
persisted experiment split is 80% training, 10% validation, and 10% locked test.
Thus, 20% is held out from training, but only the final 10% is the untouched test
partition. For stricter future experiments, grouping by normalized text would also
prevent the two cross-ID repeated texts from crossing split boundaries.

## Multilabel Conditioned Baseline

The first correction uses two RoBERTa models:

1. A multilabel aspect model predicts every aspect present in a review.
2. A polarity model receives the review and a candidate aspect, then predicts the
   polarity for that specific aspect.

Per-aspect validation thresholds are tuned without using test labels. This design
can emit multiple correctly paired labels for a single review.

On the locked 258-review test split, containing 316 gold pairs:

| System | Pair micro-F1 | Exact-set accuracy |
|---|---:|---:|
| Legacy single-label pipeline | 0.6725 | 0.5814 |
| Multilabel conditioned pipeline | **0.7223** | **0.6395** |

The corrected pipeline produced 225 true pairs, 82 false positives, and 91 false
negatives. Of the missed pairs, 45 came from missed aspects and 46 from incorrect
polarity after detecting the aspect. Multi-aspect reviews remained harder than
single-aspect reviews: pair F1 was 0.6974 versus 0.7336, while exact-set accuracy
was 0.3529 versus 0.7101.

Minority polarity classes were the weakest. Test pair F1 was 0.819 for positive,
0.657 for negative, 0.500 for neutral, and 0.452 for conflict. This motivated the
controlled loss experiment.

## Implemented Experiment Framework

The staged experiment is implemented in:

- `src/zeroshot_classifier/imbalance_losses.py`
- `src/zeroshot_classifier/joint_absa_experiment.py`
- `scripts/run_joint_absa_experiment.py`
- `scripts/train_absa_variant.py`

It provides:

- inverse-frequency weighted cross-entropy;
- weighted focal loss with configurable gamma;
- class-balanced focal loss using effective-number weights;
- a candidate-conditioned joint RoBERTa model;
- per-aspect threshold tuning;
- deterministic seeds and grouped splits;
- epoch checkpoints and interruption-safe resume;
- early stopping based on validation pair micro-F1;
- three-seed confirmation and logit ensembling;
- MLflow logging to `sqlite:///artifacts/mlflow.db`;
- terminal progress bars and structured result files;
- a locked-selection file before test labels are accessed;
- paired review-level bootstrap comparison for final systems.

The joint model uses one shared RoBERTa encoder. Each review is paired with all
five candidate aspects. An aspect-presence head predicts whether the candidate is
present, while a four-class polarity head is trained only for present candidates.
The joint loss is weighted binary cross-entropy for aspect presence plus the
selected masked polarity loss.

The configured maximum is 50 epochs, with validation every 5 epochs, early-stop
patience of 3 validation checks, and a minimum F1 improvement of 0.001. Checkpoint
selection retains the best epoch rather than the last epoch.

## Completed Validation Results

### Separate-model polarity loss screen, seed 42

The aspect checkpoint and thresholds were held fixed so this phase isolated the
effect of polarity loss.

| Polarity loss | Best epoch | Validation pair micro-F1 |
|---|---:|---:|
| Weighted cross-entropy | 30 | **0.7740** |
| Weighted focal, gamma 1 | 45 | 0.7616 |
| Weighted focal, gamma 2 | 40 | 0.7678 |
| Class-balanced focal, beta 0.999, gamma 2 | 30 | 0.7554 |

Weighted cross-entropy won the separate-model screen. Stronger focusing did not
improve overall pair F1, suggesting that emphasizing hard or noisy minority
examples reduced generalization on this validation split.

### Joint-model loss screen, seed 42

| Joint polarity loss | Best epoch | Validation pair micro-F1 |
|---|---:|---:|
| Weighted cross-entropy | 40 | 0.7757 |
| Weighted focal, gamma 2 | 35 | **0.7836** |

The joint model benefited from focal loss even though the separate model did not.
Joint focal gamma 2 currently has the strongest single validation result, exceeding
separate weighted cross-entropy by 0.0096 absolute F1 on seed 42.

### Separate-model confirmation

| Seed | Best epoch | Validation pair micro-F1 |
|---:|---:|---:|
| 17 | 25 | 0.7523 |
| 42 | 30 | 0.7740 |
| 73 | 30 | 0.7570 |

All three separate weighted-cross-entropy seeds are complete. The final comparison
uses their ensemble rather than selecting the strongest individual seed.

### Joint-model confirmation

| Seed | Best epoch | Validation pair micro-F1 |
|---:|---:|---:|
| 17 | 45 | 0.7664 |
| 42 | 35 | 0.7836 |
| 73 | 40 | 0.7607 |

The mean joint validation F1 was 0.7702 with a standard deviation of 0.0097.
The separate-model mean was 0.7611 with a standard deviation of 0.0093.

### Ensemble validation and locked selection

| Architecture | Selected loss | Ensemble validation F1 |
|---|---|---:|
| Separate | Weighted cross-entropy | 0.7554 |
| Joint | Weighted focal, gamma 2 | **0.7850** |

The joint ensemble and its thresholds were locked before test inference. Its tuned
aspect thresholds were `[0.50, 0.75, 0.50, 0.30, 0.25]`. The separate thresholds
were `[0.65, 0.85, 0.90, 0.90, 0.50]`.

## Final Untouched Test Results

| System | Pair micro-F1 | Exact-set accuracy | Single-aspect exact | Multi-aspect exact |
|---|---:|---:|---:|---:|
| Separate three-seed ensemble | **0.7448** | **0.6550** | **0.7150** | **0.4118** |
| Locked joint three-seed ensemble | 0.7244 | 0.6279 | 0.7005 | 0.3333 |
| Existing conditioned pipeline | 0.7223 | 0.6395 | 0.7101 | 0.3529 |
| Legacy single-label baseline | 0.6725 | 0.5814 | 0.7246 | 0.0000 |

The validation-selected joint ensemble improved only 0.0021 absolute test F1 over
the existing conditioned pipeline. The separate ensemble, although not selected by
validation, produced the highest observed test F1 and improved 0.0225 over the
existing pipeline. It also gave the best multi-aspect exact-set accuracy.

This test observation must not be used to retroactively change the locked winner or
tune thresholds. Doing so would turn the test partition into validation data. It is
instead evidence for a future experiment with a larger or repeated validation
protocol.

### Pair-level polarity results

| System | Positive F1 | Negative F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|
| Separate ensemble | **0.8285** | **0.6939** | 0.5278 | 0.4000 |
| Joint ensemble | 0.8257 | 0.6203 | **0.5333** | **0.4828** |
| Existing pipeline | 0.8194 | 0.6569 | 0.5000 | 0.4516 |
| Legacy baseline | 0.7733 | 0.5827 | 0.4935 | 0.3077 |

Joint focal loss improved neutral and conflict F1 relative to the separate ensemble,
but it substantially reduced negative F1. The separate ensemble's stronger negative
classification and aspect-set recovery outweighed the joint model's minority-class
gains in overall pair F1.

### Bootstrap interpretation

The paired review-level bootstrap estimated joint minus separate F1 as -0.0203,
with a 95% interval from -0.0513 to +0.0099. The locked joint winner minus the
existing pipeline was +0.0023, with a 95% interval from -0.0315 to +0.0372.

Both intervals include zero. Consequently, this split does not provide strong
statistical evidence that the joint ensemble differs from the separate ensemble or
that it improves upon the existing conditioned pipeline.

## Training and Evaluation Runtime Changes

Training runs one model at a time. Initially, scheduled validation constructed a
fresh CPU model, copied the current training weights into it, ran inference, and
deleted it. This kept the GPU training model resident but caused repeated
Hugging Face `Loading weights` messages and unnecessary CPU work.

The current implementation now evaluates the resident training model directly on
its existing device. When CUDA is available, scheduled validation, fixed-aspect
inference, and final ensemble inference run on GPU. Only output logits are moved to
CPU for metrics, and checkpoint state is copied to CPU for durable serialization.
Separate ensemble members are still loaded sequentially to control peak VRAM.

The completed run used detached tmux window `NLP:6`, named `absa-joint-gpu`.
Output was appended to
`artifacts/experiments/absa-imbalance-joint-v1/resume.log`. The final runner
invocation recorded 85.5 minutes; earlier completed screening stages were resumed
from checkpoints and are not included in that invocation timer.

## Verification Completed

- Full unit suite: 36 tests passed.
- Python compilation checks passed.
- `git diff --check` passed.
- A real RoBERTa forward/backward smoke test produced finite joint loss and the
  expected aspect and polarity output shapes.
- The official evaluator accepted generated gold and prediction files.
- Split validation confirmed no review-ID overlap.

## Conclusions and Recommended Next Work

The experiment answered two separate questions:

1. Loss reweighting is architecture-dependent. Weighted cross-entropy was best for
   the separate polarity model, while focal gamma 2 was best for the joint model.
2. The joint model won validation but did not generalize better on the locked test
   split. The separate ensemble produced the strongest observed test result, though
   the comparison was not statistically conclusive.

A follow-up should preserve the test partition and improve selection stability by
using grouped cross-validation or repeated validation splits on the current
training/validation portion. Candidate experiments should focus on joint-loss
balancing or negative-class confusion rather than increasing class weights further.
The final generated artifacts are:

- `artifacts/experiments/absa-imbalance-joint-v1/results.json`
- `artifacts/experiments/absa-imbalance-joint-v1/report.md`
- `artifacts/experiments/absa-imbalance-joint-v1/locked_selection.json`
