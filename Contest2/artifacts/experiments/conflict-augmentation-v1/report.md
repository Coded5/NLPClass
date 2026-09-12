# Synthetic conflict augmentation pilot

Five grouped folds, RoBERTa-base, natural gold-aspect polarity evaluation. All checkpoints selected using natural inner-selection macro-F1. Synthetic labels are provisional assistant judgments. No historical validation or test was evaluated.

| Condition | Accuracy | Macro-F1 | Positive F1 | Negative F1 | Neutral F1 | Conflict F1 | Conflict FP rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.8067 | 0.6663 | 0.9028 | 0.7579 | 0.6558 | 0.3487 | 0.0489 |
| repetition | 0.8004 | 0.6690 | 0.8942 | 0.7621 | 0.6311 | 0.3887 | 0.0514 |
| synthetic | 0.8143 | 0.6817 | 0.9026 | 0.7757 | 0.6525 | 0.3960 | 0.0439 |

Primary conflict-F1 delta: +0.0073; paired review-bootstrap 95% interval [-0.0618, +0.0776]. Improved folds: 2/5.

Advance to ensemble confirmation: False.

The intervals condition on fitted models and do not include training-seed uncertainty. Cross-aspect diagnostic and conjunction-subset results are descriptive only. The augmentation and repetition arms have matched epoch lengths and maximum update budgets; early stopping can produce different realized update counts. This pilot does not select a new complete-system winner.

Full class precision/recall/support, confusion matrices, per-fold variation, guardrail checks, epochs, timing and memory are in `summary.json`; logits and texts are in each `result.json`.
