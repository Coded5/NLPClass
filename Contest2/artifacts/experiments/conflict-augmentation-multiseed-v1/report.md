# Multi-seed synthetic-conflict isolation

RoBERTa polarity-only comparison on five grouped outer folds. Both arms add exactly 40 conflict rows per fold with identical aspect counts. Repetition duplicates frozen real conflict rows; synthetic uses reviewed same-aspect positive/negative clause combinations. Natural selection and heldout partitions are unchanged.

| Three-seed ensemble | Accuracy | Macro-F1 | Neutral F1 | Conflict F1 | Conflict FP rate |
|---|---:|---:|---:|---:|---:|
| repetition | 0.8147 | 0.6827 | 0.6489 | 0.4026 | 0.0485 |
| synthetic | 0.8202 | 0.6941 | 0.6482 | 0.4328 | 0.0439 |

Primary conflict-F1 delta: +0.0302; paired review-bootstrap 95% interval [-0.0325, +0.0931].
Per-seed pooled conflict-F1 deltas: -0.0055, -0.0241, +0.0377.
Synthetic wording supported: False.

Full per-seed and fold-seed results, class metrics, confusion matrices, timing, selected epochs and decision thresholds are in `summary.json`. Historical validation and test partitions were not used.
