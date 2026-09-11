# Multi-label aspect experiment

## Validation selection

| Seed | Best epoch | Validation micro-F1 |
|---:|---:|---:|
| 42 | 10 | 0.888545 |
| 43 | 35 | 0.886536 |
| 44 | 10 | 0.878125 |

Three-seed mean: 0.884402.
Three-seed sample standard deviation: 0.005528.
Ensemble validation micro-F1: 0.894488.
Locked ensemble thresholds: `[0.55, 0.75, 0.75, 0.7, 0.75]`.

## Held-out test comparison

| System | Aspect micro-F1 | Aspect macro-F1 | Exact-set accuracy | Multi-aspect exact accuracy |
|---|---:|---:|---:|---:|
| Three-seed ensemble | 0.877419 | 0.870611 | 0.794574 | 0.549020 |
| Prior one-seed multi-label | 0.869984 | 0.864261 | 0.779070 | 0.549020 |
| Legacy single-label | 0.815331 | 0.782845 | 0.709302 | 0.000000 |

Ensemble minus legacy baseline paired-bootstrap delta: +0.062096
(95% CI +0.031789 to +0.093155;
10000 ID-grouped samples).

Checkpoint and threshold selection used validation data only. The held-out test split
has been evaluated by earlier repository experiments, so it is not described as globally untouched.

Runtime: 17.5 minutes.
