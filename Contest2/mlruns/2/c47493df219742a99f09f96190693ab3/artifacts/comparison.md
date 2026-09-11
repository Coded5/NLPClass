# Multilabel aspect and conditioned polarity experiment

## Selected checkpoints

| Stage | Epoch | Validation selection F1 |
|---|---:|---:|
| Aspect | 10 | 0.888545 |
| Polarity | 45 | 0.736755 |

Aspect thresholds: `[0.65, 0.85, 0.9, 0.9, 0.5]`

## Composed pipeline metrics

| Split | Aspect micro-F1 | Polarity micro-F1 | Overall pair micro-F1 | Overall exact-set accuracy |
|---|---:|---:|---:|---:|
| Validation | 0.888545 | 0.836697 | 0.752322 | 0.689922 |
| Untouched test | 0.869984 | 0.808194 | 0.722311 | 0.639535 |
| Legacy baseline on test | 0.815331 | 0.805293 | 0.672474 | 0.581395 |

Overall pair micro-F1 change versus legacy baseline: +0.049838.

Runtime: 26.8 minutes.

The test split was used only after checkpoint and threshold selection. Results come from one seed (`42`).
