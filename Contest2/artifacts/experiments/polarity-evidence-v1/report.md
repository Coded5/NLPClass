# Validation-calibrated polarity evidence experiment

Evidence targets encode positive and negative evidence independently: positive `10`, negative `01`, neutral `00`, and conflict `11`. Aspect checkpoints and thresholds were frozen.

Selected lambda: `0.25`. Locked winner: `calibrated_original`.

## Validation

| System | Pair F1 | Polarity F1 | Exact set | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|
| original | 0.765891 | 0.830882 | 0.686047 | 0.702703 | 0.466667 |
| calibrated_original | 0.778295 | 0.845173 | 0.705426 | 0.746988 | 0.500000 |
| evidence_raw | 0.765891 | 0.850909 | 0.697674 | 0.736842 | 0.466667 |
| evidence_calibrated | 0.772093 | 0.856624 | 0.705426 | 0.753247 | 0.482759 |

## Held-out test

| System | Pair F1 | Polarity F1 | Exact set | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|
| original | 0.754358 | 0.811111 | 0.662791 | 0.640000 | 0.500000 |
| calibrated_original | 0.744849 | 0.808118 | 0.662791 | 0.658824 | 0.434783 |
| evidence_raw | 0.751189 | 0.808989 | 0.658915 | 0.648649 | 0.428571 |
| evidence_calibrated | 0.754358 | 0.812734 | 0.662791 | 0.648649 | 0.444444 |

Selection used validation only. Test results are analysis-only and did not change the locked winner. This test split was accessed by prior experiments.

Runtime: 50.9 minutes.
