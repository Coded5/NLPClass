# Multi-label aspect plus existing polarity experiment

## Validation selection

| Candidate | Pair micro-F1 | Exact-set accuracy | Pair-tuned thresholds |
|---|---:|---:|---|
| one_seed | 0.757098 | 0.705426 | `[0.55, 0.8, 0.75, 0.7, 0.75]` |
| three_seed | 0.765891 | 0.686047 | `[0.55, 0.8, 0.9, 0.7, 0.45]` |

Locked polarity candidate: `three_seed`.

## Held-out test

| System | Aspect micro-F1 | Polarity micro-F1 | Pair micro-F1 | Exact-set accuracy | Multi-aspect exact |
|---|---:|---:|---:|---:|---:|
| New composed winner | 0.874802 | 0.811111 | 0.754358 | 0.662791 | 0.431373 |
| Previous one-seed pipeline | 0.869984 | 0.808194 | 0.722311 | 0.639535 | 0.352941 |
| Previous separate ensemble | 0.869984 | 0.816327 | 0.744783 | 0.655039 | 0.411765 |
| Previous joint ensemble | 0.856693 | 0.794824 | 0.724409 | 0.627907 | 0.333333 |
| Legacy single-label | 0.815331 | 0.805293 | 0.672474 | 0.581395 | 0.000000 |

Oracle-aspect polarity pair micro-F1: 0.835443.

## Pair-level polarity breakdown

| System | Polarity | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| New composed winner | positive | 0.833333 | 0.855615 | 0.844327 | 187 |
| New composed winner | negative | 0.658228 | 0.722222 | 0.688742 | 72 |
| New composed winner | neutral | 0.588235 | 0.500000 | 0.540541 | 40 |
| New composed winner | conflict | 0.600000 | 0.352941 | 0.444444 | 17 |
| Previous one-seed pipeline | positive | 0.826087 | 0.812834 | 0.819407 | 187 |
| Previous one-seed pipeline | negative | 0.692308 | 0.625000 | 0.656934 | 72 |
| Previous one-seed pipeline | neutral | 0.477273 | 0.525000 | 0.500000 | 40 |
| Previous one-seed pipeline | conflict | 0.500000 | 0.411765 | 0.451613 | 17 |
| Previous separate ensemble | positive | 0.817708 | 0.839572 | 0.828496 | 187 |
| Previous separate ensemble | negative | 0.680000 | 0.708333 | 0.693878 | 72 |
| Previous separate ensemble | neutral | 0.593750 | 0.475000 | 0.527778 | 40 |
| Previous separate ensemble | conflict | 0.625000 | 0.294118 | 0.400000 | 17 |
| Previous joint ensemble | positive | 0.827957 | 0.823529 | 0.825737 | 187 |
| Previous joint ensemble | negative | 0.569767 | 0.680556 | 0.620253 | 72 |
| Previous joint ensemble | neutral | 0.571429 | 0.500000 | 0.533333 | 40 |
| Previous joint ensemble | conflict | 0.583333 | 0.411765 | 0.482759 | 17 |
| Legacy single-label | positive | 0.847134 | 0.711230 | 0.773256 | 187 |
| Legacy single-label | negative | 0.672727 | 0.513889 | 0.582677 | 72 |
| Legacy single-label | neutral | 0.513514 | 0.475000 | 0.493506 | 40 |
| Legacy single-label | conflict | 0.444444 | 0.235294 | 0.307692 | 17 |

Winner minus previous one-seed pipeline: +0.032053
(95% CI +0.001669 to +0.063628).

Winner minus legacy baseline: +0.081640
(95% CI +0.040456 to +0.124683).

Polarity candidate selection and pair-threshold tuning used validation data only.
This test split has been evaluated by previous experiments and is therefore held-out
for this run, but not globally untouched.

Runtime: 0.8 minutes.
