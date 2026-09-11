# Neutral and conflict polarity reweighting

## Weighting

The existing inverse-frequency weights were multiplied only for neutral and
conflict. The validation pair-F1 tolerance was `0.005`.

Baseline validation pair micro-F1: 0.765891.
Baseline validation minority F1: 0.584685.

| Candidate | Neutral multiplier | Conflict multiplier | Pair F1 | Minority F1 | Eligible |
|---|---:|---:|---:|---:|---|
| weighted-ce-n1.25-c1.5 | 1.25 | 1.5 | 0.755832 | 0.574031 | False |
| weighted-ce-n1.5-c2 | 1.5 | 2 | 0.786936 | 0.579747 | True |
| weighted-ce-n2-c2 | 2 | 2 | 0.778125 | 0.595660 | False |
| weighted-ce-n1.5-c3 | 1.5 | 3 | 0.771384 | 0.568008 | False |
| weighted-ce-n2-c3 | 2 | 3 | 0.765163 | 0.546474 | False |

Selected screening candidate: `weighted-ce-n1.5-c2`.

## Three-seed validation

| System | Pair F1 | Minority F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|
| baseline | 0.765891 | 0.584685 | 0.702703 | 0.466667 |
| reweighted | 0.777605 | 0.595373 | 0.753247 | 0.437500 |

Locked winner: `reweighted`.

## Held-out test

| System | Pair F1 | Polarity F1 | Exact set | Minority F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.754358 | 0.811111 | 0.662791 | 0.570000 | 0.640000 | 0.500000 |
| reweighted | 0.735669 | 0.799263 | 0.662791 | 0.527677 | 0.710526 | 0.344828 |

Selection and threshold tuning used validation only. This test split has been
evaluated by earlier experiments, so it is not globally untouched.

Runtime: 64.2 minutes.
