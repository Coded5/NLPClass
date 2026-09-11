# Class imbalance and joint ABSA experiment

## Loss screening (seed 42)

| Separate polarity loss | Validation pair micro-F1 | Best epoch |
|---|---:|---:|
| weighted-ce | 0.773994 | 30 |
| weighted-focal-g1 | 0.761610 | 45 |
| weighted-focal-g2 | 0.767802 | 40 |
| class-balanced-focal-b0.999-g2 | 0.755418 | 30 |

Selected non-baseline alternative: `weighted-focal-g2`.

## Joint architecture screening (seed 42)

| Joint polarity loss | Validation pair micro-F1 | Best epoch |
|---|---:|---:|
| weighted-ce | 0.775701 | 40 |
| weighted-focal-g2 | 0.783570 | 35 |

## Three-seed confirmation

| Architecture | Loss | Validation mean | Validation std. dev. | Ensemble validation F1 |
|---|---|---:|---:|---:|
| separate | weighted-ce | 0.761108 | 0.009310 | 0.755418 |
| joint | weighted-focal-g2 | 0.770227 | 0.009720 | 0.785047 |

Seeds: `[17, 42, 73]`. The final configuration was locked as
`joint` before test inference.

## Untouched test results

| System | Pair micro-F1 | Exact-set accuracy | Single-aspect exact | Multi-aspect exact |
|---|---:|---:|---:|---:|
| separate | 0.744783 | 0.655039 | 0.714976 | 0.411765 |
| joint | 0.724409 | 0.627907 | 0.700483 | 0.333333 |
| existing_pipeline | 0.722311 | 0.639535 | 0.710145 | 0.352941 |
| legacy_baseline | 0.672474 | 0.581395 | 0.724638 | 0.000000 |

### Pair-level polarity breakdown

| System | Polarity | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| separate | positive | 0.817708 | 0.839572 | 0.828496 | 187 |
| separate | negative | 0.680000 | 0.708333 | 0.693878 | 72 |
| separate | neutral | 0.593750 | 0.475000 | 0.527778 | 40 |
| separate | conflict | 0.625000 | 0.294118 | 0.400000 | 17 |
| joint | positive | 0.827957 | 0.823529 | 0.825737 | 187 |
| joint | negative | 0.569767 | 0.680556 | 0.620253 | 72 |
| joint | neutral | 0.571429 | 0.500000 | 0.533333 | 40 |
| joint | conflict | 0.583333 | 0.411765 | 0.482759 | 17 |
| existing_pipeline | positive | 0.826087 | 0.812834 | 0.819407 | 187 |
| existing_pipeline | negative | 0.692308 | 0.625000 | 0.656934 | 72 |
| existing_pipeline | neutral | 0.477273 | 0.525000 | 0.500000 | 40 |
| existing_pipeline | conflict | 0.500000 | 0.411765 | 0.451613 | 17 |
| legacy_baseline | positive | 0.847134 | 0.711230 | 0.773256 | 187 |
| legacy_baseline | negative | 0.672727 | 0.513889 | 0.582677 | 72 |
| legacy_baseline | neutral | 0.513514 | 0.475000 | 0.493506 | 40 |
| legacy_baseline | conflict | 0.444444 | 0.235294 | 0.307692 | 17 |

Joint minus separate paired-bootstrap delta: -0.020271
(95% CI -0.051282 to
+0.009861).

Locked winner minus existing pipeline delta: +0.002285
(95% CI -0.031510 to
+0.037239).

Runtime: 85.5 minutes. All configuration and threshold choices
used validation data only; test results were generated after `locked_selection.json`.
