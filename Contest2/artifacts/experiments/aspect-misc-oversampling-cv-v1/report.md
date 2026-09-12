# Targeted multi-aspect miscellaneous oversampling

## Train-only five-fold CV

| Configuration | Aspect micro-F1 | Misc F1 | Multi-misc recall | Fold non-regressions | Eligible |
|---|---:|---:|---:|---:|---|
| reference | 0.878770 | 0.837027 | 0.400000 | - | True |
| sampling-control | 0.881454 | 0.837750 | 0.421053 | - | False |
| targeted-2x | 0.883410 | 0.850814 | 0.431579 | 4 | True |
| targeted-3x | 0.885714 | 0.852713 | 0.442105 | 4 | True |
| targeted-4x | 0.880233 | 0.843564 | 0.442105 | 3 | False |

CV winner: `targeted-3x`.

## Three-seed confirmation

| System | Val aspect F1 | Val subgroup recall | Val pair F1 | Test aspect F1 | Test subgroup recall | Test pair F1 |
|---|---:|---:|---:|---:|---:|---:|
| reference | 0.894488 | 0.583333 | 0.779014 | 0.877419 | 0.285714 | 0.762987 |
| targeted-3x | 0.890966 | 0.666667 | 0.770671 | 0.874404 | 0.428571 | 0.744783 |

Locked winner: `reference`. Test was not used for selection.
