# ABSA learning-curve diagnostic

This is five-fold grouped out-of-fold evaluation on the original training partition.
The historical validation and test partitions were not used.

| Training data | Aspect micro-F1 | Polarity macro-F1 | Pair micro-F1 | Pooled pair F1 |
|---|---:|---:|---:|---:|
| 25% | 0.8607 +/- 0.0091 | 0.6053 +/- 0.0280 | 0.6915 +/- 0.0153 | 0.6916 |
| 50% | 0.8760 +/- 0.0101 | 0.6810 +/- 0.0143 | 0.7312 +/- 0.0185 | 0.7311 |
| 75% | 0.8778 +/- 0.0066 | 0.6858 +/- 0.0605 | 0.7248 +/- 0.0392 | 0.7248 |
| 100% | 0.8818 +/- 0.0178 | 0.6935 +/- 0.0357 | 0.7499 +/- 0.0231 | 0.7499 |

## Aspect detail

| Data | Macro-F1 | Exact set | Multi exact | Misc F1 | Gold cardinality | Predicted cardinality |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 0.8585 | 0.7490 | 0.6077 | 0.8191 | 1.220 | 1.292 |
| 50% | 0.8717 | 0.7805 | 0.6436 | 0.8397 | 1.220 | 1.275 |
| 75% | 0.8737 | 0.7771 | 0.6718 | 0.8377 | 1.220 | 1.309 |
| 100% | 0.8776 | 0.7940 | 0.6436 | 0.8452 | 1.220 | 1.273 |

## Gold-aspect polarity detail

| Data | Accuracy | Macro-F1 | Positive F1 | Negative F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 0.7842 | 0.6094 | 0.8862 | 0.7397 | 0.5523 | 0.2594 |
| 50% | 0.8206 | 0.6830 | 0.9026 | 0.7816 | 0.6579 | 0.3900 |
| 75% | 0.8162 | 0.6823 | 0.9041 | 0.7794 | 0.6430 | 0.4028 |
| 100% | 0.8321 | 0.6941 | 0.9170 | 0.7935 | 0.6545 | 0.4113 |

## Composed-pair detail

| Data | Exact set | Single exact | Multi exact | Positive F1 | Negative F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 25% | 0.6122 | 0.6549 | 0.4282 | 0.7924 | 0.6222 | 0.4647 | 0.2578 |
| 50% | 0.6596 | 0.7020 | 0.4769 | 0.8177 | 0.6586 | 0.5743 | 0.3700 |
| 75% | 0.6494 | 0.6907 | 0.4718 | 0.8213 | 0.6554 | 0.5455 | 0.3763 |
| 100% | 0.6784 | 0.7223 | 0.4897 | 0.8365 | 0.6889 | 0.5837 | 0.3874 |

| Comparison | Pair-F1 delta | Paired review bootstrap 95% CI |
|---|---:|---:|
| 25% to 50% | +0.0395 | [+0.0252, +0.0536] |
| 50% to 75% | -0.0063 | [-0.0196, +0.0068] |
| 75% to 100% | +0.0250 | [+0.0131, +0.0373] |

## 75% to 100% diagnosis

| Component | Metric | Delta | Paired bootstrap 95% CI | Diagnosis |
|---|---|---:|---:|---|
| aspect | micro_f1 | +0.0038 | [-0.0038, +0.0113] | inconclusive |
| polarity | macro_f1 | +0.0118 | [-0.0119, +0.0351] | inconclusive |
| pair | micro_f1 | +0.0250 | [+0.0131, +0.0373] | data-limited |

Primary 75% to 100% diagnosis: **data-limited**.

Data-limited requires a gain of at least 0.01 with the interval excluding zero.
Plateaued means the interval's upper bound is below 0.01. Otherwise the result is inconclusive.
