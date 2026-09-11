# roberta-aspect-mixed-polarity

## End-to-end evaluation

| Split | Pair micro-F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|
| Validation | 0.779014 | 0.736842 | 0.466667 |
| Test | 0.762987 | 0.608696 | 0.518519 |

Thresholds and model checkpoints were locked on validation before test labels were loaded.
Shared orchestration runtime: 61.4 minutes.

These are official end-to-end metrics. They include aspect detection, and the
polarity columns use the evaluator's `(id, polarity)` set semantics.

## Gold-aspect polarity comparison

The polarity backbones were also compared on the same 316 held-out test aspect
rows with the correct aspect supplied to each classifier. This measures strict
four-class polarity performance without aspect-detection errors.

| Polarity model | Accuracy | Macro-F1 | Positive F1 | Negative F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
| RoBERTa, 3 seeds | 0.8354 | 0.7088 | 0.9167 | 0.7785 | 0.6400 | 0.5000 |
| DeBERTa-v3, 3 seeds | 0.8418 | 0.7187 | 0.9186 | **0.8138** | **0.6575** | 0.4848 |
| Equal-logit mixture | **0.8449** | **0.7215** | **0.9267** | 0.7919 | 0.6486 | **0.5185** |

DeBERTa is marginally better than RoBERTa overall, and the equal-logit mixture
is the strongest polarity classifier. The mixture correctly classifies 267 of
316 rows, compared with 266 for DeBERTa and 264 for RoBERTa.

An earlier ad-hoc audit reported DeBERTa accuracy as `0.6044`; that number was
invalid because the five aspect candidates per review were not correctly
aligned with the gold-aspect rows. The corrected comparison explicitly maps
each `(review ID, gold aspect)` to its candidate logit.

Reproducible outputs:

- `scripts/compare_polarity_backbones.py`
- `artifacts/experiments/deberta-v3-three-systems-v1/polarity_backbone_comparison.json`
- `artifacts/experiments/deberta-v3-three-systems-v1/systems/roberta-aspect-mixed-polarity/polarity_error_audit.md`
