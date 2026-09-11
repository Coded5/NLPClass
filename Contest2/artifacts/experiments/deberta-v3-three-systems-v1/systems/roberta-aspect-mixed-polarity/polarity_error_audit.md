# Polarity Backbone Comparison and Error Audit

## Correction

An earlier version reported DeBERTa accuracy as 0.6044. That result was invalid
because the ad-hoc analysis did not reliably align each review's five polarity
candidates with the annotated gold aspect. The corrected comparison uses
`scripts/compare_polarity_backbones.py`, which explicitly maps every
`(review ID, gold aspect)` to its corresponding candidate logit.

## Scope

All models are evaluated on the same 316 held-out test aspect rows. Each model
receives the review text and gold aspect, then predicts `positive`, `negative`,
`neutral`, or `conflict`. Aspect detection therefore cannot affect the result.
This differs from end-to-end pair F1 and the official `(id, polarity)` metric.

## Correct comparison

| Model | Accuracy | Macro-F1 | Positive F1 | Negative F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
| RoBERTa, 3 seeds | 0.8354 | 0.7088 | 0.9167 | 0.7785 | 0.6400 | 0.5000 |
| DeBERTa-v3, 3 seeds | 0.8418 | 0.7187 | 0.9186 | **0.8138** | **0.6575** | 0.4848 |
| Equal-logit mixture | **0.8449** | **0.7215** | **0.9267** | 0.7919 | 0.6486 | **0.5185** |

DeBERTa is slightly better than RoBERTa: +0.63 accuracy points and +0.99
macro-F1 points. The mixture is best overall, improving on RoBERTa by +0.95
accuracy points and +1.27 macro-F1 points. These are small differences:
DeBERTa gets two more rows correct, and the mixture gets three more.

## Confusion matrices

Rows are gold labels; columns are predicted positive, negative, neutral, and
conflict.

| Model | Positive row | Negative row | Neutral row | Conflict row |
|---|---|---|---|---|
| RoBERTa | 176, 6, 5, 0 | 9, 58, 5, 0 | 8, 7, 24, 1 | 4, 6, 1, 6 |
| DeBERTa | 175, 5, 2, 5 | 6, 59, 5, 2 | 10, 5, 24, 1 | 3, 4, 2, 8 |
| Mixture | 177, 6, 3, 1 | 6, 59, 6, 1 | 8, 7, 24, 1 | 4, 5, 1, 7 |

## Class-level interpretation

- **Positive:** the mixture is best, finding 177/187 with F1 0.9267.
- **Negative:** DeBERTa is best, finding 59/72 with F1 0.8138.
- **Neutral:** every model correctly classifies 24/40. DeBERTa has the best F1
  because it makes fewer false-neutral predictions.
- **Conflict:** DeBERTa finds the most (8/17) but also predicts conflict
  incorrectly eight times. RoBERTa finds 6/17 with higher precision. The
  mixture balances them and has the best conflict F1, 0.5185.

Conflict remains weakest: all models have recall below 0.50 on only 17 gold
examples. Neutral recall remains 0.60 for all three, so changing the backbone
alone did not improve neutral detection.

## Conclusion

DeBERTa is **not worse** in the corrected gold-aspect polarity evaluation. It is
marginally better than RoBERTa overall, while the equal-logit mixture is best.
DeBERTa particularly helps negative and conflict recall; RoBERTa is more precise
on conflict.

Raw results are in
`artifacts/experiments/deberta-v3-three-systems-v1/polarity_backbone_comparison.json`.
