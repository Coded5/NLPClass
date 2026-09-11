# Aspect Error Audit

## Scope

This audit covers the current winning end-to-end system's aspect component: the
three-seed RoBERTa multilabel ensemble with pair-optimized thresholds
`[0.55, 0.80, 0.90, 0.75, 0.75]` for ambience,
anecdotes/miscellaneous, food, price, and service. It uses the exactly
reproduced held-out test predictions from
`reevaluations/current-winner-20260911`.

Polarity is ignored. Each `(review ID, aspect)` is scored once, so polarity
errors cannot contaminate this analysis. The test set is used only for
diagnosis; no model or threshold is selected from these findings.

## Overall result

| Measure | Value |
|---|---:|
| Review IDs | 258 |
| Gold aspect pairs | 316 |
| Predicted aspect pairs | 300 |
| True positives | 269 |
| False positives | 31 |
| False negatives | 47 |
| Micro precision | 0.8967 |
| Micro recall | 0.8513 |
| Micro-F1 | 0.8734 |
| Exact aspect sets | 204/258 (0.7907) |

The model is conservative: it emits 16 fewer aspects than exist in gold, and
misses exceed false positives by 47 to 31.

### Complete aggregate metrics

| Aggregation | Precision | Recall | F1 |
|---|---:|---:|---:|
| Micro | 0.896667 | 0.851266 | 0.873377 |
| Macro across five aspects | 0.882071 | 0.850972 | 0.865092 |
| Sample average across review IDs | 0.899225 | 0.871124 | 0.875286 |

| Additional multilabel metric | Value |
|---|---:|
| Micro Jaccard index | 0.775216 |
| Sample-average Jaccard index | 0.854328 |
| Hamming loss | 0.060465 |
| Exact aspect-set accuracy | 0.790698 |
| Gold labels per review | 1.224806 |
| Predicted labels per review | 1.162791 |

Micro scores pool all aspect decisions. Macro scores weight each aspect
equally. Sample-average scores first calculate each review's set score and then
average across 258 IDs. Hamming loss is the fraction of the 1,290 binary
ID-aspect decisions that are wrong.

## Per-aspect results

| Aspect | Gold | Pred. | TP | FP | FN | TN | Precision | Recall | F1 | Specificity | Binary accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ambience | 37 | 37 | 31 | 6 | 6 | 215 | 0.837838 | 0.837838 | 0.837838 | 0.972851 | 0.953488 |
| anecdotes/miscellaneous | 96 | 81 | 74 | 7 | **22** | 155 | **0.913580** | **0.770833** | 0.836158 | 0.956790 | 0.887597 |
| food | 105 | 106 | 98 | 8 | 7 | 145 | 0.924528 | 0.933333 | **0.928910** | 0.947712 | 0.941860 |
| price | 28 | 29 | 25 | 4 | 3 | 226 | 0.862069 | 0.892857 | 0.877193 | 0.982609 | 0.972868 |
| service | 50 | 47 | 41 | 6 | 9 | 202 | 0.872340 | 0.820000 | 0.845361 | 0.971154 | 0.941860 |

Food is the strongest aspect. Anecdotes/miscellaneous has the weakest recall
and contributes 22/47 misses (46.8%), despite having the highest precision.
This indicates underprediction rather than indiscriminate use of the catch-all
label.

## Single- versus multi-aspect reviews

| Review group | IDs | TP | FP | FN | Precision | Recall | F1 | Exact aspect set |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Single-aspect | 207 | 186 | 29 | 21 | 0.865116 | 0.898551 | 0.881517 | 178/207 (0.859903) |
| Multi-aspect | 51 | 83 | 2 | 26 | 0.976471 | 0.761468 | 0.855670 | 26/51 (0.509804) |

Multi-aspect reviews are the clearest weakness. Of the 25 incorrect
multi-aspect reviews:

- 23 contain only missing aspects and no extra prediction;
- 2 contain both a missing and an extra aspect;
- none contain only an extra aspect;
- 20 two-aspect reviews are reduced to one predicted aspect.

The model usually recognizes the dominant aspect but fails to emit a secondary
one. This is a label-cardinality/recall problem, not broad overprediction.

### Label cardinality

| Number of aspects | Gold review count | Predicted review count |
|---:|---:|---:|
| 1 | 207 | 220 |
| 2 | 45 | 34 |
| 3 | 5 | 4 |
| 4 | 1 | 0 |

Every review receives at least one prediction because the decoder falls back to
the highest-scoring aspect when no threshold is crossed. The excess of
one-label predictions and absence of four-label predictions quantify the
under-cardinality tendency.

### Error-set shapes

| Error shape | Review IDs |
|---|---:|
| Exact set | 204 |
| Missing aspects only | 23 |
| Extra aspects only | 8 |
| Both missing and extra aspects | 23 |

All 23 missing-only cases are multi-aspect reviews. Among single-aspect errors,
21 are substitutions and 8 retain the correct aspect but add another label.

## Recurring error patterns

### 1. Overall sentiment is not recognized as anecdotes/miscellaneous

The model often predicts a concrete aspect but omits the simultaneously
annotated overall or miscellaneous judgment:

- ID 304: predicts food but misses miscellaneous in “enjoyed everything.”
- ID 1213: finds food, price, and service but misses the review-level
  disappointment.
- ID 1438: finds food but misses the broader restaurant judgment.
- ID 2777: finds food but misses “all the other deficiencies.”
- ID 3435: finds food but misses “an overall good restaurant.”

Ten miscellaneous misses occur in multi-aspect reviews. Another twelve occur
in single-aspect reviews, commonly because concrete words such as “meal,”
“dumpling,” “chili,” “waiter,” or “prices” pull the prediction toward food,
service, or price. Among one-label substitutions, miscellaneous is changed to
food five times and to service four times.

### 2. Indirect aspect references are missed

Several annotations require reasoning beyond explicit aspect nouns:

- **Service:** custom-order refusal (ID 347), waiting (2053 and 2271), visiting
  regulars (3505), billing mistakes (3060), and restaurant turnover policy
  (3450).
- **Ambience:** patrons (1160), eating with coats on (2220), live jazz (2511),
  decor/signage (2808), and a generic statement that “the place ... was great”
  (3681).
- **Food:** “worth it once you take a bite” (1035), itemized dishes (1964 and
  2101), portion aftermath (3199), and complimentary dessert (3663).
- **Price:** prix-fixe wording (878), “affordable” (2464), and “on the house”
  (3663).

### 3. Lexical cues belong to a different aspect

Strong surface words sometimes create a plausible but wrong aspect:

- ID 3060 is annotated service for an erroneous bill, but `$60` and `$80`
  trigger price.
- ID 3450 is annotated service for a no-dessert policy, but dessert terms
  trigger food.
- ID 2432 is annotated service for the manager's behavior, while “noise level”
  adds a false ambience prediction.
- ID 1727 is annotated ambience for a DJ, while “takes requests” adds false
  service.
- ID 1077 is annotated price, while “dosa” adds false food.

These examples expose fuzzy annotation boundaries as well as model errors. A
few predictions are semantically defensible even when they do not match the
gold set.

### 4. Ambience and miscellaneous overlap

Generic experiential statements and entertainment references frequently cross
this boundary. IDs 1160, 2220, and 2511 change ambience to miscellaneous;
IDs 2570 and 2674 change miscellaneous to ambience. IDs 811, 2404, and 2707
receive both labels when gold contains only one. This boundary accounts for a
substantial share of the single-aspect errors.

## Effect of aspect-only thresholds

The same backbone has separately locked aspect-only thresholds
`[0.55, 0.75, 0.75, 0.70, 0.75]`. Applying those predictions gives:

| Threshold objective | Precision | Recall | F1 | Exact sets |
|---|---:|---:|---:|---:|
| Current pair-optimized | 0.8967 | 0.8513 | 0.8734 | 204 |
| Aspect-only optimized | 0.8947 | 0.8608 | **0.8774** | **205** |

Only four review predictions differ. Aspect-only decoding recovers service for
IDs 2271 and 3450 and ambience for ID 2808, while adding false ambience for ID
1800. It therefore gains three true positives for one false positive.

For an aspect-only deliverable, the aspect-optimized thresholds are the current
best choice. For the full ABSA system, the pair-optimized thresholds remain the
proper locked configuration because they were selected against complete
aspect-polarity pairs.

## Recommended next experiment

The next aspect experiment should target secondary-label recall without
substantially reducing precision:

1. Keep the current three-seed RoBERTa ensemble as the control.
2. Add validation-selected cardinality-aware decoding, such as a small label
   count head or a top-two rule gated by calibrated probabilities.
3. Tune per-aspect thresholds with explicit recall floors for
   anecdotes/miscellaneous and service, then evaluate pair F1 as a guardrail.
4. Stratify validation diagnostics by one versus multiple gold aspects so a
   gain on easy single-aspect reviews cannot hide secondary-label regression.
5. Add targeted training examples for implicit service, ambience, and price
   references, plus hard negatives where an aspect word belongs to another
   target.
6. Lock the decoder on validation before one final test evaluation.

The highest-value target is not food classification. It is recovering
miscellaneous and service as secondary labels in multi-aspect reviews while
preserving the ensemble's current precision.

The probability-level root-cause analysis is in `misc_conservatism_audit.md`.
