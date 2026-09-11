# Why the Aspect Model Is Conservative on Anecdotes/Miscellaneous

## Finding

The model is conservative primarily because multi-aspect miscellaneous labels
are a rare and semantically weak training pattern. Concrete food, service,
price, or ambience cues usually occur without a miscellaneous label, so the
binary miscellaneous head learns that those cues are negative evidence. In
multi-aspect reviews, the concrete aspect dominates and the broader restaurant
judgment receives a genuinely low score.

The `0.80` pair-optimized threshold reinforces this behavior, but does not cause
most of it. Most missed multi-aspect miscellaneous scores are far below any
reasonable threshold, and all three ensemble seeds agree on the low scores.

## Data evidence

Miscellaneous is not globally rare:

| Split | Review IDs | Misc. IDs | Prevalence | Misc. only | Misc. with another aspect |
|---|---:|---:|---:|---:|---:|
| Train | 2,068 | 763 | 36.9% | 668 | 95 |
| Validation | 258 | 95 | 36.8% | 83 | 12 |
| Test | 258 | 96 | 37.2% | 82 | 14 |

However, only 95/763 training miscellaneous examples (12.5%) are multi-aspect.
Conditional co-occurrence is especially sparse:

| Concrete aspect in training | Reviews with aspect | Also miscellaneous | Conditional rate |
|---|---:|---:|---:|
| ambience | 294 | 22 | 7.5% |
| food | 841 | 42 | 5.0% |
| price | 220 | 22 | 10.0% |
| service | 405 | 22 | 5.4% |

Thus, 95% of training food reviews and 94.6% of service reviews teach the
miscellaneous output to remain negative. The model can technically emit both
labels, but the observed data makes that combination unusual.

## Test recall collapse by label cardinality

| Gold miscellaneous context | Correct | Total | Recall |
|---|---:|---:|---:|
| Sole aspect | 70 | 82 | **0.8537** |
| Co-occurs with another aspect | 4 | 14 | **0.2857** |
| All miscellaneous | 74 | 96 | 0.7708 |

Ten of the 14 multi-aspect miscellaneous annotations are missed. They account
for almost half of all 22 miscellaneous false negatives despite being only
14.6% of the class's test examples.

## Probability evidence

The three-seed ensemble's miscellaneous probabilities differ sharply by
context:

| Split and group | Mean | Median | P25 | P75 |
|---|---:|---:|---:|---:|
| Validation, misc. only | 0.9279 | 0.9975 | 0.9960 | 0.9979 |
| Validation, multi-aspect misc. | 0.6370 | 0.8755 | 0.3870 | 0.9508 |
| Test, misc. only | 0.8889 | 0.9972 | 0.9852 | 0.9979 |
| Test, multi-aspect misc. | **0.3952** | **0.1850** | **0.0341** | 0.8813 |
| Test, non-miscellaneous | 0.1229 | 0.0317 | 0.0124 | 0.0856 |

The validation and test multi-aspect groups contain only 12 and 14 reviews,
respectively. Their median scores shift from 0.8755 to 0.1850. This rare
subgroup is too small and heterogeneous for one validation split to estimate
its behavior reliably.

Among the 14 test multi-aspect miscellaneous examples:

| Probability bucket | Count |
|---|---:|
| Below 0.10 | 6 |
| 0.10 to below 0.50 | 3 |
| 0.50 to below 0.80 | 1 |
| At least 0.80 | 4 |

Only one miss lies moderately close to the threshold. Nine have scores below
0.50, including six below 0.10. This is learned suppression, not ordinary
threshold clipping.

## Seed agreement

Individual-seed recall at threshold `0.80` is stable but poor for this subgroup:

| Split/group | Seed 42 | Seed 43 | Seed 44 |
|---|---:|---:|---:|
| Test miscellaneous only | 0.8659 | 0.8659 | 0.8537 |
| Test multi-aspect miscellaneous | 0.2857 | 0.3571 | 0.2857 |

For the six lowest-scored test examples—IDs 1438, 304, 3512, 2777, 967, and
25—all three seeds assign probabilities below 0.10. Averaging is not cancelling
a correct minority model; the ensemble members learned the same shortcut.

## What the model is using

### Concrete lexical cues dominate

- IDs 304, 967, 1438, 2777, and 3435 contain food words and are reduced to
  `food` even though gold also marks a broader restaurant judgment.
- IDs 25 and 3512 contain waiter/service language and are reduced to `service`.
- ID 1303 contains explicit prices and is reduced to `price`.
- ID 1857 begins with ambience language and is reduced to `ambience`.

### Miscellaneous evidence is abstract or discourse-level

The missed evidence is usually expressed through phrases such as “everything,”
“overall,” “this place,” “all the other deficiencies,” or a recommendation.
Unlike food or price, miscellaneous has no stable entity vocabulary. It often
requires interpreting the scope of an evaluation across the whole restaurant
rather than locating a noun.

### Annotation boundaries are fuzzy

Some single-aspect miscellaneous errors contain strong concrete vocabulary:
“meal,” “dumpling,” “chili,” “waiter,” and “prices.” The model's concrete
prediction is often linguistically plausible even when the gold annotation
uses miscellaneous. This increases label noise at the class boundary.

## Threshold analysis

Without the decoder fallback, the raw test miscellaneous operating points are:

| Threshold | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 80 | 15 | 16 | 0.8421 | 0.8333 | 0.8377 |
| 0.75 | 73 | 6 | 23 | 0.9241 | 0.7604 | 0.8343 |
| 0.80 | 73 | 6 | 23 | 0.9241 | 0.7604 | 0.8343 |
| 0.90 | 69 | 4 | 27 | 0.9452 | 0.7188 | 0.8166 |

The final decoder's fallback adds one true and one false miscellaneous label,
producing TP 74, FP 7, FN 22, and F1 0.8362.

Validation selected `0.75` for aspect-only decoding, where miscellaneous F1 was
0.8617. The complete pair decoder selected `0.80`. Test-retrospective threshold
search would prefer `0.55`, but using that result would be test leakage. More
importantly, lowering the threshold cannot recover the six multi-aspect cases
below 0.10 without a large false-positive increase.

## Root-cause ranking

1. **Sparse multi-aspect miscellaneous supervision:** only 95 training reviews
   demonstrate that miscellaneous should coexist with a concrete aspect.
2. **Semantic competition:** concrete lexical cues are easier than broad,
   discourse-level miscellaneous evidence.
3. **Rare-subgroup validation variance:** validation and test behavior differs
   sharply across only 12 versus 14 multi-aspect examples.
4. **Threshold objective:** pair optimization favors precision and adds a small
   recall penalty, but is not the main source of low scores.
5. **Global class weighting:** miscellaneous receives positive weight 1.71
   because it is common, but food receives an even lower weight (1.46) and has
   excellent recall. Weighting alone does not explain the failure.

## Recommended experiment

Do not oversample all miscellaneous examples; most are easy single-aspect cases
that the model already handles. Instead:

1. Oversample or loss-weight only reviews where miscellaneous co-occurs with a
   concrete aspect.
2. Use grouped cross-validation to tune this subgroup weight because a single
   validation fold contains only 12 examples.
3. Track multi-aspect miscellaneous recall as a guardrail alongside overall
   aspect F1 and end-to-end pair F1.
4. Compare the weighted control with a cardinality-aware decoder or auxiliary
   label-count head.
5. Include hard negatives: concrete-aspect reviews without a miscellaneous
   annotation, so improved recall does not turn into indiscriminate dual labels.
6. Lock the configuration on validation/OOF predictions before evaluating the
   historical test split.

The most targeted first experiment is subgroup weighting for
`miscellaneous + any concrete aspect`, not a lower global threshold and not
global miscellaneous oversampling.
