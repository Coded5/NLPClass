# Test error analysis

## Scope

This analysis uses the untouched test split after model and threshold selection.
It analyzes the final multilabel-aspect and aspect-conditioned-polarity pipeline,
not the legacy baseline. The test split contains 258 reviews and 316 gold
`(aspect, polarity)` pairs.

## Executive summary

The model predicted 307 pairs. Of these, 225 were correct, 82 were false
positives, and 91 gold pairs were missed. This gives the official overall
micro-F1 of 0.722311.

The missed pairs divide almost evenly between the two pipeline stages:

| Source of missed gold pair | Count | Share of 91 false negatives |
|---|---:|---:|
| Aspect was not detected | 45 | 49.5% |
| Aspect was detected but polarity was wrong | 46 | 50.5% |

The 82 false-positive pairs similarly consist of 36 extra aspects and 46
replacement pairs created by incorrect polarity predictions. Improving only
the polarity loss can therefore address roughly half of the current pair-level
errors; aspect errors still impose a substantial ceiling.

## Aspect errors

Aspect detection produced 271 true positives, 36 false positives, and 45 false
negatives. Food is the strongest aspect. Ambience and miscellaneous mentions
are the most difficult to recover as correct complete pairs.

| Gold aspect | Support | Missed aspect | Wrong polarity after detection | Exact-pair recall |
|---|---:|---:|---:|---:|
| food | 105 | 6 | 12 | 0.829 |
| price | 28 | 4 | 5 | 0.679 |
| service | 50 | 8 | 11 | 0.620 |
| ambience | 37 | 7 | 8 | 0.595 |
| anecdotes/miscellaneous | 96 | 20 | 10 | 0.688 |

The high miscellaneous miss count indicates that some examples express a
general restaurant-level judgment alongside a specific aspect. For example,
review 1213 explicitly mentions service, food, and price, while the gold data
also assigns a neutral miscellaneous label. Such implicit labels are harder to
identify from lexical cues.

## Polarity errors

The polarity training split is materially imbalanced:

| Polarity | Training examples | Share | Current cross-entropy weight |
|---|---:|---:|---:|
| positive | 1,502 | 59.5% | 0.420 |
| negative | 571 | 22.6% | 1.105 |
| neutral | 318 | 12.6% | 1.985 |
| conflict | 134 | 5.3% | 4.711 |

The rarest class has 11.2 times fewer examples than the largest class. The
current weighted cross-entropy already compensates using inverse-frequency
weights, but it treats every example within a class equally. Pair-level results
show that minority classes remain substantially weaker:

| Gold polarity | Support | Missed aspect | Wrong polarity | Pair precision | Pair recall | Pair F1 |
|---|---:|---:|---:|---:|---:|---:|
| positive | 187 | 21 | 14 | 0.826 | 0.813 | 0.819 |
| negative | 72 | 11 | 16 | 0.692 | 0.625 | 0.657 |
| neutral | 40 | 10 | 9 | 0.477 | 0.525 | 0.500 |
| conflict | 17 | 3 | 7 | 0.500 | 0.412 | 0.452 |

Among the 271 correctly detected aspects, polarity was correct for 225, or
83.0%. The most common polarity confusions were:

| Gold | Predicted | Count |
|---|---|---:|
| negative | positive | 9 |
| positive | neutral | 6 |
| negative | neutral | 6 |
| neutral | positive | 5 |
| positive | negative | 5 |
| neutral | negative | 3 |
| positive | conflict | 3 |
| conflict | negative | 3 |

These errors support changing the polarity loss, particularly because hard
negative and mixed-sentiment examples dominate the confusion. They also expose
label-semantic difficulty that weighting alone cannot solve:

- Review 1018 contains both good and bad food and was labeled `conflict`, but
  predicted `neutral`.
- Review 1035 says the restaurant is overpriced but worth it; price was labeled
  `conflict`, but predicted `negative`.
- Review 1087 uses sarcasm to express negative food sentiment, but predicted
  `positive`.
- Very short or context-poor reviews such as review 123 (`Ciao Bella`) can have
  weak textual evidence for the assigned neutral label.

## Single-aspect versus multi-aspect reviews

| Review group | Reviews | Gold pairs | Pair micro-F1 | Exact-set accuracy |
|---|---:|---:|---:|---:|
| One gold aspect | 207 | 207 | 0.733645 | 0.710145 |
| Multiple gold aspects | 51 | 109 | 0.697436 | 0.352941 |

The multilabel architecture fixes the legacy model's inability to emit more
than one aspect, but complete recovery of multi-aspect reviews remains the
largest review-level weakness. One missed or incorrectly polarized aspect makes
the entire review fail exact-set accuracy.

## Metric interpretation

The official evaluator's standalone sentiment metric compares `(id, polarity)`
sets and does not require the polarity to be attached to the correct aspect.
Consequently, its test polarity micro-F1 of 0.808194 can conceal aspect-polarity
pairing mistakes. The stricter pair-level polarity table above should be used
for loss-function diagnosis, while overall `(id, aspect, polarity)` micro-F1
should remain the primary model-selection metric.

## Recommended loss experiment

Change only the polarity loss first and keep the current aspect checkpoint and
thresholds fixed. This isolates whether the gain comes from handling polarity
imbalance rather than from a different aspect model.

Compare the following configurations:

1. Current inverse-frequency weighted cross-entropy baseline.
2. Weighted focal loss with `gamma=1`.
3. Weighted focal loss with `gamma=2`.
4. Class-balanced focal loss with `beta=0.999` and `gamma=2`.

For class-balanced focal loss, calculate each class weight from its training
count `n_c` as `(1 - beta) / (1 - beta ** n_c)`, then normalize the weights to
have a mean of one. Do not tune weights using the test split.

Select the configuration using validation overall pair micro-F1. Use
validation pair-level macro-F1 and per-class recall as secondary diagnostics.
In particular, confirm that gains in conflict recall do not cause excessive
false conflict predictions. After selecting exactly one configuration, evaluate
it once on the untouched test split and compare it with this report.

Because 49.5% of missed pairs originate in aspect detection, a successful
polarity-loss experiment should be followed by end-to-end aspect threshold
tuning or hard-negative polarity training rather than further increasing class
weights indefinitely.
