# UWB-Style TF-IDF Ensemble

## Purpose

Test whether a SemEval-style sparse lexical model contributes complementary
information to the transformer system. The experiment reproduces the main
ideas of UWB's supplied-data-only SemEval-2014 system: multilabel aspect
detection, aspect-specific polarity classification, bag-of-words, bigrams,
and TF-IDF features.

The linear model is expected to be weaker alone. Its value depends on whether
its lexical evidence corrects transformer errors involving distinctive terms
such as *overpriced*, *rude*, or *delicious*.

## Data Protocol

- Use only the original contest/SemEval training rows.
- Do not use SemEval augmentation, external lexicons, private data, or
  historical-test labels during model or ensemble selection.
- Use five-fold nested cross-validation.
- Keep duplicate text and all rows belonging to one review ID in the same
  partition.
- Fit vectorizers and classifiers only on each fold's fitting partition.

## Linear Models

Evaluate three variants:

| Variant | Features |
|---|---|
| Existing BoW control | Binary/count word features |
| UWB TF-IDF | Word unigrams and bigrams with TF-IDF |
| Combined linear | Count features plus TF-IDF features |

Each variant uses:

- Five independent binary logistic-regression classifiers for multilabel
  aspect detection.
- One four-class polarity classifier per aspect.
- Whole-sentence features for polarity prediction.

Search the following configuration on each inner validation partition:

| Parameter | Values |
|---|---|
| `ngram_range` | `(1, 1)`, `(1, 2)` |
| `min_df` | `1`, `2` |
| `sublinear_tf` | `false`, `true` |
| `C` | `0.1`, `0.3`, `1`, `3`, `10` |
| `class_weight` | none, balanced |
| `norm` | `l2` |

Select aspect and polarity configurations independently using only the inner
validation partition.

## Inference Cache

Infer every model once per outer fold and save aligned arrays:

```text
review_ids
transformer_aspect_probabilities      [reviews, 5]
linear_aspect_probabilities           [reviews, 5]
transformer_polarity_probabilities    [reviews, 5, 4]
linear_polarity_probabilities         [reviews, 5, 4]
```

Record the fitting, selection, and heldout ID hashes. All interpolation and
threshold searches must operate on these caches without rerunning inference.

## Logit Interpolation

Convert aspect probabilities to binary logits:

```text
logit(p) = log(p / (1 - p))
```

Convert polarity probabilities to normalized log probabilities. Interpolate
the two components independently:

```text
aspect_logit =
    (1 - aspect_alpha) * transformer_logit
    + aspect_alpha * linear_logit

polarity_log_probability =
    (1 - polarity_alpha) * transformer_log_probability
    + polarity_alpha * linear_log_probability
```

Search:

```text
aspect_alpha:   0.00 to 0.50, step 0.05
polarity_alpha: 0.00 to 0.50, step 0.05
```

Weight `0.00` must exactly reproduce the transformer endpoint. Restricting the
initial search to `0.50` prevents the weaker linear model from dominating.

For each outer fold:

1. Select interpolation weights on the inner validation partition.
2. Tune all five aspect thresholds on that partition.
3. Lock the weights and thresholds.
4. Apply them unchanged to the outer heldout partition.

## Comparisons and Metrics

Report pooled out-of-fold results for:

- Transformer alone.
- Existing count-BoW alone.
- TF-IDF alone.
- Count plus TF-IDF alone.
- Transformer plus count-BoW.
- Transformer plus TF-IDF.
- Transformer plus the combined linear model.

For every system report:

- Overall pair micro-F1.
- Aspect micro-F1 and macro-F1.
- Polarity micro-F1 and macro-F1.
- Gold-aspect polarity accuracy.
- Exact-set accuracy.
- Per-class polarity F1.
- Individual fold scores and fold wins against the transformer.
- Grouped bootstrap confidence interval for the pair-F1 difference.

## Decision Rules

Promote an ensemble only if:

- Pooled pair F1 improves by at least `+0.003`.
- It wins at least three of five outer folds.
- The grouped bootstrap interval shows no meaningful downside.
- Neutral and conflict F1 do not collapse.
- Multiple folds select a nonzero linear-model weight.

Interpret outcomes as follows:

- Zero weight in every fold means the lexical model provides no useful
  complementary signal.
- Positive inner-validation weights followed by worse heldout results indicate
  interpolation overfitting.
- Improved polarity with degraded aspect performance means TF-IDF should be
  retained only for polarity.

Only an ensemble that passes grouped CV should receive one locked historical-
test evaluation. The private set remains untouched.

## Reference

The design is based on [UWB: Machine Learning Approach to Aspect-Based
Sentiment Analysis](https://aclanthology.org/S14-2145/), which used supplied
SemEval data with binary Maximum Entropy aspect classifiers and
aspect-specific polarity classifiers using BoW, bigrams, and TF-IDF.
