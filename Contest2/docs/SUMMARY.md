# ABSA Experiment Summary

This document recaps the Contest 2 experiments chronologically. Metrics from
different stages are not always directly comparable: some measure row-level
classification, validation performance, the combined 20% holdout, the untouched
10% test split, or the training set.

## 1. Initial zero-shot classifier

The original prompt-based pipeline achieved:

- overall pair micro-F1: `0.6234`;
- exact-set accuracy: `0.0522`.

The separate `verification` result of `1.0` was a pipeline smoke test, not a real
benchmark.

## 2. Early Hugging Face RoBERTa classifiers

Three conventional fine-tuning experiments were produced:

- `roberta_aspectCategory`: single-label aspect classification;
- `roberta_polarity`: single-label polarity classification;
- `roberta_multitask`: one model predicting row-level aspect and polarity.

The aspect model reached approximately `0.7350` validation accuracy after four
epochs, the polarity model peaked around `0.8044` validation accuracy, and the
multitask run recorded a best metric of `0.7492`.

All three still treated each CSV row independently. They could not reconstruct
multiple aspect-polarity pairs for repeated review text.

## 3. Custom two-model RoBERTa run

A resumable training pipeline was implemented with separate aspect and polarity
models, grouped 80/10/10 splits, mixed precision, checkpointing, MLflow, and the
official evaluator.

- Run: `artifacts/training/roberta-exp1`
- Completed epochs: 4
- Best epoch: 3
- Best combined validation pair F1: `0.6318`

This validated the training infrastructure, but the prediction architecture was
still single-label.

## 4. Long-running single-task experiments

Aspect and polarity models were then trained one at a time so only the actively
trained model occupied the GPU.

### Aspect

- Run: `roberta-aspect-exp1`
- Completed epoch: 86
- Best epoch: 10
- Best validation micro-F1: `0.7923`

### Polarity

| Run | Completed epoch | Best epoch | Best validation score |
|---|---:|---:|---:|
| `roberta-polarity-exp2` | 27 | 17 | **0.8496** |
| `roberta-polarity-exp3` | 7 | 7 | 0.8271 |
| `roberta-polarity-exp4` | 5 | 4 | 0.8233 |
| `roberta-polarity-exp5` | 60 | 20 | 0.8233 |

`roberta-polarity-exp2` supplied the best polarity checkpoint. The stored
manifests do not contain enough information to confidently attribute differences
among these attempts to particular changes.

## 5. Best-checkpoint single-label ensemble

The best aspect checkpoint from epoch 10 and polarity checkpoint from epoch 17
were combined.

### Combined 20% holdout

This combines 10% validation and 10% test, so it contains checkpoint-selection
bias.

- Pair micro-F1: `0.6765`
- Exact-set accuracy: `0.5930`

### Full dataset

This is an in-sample diagnostic, not a generalization estimate.

- Pair micro-F1: `0.8351`
- Exact-set accuracy: `0.7655`

These evaluations showed that high row-level scores did not solve multi-aspect
ABSA.

## 6. Multilabel aspect plus conditioned polarity

The task formulation was corrected by grouping all rows for each review,
predicting a multilabel aspect set, and predicting polarity separately for each
detected aspect. Per-aspect thresholds were tuned using validation data only.

| Split or system | Pair micro-F1 | Exact-set accuracy |
|---|---:|---:|
| Validation | `0.7523` | `0.6899` |
| Untouched test | **0.7223** | **0.6395** |
| Legacy single-label test baseline | `0.6725` | `0.5814` |

This improved test pair F1 by approximately five absolute points and enabled
multiple predictions per review.

## 7. Error analysis

The conditioned pipeline's untouched test predictions contained:

- 316 gold pairs;
- 225 correct pairs;
- 82 false positives;
- 91 false negatives;
- 45 pairs missed because the aspect was not detected;
- 46 pairs missed because polarity was incorrect.

Single-aspect pair F1 was `0.7336`, compared with `0.6974` for multi-aspect
reviews.

| Polarity | Pair F1 |
|---|---:|
| Positive | 0.8194 |
| Negative | 0.6569 |
| Neutral | 0.5000 |
| Conflict | 0.4516 |

Class imbalance mattered, but polarity errors accounted for only about half the
missed pairs. Aspect detection imposed a similar error ceiling.

## 8. Separate-model loss screen

The aspect checkpoint was held fixed while polarity loss was varied using seed 42.

| Loss | Best epoch | Validation pair F1 |
|---|---:|---:|
| Weighted cross-entropy | 30 | **0.7740** |
| Weighted focal, gamma 1 | 45 | 0.7616 |
| Weighted focal, gamma 2 | 40 | 0.7678 |
| Class-balanced focal, beta 0.999, gamma 2 | 30 | 0.7554 |

Inverse-frequency weighted cross-entropy was best for the separate architecture.
Increasing emphasis on hard or minority examples did not improve overall
validation generalization.

## 9. Candidate-conditioned joint ABSA screen

A shared RoBERTa encoder was given each review paired with all five candidate
aspects. An aspect-presence head identifies whether the candidate is present, and
a four-class polarity head is trained only for present aspects.

| Joint polarity loss | Best epoch | Validation pair F1 |
|---|---:|---:|
| Weighted cross-entropy | 40 | 0.7757 |
| Weighted focal, gamma 2 | 35 | **0.7836** |

Unlike the separate model, the joint model benefited from focal loss.

## 10. Three-seed confirmation

The best separate and joint configurations were confirmed using seeds 17, 42,
and 73.

### Separate weighted cross-entropy

| Seed | Best epoch | Validation F1 |
|---:|---:|---:|
| 17 | 25 | 0.7523 |
| 42 | 30 | 0.7740 |
| 73 | 30 | 0.7570 |

- Mean: `0.7611`
- Standard deviation: `0.0093`
- Ensemble validation F1: `0.7554`

### Joint focal, gamma 2

| Seed | Best epoch | Validation F1 |
|---:|---:|---:|
| 17 | 45 | 0.7664 |
| 42 | 35 | 0.7836 |
| 73 | 40 | 0.7607 |

- Mean: `0.7702`
- Standard deviation: `0.0097`
- Ensemble validation F1: **0.7850**

The joint ensemble was locked as the winner before test inference.

## 11. Final ensemble test evaluation

Both three-seed finalists were evaluated once on the untouched 10% test split.

| System | Pair F1 | Exact set | Multi-aspect exact |
|---|---:|---:|---:|
| Separate weighted-CE ensemble | **0.7448** | **0.6550** | **0.4118** |
| Joint focal ensemble | 0.7244 | 0.6279 | 0.3333 |
| Existing conditioned pipeline | 0.7223 | 0.6395 | 0.3529 |
| Legacy single-label baseline | 0.6725 | 0.5814 | 0.0000 |

Validation selected the joint model, but the separate ensemble produced the
highest observed test score. The joint ensemble improved neutral and conflict F1,
but its negative F1 fell from `0.6939` for the separate ensemble to `0.6203`.
That regression outweighed its minority-class improvements.

The separate model cannot be retroactively declared the protocol winner based on
test performance. Doing so would use the test set for model selection.

## 12. Bootstrap comparison

Paired review-level bootstrap results were:

- joint minus separate: `-0.0203`, 95% interval `[-0.0513, +0.0099]`;
- joint minus existing pipeline: `+0.0023`, 95% interval
  `[-0.0315, +0.0372]`.

Both intervals include zero, so the test split does not establish a statistically
reliable difference between these systems.

## 13. Three-seed multilabel aspect ensemble

The multilabel aspect model was trained independently with seeds 42, 43, and 44.
The three checkpoints' sigmoid probabilities were averaged, and one threshold
per aspect was tuned on validation data.

| Seed | Best epoch | Validation aspect micro-F1 |
|---:|---:|---:|
| 42 | 10 | 0.8885 |
| 43 | 35 | 0.8865 |
| 44 | 10 | 0.8781 |

The individual validation mean was `0.8844` with standard deviation `0.0055`.
The ensemble reached `0.8945` validation aspect micro-F1.

| Test system | Aspect micro-F1 | Aspect macro-F1 | Exact aspect set | Multi-aspect exact |
|---|---:|---:|---:|---:|
| Three-seed ensemble | **0.8774** | **0.8706** | **0.7946** | **0.5490** |
| Previous one-seed multilabel | 0.8700 | 0.8643 | 0.7791 | 0.5490 |
| Legacy single-label | 0.8153 | 0.7828 | 0.7093 | 0.0000 |

The ensemble-minus-legacy aspect micro-F1 difference was `+0.0621`, with a
95% paired review-level bootstrap interval of `[+0.0318, +0.0932]`. This directly
confirmed the benefit of predicting a set of aspects: the single-label model
could not exactly solve any of the 51 multi-aspect test reviews, while the new
ensemble solved 28.

## 14. New aspect ensemble with existing polarity models

The completed aspect ensemble was composed with two existing aspect-conditioned
polarity options. Candidate selection and pair-threshold tuning used validation
data only.

| Polarity candidate | Validation pair F1 | Exact-set accuracy |
|---|---:|---:|
| Previous one-seed conditioned model | 0.7571 | **0.7054** |
| Three-seed weighted-CE ensemble | **0.7659** | 0.6860 |

The three-seed weighted-CE polarity ensemble was locked before test inference.
It combines seeds 17, 42, and 73 by averaging polarity logits for every
`(review, candidate aspect)` input.

| Test system | Aspect F1 | Polarity F1 | Pair F1 | Exact set | Multi-aspect exact |
|---|---:|---:|---:|---:|---:|
| New composed system | **0.8748** | 0.8111 | **0.7544** | **0.6628** | **0.4314** |
| Previous separate ensemble | 0.8700 | **0.8163** | 0.7448 | 0.6550 | 0.4118 |
| Previous one-seed pipeline | 0.8700 | 0.8082 | 0.7223 | 0.6395 | 0.3529 |
| Previous joint ensemble | 0.8567 | 0.7948 | 0.7244 | 0.6279 | 0.3333 |
| Legacy single-label | 0.8153 | 0.8053 | 0.6725 | 0.5814 | 0.0000 |

Using gold aspects with the selected polarity ensemble produced pair F1
`0.8354`, showing the remaining ceiling imposed by aspect detection. The final
pair thresholds differ from the aspect-only thresholds because they optimize
complete `(aspect, polarity)` F1 rather than aspect F1 alone.

The new composition improved pair F1 over the previous one-seed pipeline by
`+0.0321`, with a 95% bootstrap interval of `[+0.0017, +0.0636]`. Its improvement
over the legacy baseline was `+0.0816`, with interval `[+0.0405, +0.1247]`.

These two follow-up experiments reused a test partition already evaluated by
earlier work. It remains held out from training and from within-run selection,
but it should no longer be described as globally untouched.

## 15. Neutral and conflict polarity reweighting

The next experiment retained the aspect-conditioned polarity architecture and
existing inverse-frequency class weights, then applied additional multipliers
only to neutral and conflict. Five configurations were screened, with a
validation pair-F1 tolerance of `0.005`.

| Candidate | Neutral multiplier | Conflict multiplier | Pair F1 | Minority F1 | Eligible |
|---|---:|---:|---:|---:|---|
| `weighted-ce-n1.25-c1.5` | 1.25 | 1.5 | 0.7558 | 0.5740 | No |
| `weighted-ce-n1.5-c2` | 1.5 | 2.0 | **0.7869** | 0.5797 | Yes |
| `weighted-ce-n2-c2` | 2.0 | 2.0 | 0.7781 | **0.5957** | No |
| `weighted-ce-n1.5-c3` | 1.5 | 3.0 | 0.7714 | 0.5680 | No |
| `weighted-ce-n2-c3` | 2.0 | 3.0 | 0.7652 | 0.5465 | No |

Here, minority F1 is the mean of neutral and conflict F1 when the gold aspect
is supplied. The only eligible screening candidate, neutral `1.5x` plus
conflict `2.0x`, was then trained across three seeds and locked on validation.

| Validation system | Pair F1 | Minority F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|
| Original weighted-CE ensemble | 0.7659 | 0.5847 | 0.7027 | **0.4667** |
| Reweighted ensemble | **0.7776** | **0.5954** | **0.7532** | 0.4375 |

The validation-locked winner was evaluated without changing its weights or
thresholds:

| Test system | Pair F1 | Polarity F1 | Exact set | Minority F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
| Original weighted-CE ensemble | **0.7544** | **0.8111** | 0.6628 | **0.5700** | 0.6400 | **0.5000** |
| Reweighted ensemble | 0.7357 | 0.7993 | 0.6628 | 0.5277 | **0.7105** | 0.3448 |

Extra weighting produced a substantial neutral improvement on test, but it did
not solve conflict and reduced both overall polarity and pair F1. Averaging the
two minority-class scores also hid their opposite movement during validation:
neutral improved while conflict regressed. Future polarity selection should
therefore protect neutral and conflict separately rather than optimize only
their mean, ideally using grouped cross-validation or repeated validation
splits because these rare-class estimates are unstable.

## Overall conclusion

The meaningful progression in untouched or increasingly rigorous pair-level
evaluation was:

`0.6234` zero-shot -> `0.6725` single-label baseline -> `0.7223` corrected
conditioned pipeline -> `0.7448` previous separate ensemble -> `0.7544` new
aspect ensemble plus existing polarity ensemble.

The largest gain came from correcting the task formulation, not changing the loss.
Multilabel aspect prediction plus aspect-conditioned polarity was essential. Loss
weighting and focal loss produced smaller, architecture-dependent effects.

The strongest observed complete system is now the three-seed multilabel aspect
ensemble composed with the existing three-seed weighted-CE polarity ensemble.
The polarity candidate was selected on validation in the follow-up experiment.
The earlier joint-versus-separate protocol still selected the joint focal model;
the later result does not retroactively change that earlier locked decision.

Additional neutral/conflict reweighting was also selected on validation, but it
did not improve the strongest observed test result. It traded a large neutral
gain for a larger conflict loss, reducing pair F1 from `0.7544` to `0.7357`.
This makes polarity calibration or explicit positive/negative evidence modeling
reasonable future experiments, not established improvements.

A future experiment should use grouped cross-validation or repeated validation
splits to make architecture selection more stable. A genuinely new final test
partition would be required for another unbiased model-selection claim.

## Related reports

- `docs/REPORT.md`
- `artifacts/ensemble/heldout-20-percent/report.md`
- `artifacts/ensemble/full-dataset/report.md`
- `artifacts/experiments/multilabel-conditioned-v1/error_analysis.md`
- `artifacts/experiments/absa-imbalance-joint-v1/report.md`
- `artifacts/experiments/multilabel-aspect-v1/report.md`
- `artifacts/experiments/multilabel-aspect-old-polarity-v1/report.md`
- `artifacts/experiments/polarity-reweighting-v1/report.md`
