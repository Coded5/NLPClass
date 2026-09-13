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

## 16. Validation-calibrated polarity evidence model

A final polarity experiment trained a shared RoBERTa encoder with two outputs:
the usual four-class polarity head and a two-logit evidence head. The evidence
targets represented positive and negative evidence independently:

- positive: `10`;
- negative: `01`;
- neutral: `00`;
- conflict: `11`.

Evidence-loss weights `0.25`, `0.5`, and `1.0` were screened using seed 42.
The selected weight, `0.25`, was confirmed with seeds 17, 42, and 73. The
existing aspect ensemble and pair thresholds `[0.55, 0.8, 0.9, 0.7, 0.45]`
were frozen. Temperature, evidence blending, and neutral/conflict logit biases
were tuned on validation. A candidate had to preserve both minority-class F1
scores and remain within `0.005` pair F1 in at least 75% of 10,000 paired
review-ID bootstrap samples.

| Validation system | Pair F1 | Polarity F1 | Exact set | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|
| Original weighted-CE ensemble | 0.7659 | 0.8309 | 0.6860 | 0.7027 | 0.4667 |
| Calibrated original | **0.7783** | 0.8452 | **0.7054** | 0.7470 | **0.5000** |
| Evidence model, four-class head | 0.7659 | 0.8509 | 0.6977 | 0.7368 | 0.4667 |
| Calibrated evidence model | 0.7721 | **0.8566** | **0.7054** | **0.7532** | 0.4828 |

The validation protocol locked the calibrated original ensemble. Its selected
parameters were temperature `1.5`, neutral bias `+1.5`, conflict bias `-0.75`,
and evidence blend `0`. The calibrated evidence model also selected evidence
blend `0`. Thus, validation did not support using the auxiliary evidence logits
in the combined decoder.

| Test system | Pair F1 | Polarity F1 | Exact set | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|
| Original weighted-CE ensemble | **0.7544** | 0.8111 | **0.6628** | 0.6400 | **0.5000** |
| Calibrated original | 0.7448 | 0.8081 | **0.6628** | **0.6588** | 0.4348 |
| Evidence model, four-class head | 0.7512 | 0.8090 | 0.6589 | 0.6486 | 0.4286 |
| Calibrated evidence model | **0.7544** | **0.8127** | **0.6628** | 0.6486 | 0.4444 |

Calibration improved every tracked validation metric but did not fully
generalize. The locked calibrated-original system gained neutral F1 on test but
lost `0.0095` pair F1 and `0.0652` conflict F1 relative to the original. The
calibrated evidence model matched the original test pair F1 and slightly
improved polarity F1, but still reduced conflict F1.

The auxiliary evidence head was also evaluated alone by converting its two
logits directly into the four evidence combinations, without the four-class
head or calibration:

| Split | Pair F1 | Polarity F1 | Exact set | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|
| Validation | 0.7659 | 0.8488 | 0.6977 | 0.7013 | 0.5000 |
| Test | 0.7322 | 0.7896 | 0.6473 | 0.6173 | 0.2963 |

Evidence-only decoding appeared competitive on validation but generalized
poorly, especially for conflict. It is not a replacement for the original
polarity ensemble.

## 17. DeBERTa-v3 backbone and cross-backbone ensembles

`microsoft/deberta-v3-base` was substituted for RoBERTa while retaining the
multilabel aspect architecture, aspect-conditioned four-class polarity model,
inverse-frequency weighted cross-entropy, grouped splits, and validation-only
threshold selection.

A 10-minute feasibility benchmark established that DeBERTa-v3-base fit on the
RTX 3070 without gradient checkpointing:

- physical batch size: `8`, with two-step gradient accumulation;
- throughput: approximately `138-140` examples per second;
- peak reserved GPU memory: `4.29 GiB`;
- projected 50-epoch aspect-plus-polarity seed: `27.8` minutes.

The initial seed-42 pilot finished training in `23.6` minutes and reached
`0.7668` validation pair F1. Its gold-aspect polarity macro-F1 was `0.7726`,
including neutral F1 `0.7200` and conflict F1 `0.5789`. This justified the
three-seed follow-up.

The follow-up used aspect seeds 42, 43, and 44 and polarity seeds 17, 42, and
73. Three systems were tuned independently on validation. For the mixed
polarity system, the three-seed RoBERTa and DeBERTa polarity logits received
equal backbone-level weight.

| Validation system | Aspect F1 | Polarity F1 | Pair F1 | Exact set | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
| DeBERTa aspect + DeBERTa polarity | 0.8913 | **0.8550** | 0.7748 | 0.7093 | 0.7250 | **0.5806** |
| RoBERTa aspect + DeBERTa polarity | **0.8924** | 0.8483 | 0.7769 | 0.7093 | 0.6905 | **0.5806** |
| RoBERTa aspect + mixed polarity | 0.8903 | 0.8545 | **0.7790** | **0.7287** | **0.7368** | 0.4667 |

The RoBERTa-aspect plus mixed-polarity system was locked by validation pair F1
before test inference. All three locked systems were then evaluated on the
previously reused 10% test partition:

| Test system | Aspect F1 | Polarity F1 | Pair F1 | Exact set | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
| DeBERTa aspect + DeBERTa polarity | 0.8720 | **0.8235** | 0.7488 | 0.6628 | 0.5946 | **0.5455** |
| RoBERTa aspect + DeBERTa polarity | **0.8736** | 0.8229 | 0.7616 | **0.6860** | **0.6111** | **0.5455** |
| RoBERTa aspect + mixed polarity | 0.8734 | 0.8231 | **0.7630** | **0.6860** | 0.6087 | 0.5185 |
| Previous strongest composed system | 0.8748 | 0.8111 | 0.7544 | 0.6628 | 0.5556 | **0.5600** |

The validation-selected mixed system improved observed test pair F1 by
`+0.0086`, polarity F1 by `+0.0120`, neutral F1 by `+0.0531`, and exact-set
accuracy by `+0.0233` relative to the previous strongest composition. Conflict
F1 decreased by `0.0415`, from `0.5600` to `0.5185`. The new system's pair-F1
gain came primarily from higher precision and fewer false-positive pairs, not
from improved conflict recognition. DeBERTa-only did not improve pair F1
because its weaker aspect detection offset its polarity gains. The stronger
result therefore came from retaining the RoBERTa aspect ensemble and adding
complementary DeBERTa polarity evidence.

### Corrected polarity-only comparison

A subsequent polarity audit compared the three polarity alternatives on the
same 316 test annotations while supplying the gold aspect to every model. This
isolates four-class, aspect-specific polarity classification from aspect
detection and from the official `(id, polarity)` set metric:

| Polarity model | Accuracy | Macro-F1 | Positive F1 | Negative F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
| RoBERTa, 3 seeds | 0.8354 | 0.7088 | 0.9167 | 0.7785 | 0.6400 | 0.5000 |
| DeBERTa-v3, 3 seeds | 0.8418 | 0.7187 | 0.9186 | **0.8138** | **0.6575** | 0.4848 |
| Equal-logit mixture | **0.8449** | **0.7215** | **0.9267** | 0.7919 | 0.6486 | **0.5185** |

The corrected result shows that DeBERTa polarity is marginally better than
RoBERTa polarity overall, while the equal-logit mixture is best. DeBERTa gets
two more rows correct than RoBERTa (266 versus 264); the mixture gets three more
(267). DeBERTa has the best negative and neutral F1, whereas the mixture has the
best positive and conflict F1.

An earlier ad-hoc audit incorrectly reported DeBERTa accuracy as `0.6044`
because it did not reliably align each review's five aspect-candidate logits
with the annotated gold aspect. That result is invalid and must not be used.
The corrected mapping and raw metrics are produced by
`scripts/compare_polarity_backbones.py` and stored in
`artifacts/experiments/deberta-v3-three-systems-v1/polarity_backbone_comparison.json`.

The complete three-system run took `61.4` minutes. Its test partition had
already been evaluated by earlier experiments, so these remain strongest
observed results rather than a fresh unbiased benchmark.

## 18. Targeted miscellaneous oversampling

Five-fold training-only CV compared the reference aspect model with sampling
controls and targeted 2x, 3x, and 4x sampling of reviews where miscellaneous
co-occurs with another aspect. Targeted 3x won CV: aspect micro-F1 increased
from `0.8788` to `0.8857`, miscellaneous F1 from `0.8370` to `0.8527`, and the
target subgroup recall from `0.4000` to `0.4421`.

The gain did not survive three-seed confirmation. Relative to the reference,
targeted 3x reduced validation pair F1 from `0.7790` to `0.7707` and historical
test pair F1 from `0.7630` to `0.7448`, although test subgroup recall improved
from `0.2857` to `0.4286`. The reference therefore remained locked. Increasing
the frequency of the rare pattern improved its recall but did not add new
language patterns and harmed the complete system.

## 19. Out-of-fold training-label audit

A deterministic review sampled 100 unique training reviews from persistent
errors in the saved aspect and polarity CV predictions. It deliberately
included 20 multi-aspect miscellaneous cases, 20 conflict cases, 20 neutral
cases, and 40 comparison cases. The historical test split was not read.

Revision 2 classifies 25 cases as clear model errors, 63 as ambiguous, eight
as suspected annotation errors, and four as lacking enough context. These are
provisional assistant judgments. The original 37/30/32/1 breakdown and its
conclusion that label uncertainty explains the plateau were overstated.
Previous categories are preserved in the review worksheet.

The revised rubric distinguishes an evaluated aspect from an incidental noun
mention, ties polarity to the correct target, and does not require every
anecdotes/miscellaneous label to express an overall opinion. Missing context
does not automatically establish neutral sentiment. No annotation manual was
located in the repository search; these criteria are not claimed as official
dataset rules. Ambiguity in this selected sample does not establish dataset-wide
annotation noise or the cause of the plateau.

Clear failures still include missing secondary aspects and opposing evaluations
of the same aspect. Because the judgments were not independently adjudicated,
the supplied labels remain authoritative and proposed alternatives must not be
used for training. The subsequent learning-curve experiment therefore retained
the original labels unchanged.

## 20. Grouped learning-curve diagnostic

The learning-curve experiment measured whether the current architecture still
benefits from additional labeled data. It used only the original training
partition; the historical validation and test partitions were not read. Five
fixed outer folds were grouped by review ID. Within each fold, a fixed grouped
selection split controlled early stopping and threshold tuning, and the
remaining IDs formed strictly nested, approximately stratified 25%, 50%, 75%,
and 100% training subsets. Every selected ID retained all of its annotation
rows.

Each fold and fraction trained one RoBERTa multilabel aspect model and one
weighted-CE polarity model for each of RoBERTa and DeBERTa-v3. The two polarity
models were combined by equal-logit averaging. This produced 60 model fits in
total. Results below are pooled out-of-fold metrics; the displayed variation is
the mean and standard deviation across the five folds.

| Training data | Aspect micro-F1 | Gold-aspect polarity macro-F1 | Pair micro-F1 | Pooled pair F1 |
|---|---:|---:|---:|---:|
| 25% | 0.8607 +/- 0.0091 | 0.6053 +/- 0.0280 | 0.6915 +/- 0.0153 | 0.6916 |
| 50% | 0.8760 +/- 0.0101 | 0.6810 +/- 0.0143 | 0.7312 +/- 0.0185 | 0.7311 |
| 75% | 0.8778 +/- 0.0066 | 0.6858 +/- 0.0605 | 0.7248 +/- 0.0392 | 0.7248 |
| 100% | **0.8818 +/- 0.0178** | **0.6935 +/- 0.0357** | **0.7499 +/- 0.0231** | **0.7499** |

The 50% to 75% pair score decreased by `0.0063`; its paired-bootstrap 95%
interval, `[-0.0196, +0.0068]`, includes zero and indicates fold-to-fold noise.
In contrast, the primary 75% to 100% pair-F1 change was `+0.0250`, with a
10,000-sample paired review bootstrap interval of `[+0.0131, +0.0373]`. Under
the pre-specified rule requiring at least `+0.01` improvement with the interval
excluding zero, the complete pipeline is classified as **data-limited**.

The component-level evidence is less decisive. Aspect micro-F1 improved by
only `+0.0038` from 75% to 100%, with interval `[-0.0038, +0.0113]`. Polarity
macro-F1 improved by `+0.0118`, with interval `[-0.0119, +0.0351]`. Both are
classified as inconclusive rather than individually data-limited or plateaued.
The significant pair gain can arise from their combined predictions and
validation-tuned decoding even when neither component estimate is sufficiently
precise alone.

At 100%, gold-aspect polarity accuracy was `0.8321`; positive, negative,
neutral, and conflict F1 were `0.9170`, `0.7935`, `0.6545`, and `0.4113`.
Conflict remained the weakest class but rose from `0.2594` at 25%. The 100%
composed system reached exact-set accuracy `0.6784`, with `0.7223` on
single-aspect and `0.4897` on multi-aspect reviews.

The pooled 100% pair F1 of `0.7499` is not a new model-selection result and does
not replace the historical-test winner at `0.7630`: the learning curve uses
one seed per fold and grouped out-of-fold training data, whereas the current
winner is a three-seed system evaluated on the historical test partition. The
learning curve supports collecting more high-quality labels, especially for
minority polarity cases, but does not itself produce a deployable replacement.

## 21. Multi-seed synthetic-conflict isolation

The synthetic-conflict follow-up tested whether joining same-aspect positive
and negative clauses teaches more than simply repeating real conflict
annotations. It used five grouped outer folds and three training seeds. In
every fold, both RoBERTa polarity-only conditions added exactly 40 conflict
rows with identical aspect counts. The repetition condition duplicated frozen
real conflict rows; the synthetic condition used the previously reviewed
clause combinations. Class weights were calculated from the natural fitting
rows, and neither the historical validation nor test partition was used. The
design produced 30 model fits.

| Three-seed OOF ensemble | Accuracy | Macro-F1 | Neutral F1 | Conflict F1 | Conflict FP rate |
|---|---:|---:|---:|---:|---:|
| Repeated real conflict rows | 0.8147 | 0.6827 | 0.6489 | 0.4026 | 0.0485 |
| Synthetic conflict rows | **0.8202** | **0.6941** | 0.6482 | **0.4328** | **0.0439** |

The synthetic condition improved ensemble conflict F1 by `+0.0302`, but the
paired review-bootstrap 95% interval was `[-0.0325, +0.0931]`. Its macro-F1
gain was `+0.0114`, with interval `[-0.0093, +0.0321]`. Neutral F1 was
effectively unchanged (`-0.0007`), and the conflict false-positive rate
decreased by `0.0046`.

Training-seed behavior was inconsistent. Synthetic-minus-repetition conflict-F1
deltas were `-0.0055`, `-0.0241`, and `+0.0377` for seeds 42, 1337, and 2024,
so only one of three seeds improved. The experiment therefore failed its
pre-specified promotion rule, which required a conflict gain of at least
`+0.03`, a confidence interval above zero, at least two winning seeds, and
bounded macro-F1, neutral-F1, and false-positive regressions.

The point estimates are compatible with a possible benefit, but this experiment
does not establish that synthetic contrastive wording is better than matched
repetition of real conflict rows. It is a polarity-only training diagnostic and
does not replace or directly evaluate the complete-system winner.

## 22. SemEval polarity augmentation

Deduplicated SemEval-2014 restaurant annotations were added only to the original
fitting partition; the existing validation and historical test partitions were
kept byte-identical. The three-seed RoBERTa aspect ensemble remained frozen.
The first condition added all 1,021 retained SemEval annotation rows: 654
positive, 221 negative, 94 neutral, and 52 conflict. DeBERTa-v3 polarity used
weighted cross-entropy and seeds 17, 42, and 73.

| All-class SemEval system | Validation pair F1 | Historical-test pair F1 | Gold-aspect polarity accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Original DeBERTa polarity | 0.7769 | 0.7616 | 0.8418 | 0.7187 |
| SemEval-augmented DeBERTa polarity | 0.7863 | **0.7744** | **0.8544** | 0.7196 |
| Previous mixed-polarity winner | 0.7790 | 0.7630 | 0.8449 | **0.7215** |
| SemEval-augmented mixed polarity | **0.7894** | 0.7680 | 0.8449 | 0.7067 |

The augmented DeBERTa-only composition produced the highest observed
historical-test pair F1, `0.7744`, a `+0.0114` point gain over the previous
winner. However, validation selected the augmented mixed system, whose test
pair F1 was `0.7680`. The augmented-DeBERTa minus original-DeBERTa bootstrap
interval was `[-0.0098, +0.0365]`, and the augmented-mixed minus previous-winner
interval was `[-0.0150, +0.0264]`; neither establishes a reliable gain. Because
this historical test partition has been inspected repeatedly, `0.7744` is an
observed result rather than a new unbiased winner claim.

## 23. Minority-only SemEval augmentation

A controlled follow-up removed every SemEval positive and negative annotation,
retaining only 94 neutral and 52 conflict rows. The resulting fitting split had
2,671 rows: 1,502 positive, 571 negative, 412 neutral, and 186 conflict. All
three DeBERTa seeds were trained fresh; aspect checkpoints and original
RoBERTa polarity checkpoints were frozen.

| Minority-only system | Validation pair F1 | Historical-test pair F1 | Gold-aspect polarity accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Original DeBERTa polarity | 0.7769 | 0.7616 | 0.8418 | 0.7187 |
| Minority-only DeBERTa polarity | 0.7551 | 0.7616 | 0.8418 | **0.7287** |
| Previous mixed-polarity winner | 0.7790 | 0.7630 | 0.8449 | 0.7215 |
| Minority-only mixed polarity | **0.7832** | **0.7648** | **0.8481** | 0.7249 |

The intervention shifted performance toward the intended classes. Relative to
all-class SemEval DeBERTa, neutral F1 rose from `0.6486` to `0.7073` and
conflict F1 from `0.4828` to `0.5000`, while positive F1 fell from `0.9443` to
`0.9297` and negative F1 from `0.8026` to `0.7778`. This tradeoff raised
macro-F1 but erased the overall pair-F1 gain. The minority-only DeBERTa model
matched the original at `0.7616`; its mixed composition reached `0.7648`, below
the all-class result. Both bootstrap intervals included zero. Minority-only
augmentation is therefore useful evidence about class tradeoffs, not a model to
promote.

## 24. Full-combined joint DeBERTa and decoder tuning

A seed-42 joint DeBERTa-v3-base pilot trained one shared encoder with aspect and
polarity heads on the original fitting partition plus all four classes from the
deduplicated SemEval augmentation. It used weighted focal polarity loss with
gamma 2, selected epoch 20 on validation, and finished in 54.4 minutes. The
historical test split was not accessed during training or checkpoint selection.

| Decoder evaluation | Pair F1 | Aspect F1 | Polarity F1 | Exact-set accuracy |
|---|---:|---:|---:|---:|
| Original, validation | 0.7852 | 0.8934 | 0.8613 | 0.7132 |
| Tuned, validation | **0.8019** | **0.8947** | **0.8721** | **0.7287** |
| Original, historical test | **0.7520** | **0.8689** | **0.8214** | **0.6705** |
| Tuned, historical test | 0.7437 | 0.8608 | 0.8177 | 0.6589 |

The original joint decoder was below the matched separate DeBERTa seed-42
validation result of `0.7950`, although it slightly exceeded the historical
original-data joint RoBERTa seed-42 validation result of `0.7836`. On validation,
its official polarity F1 values were `0.9274` positive, `0.8387` negative,
`0.7342` neutral, and `0.6286` conflict.

A low-cost decoder search then adjusted five aspect thresholds, polarity-logit
biases, an aspect-count cap, a confidence gate, and a conflict margin. The
selected validation decoder lowered the negative and conflict logits, raised
neutral, and limited predictions to three aspects. Its `+0.0167` validation
pair-F1 gain did not generalize: historical-test pair F1 fell by `0.0083`, and
all principal test metrics decreased. Conflict pair F1 was unchanged at
`0.5806`; positive, negative, and neutral pair F1 all declined slightly.

The tuned decoder is therefore rejected and the original decoder is retained
for this joint checkpoint. Even the retained test score of `0.7520` is below
the strongest separate SemEval-augmented systems. This result does not support
advancing the full-combined joint architecture into expensive grouped CV. The
next low-cost roadmap experiment is validation-only interpolation of the saved
all-class and minority-only DeBERTa polarity logits.

## 25. Existing DeBERTa logit interpolation

The roadmap's no-training interpolation pilot combined saved all-class and
minority-only three-seed DeBERTa polarity logits while freezing the existing
RoBERTa aspect ensemble. It used only the established validation partition and
did not access the historical test split. Alpha denotes minority-model weight.

| Alpha | Fixed-threshold pair F1 | Retuned pair F1 | Gold-aspect macro-F1 | Gold-aspect conflict F1 |
|---:|---:|---:|---:|---:|
| 0.00 | **0.7863** | **0.7863** | 0.7645 | 0.5455 |
| 0.25 | **0.7863** | **0.7863** | **0.7771** | **0.5882** |
| 0.50 | 0.7738 | 0.7738 | 0.7625 | 0.5714 |
| 0.75 | 0.7613 | 0.7613 | 0.7387 | 0.5000 |
| 1.00 | 0.7551 | 0.7551 | 0.7274 | 0.4500 |

Threshold retuning selected the same thresholds for every alpha and did not
change any score. A 25% minority blend improved gold-aspect macro-F1 and
conflict F1, and raised pair-level conflict F1 from `0.5333` to `0.5806`, but
it did not improve the primary pair-F1 metric. Under the predefined exact-tie
rule favoring less minority weight, alpha zero remains selected.

The minority-only ensemble is therefore retained as a class-balance diagnostic,
not included in the initial grouped-CV recipe. The roadmap proceeds to matched
seed-42 ModernBERT and ELECTRA polarity pilots; only the stronger useful
DeBERTa blend should enter the first grouped CV.

## 26. ModernBERT polarity pilot

A seed-42 `answerdotai/ModernBERT-base` polarity model was trained with the
same all-class SemEval fitting data, weighted cross-entropy, frozen RoBERTa
aspect ensemble, and validation partition as the DeBERTa comparison. A real
forward/backward smoke check confirmed finite loss, FP32 trainable parameters,
an optimizer update, four logits, and compatible text/aspect pair encoding.
The selected checkpoint was epoch 20, and the complete run took 15.1 minutes.
The historical test split was not accessed.

| ModernBERT weight | Matched seed-42 pair F1 | Three-seed DeBERTa blend pair F1 |
|---:|---:|---:|
| 0.00 | **0.7950** | **0.7863** |
| 0.25 | 0.7919 | 0.7800 |
| 0.50 | 0.7888 | 0.7843 |
| 0.75 | 0.7516 | 0.7488 |
| 1.00 | 0.7395 | 0.7395 |

ModernBERT was substantially weaker alone, and no nonzero interpolation weight
improved either DeBERTa baseline. Its standalone gold-aspect polarity macro-F1
was `0.7001`; neutral F1 was `0.6667` and conflict F1 was `0.4516`. Although
the 50% matched-seed blend raised conflict F1 relative to standalone, it still
reduced complete pair F1 from `0.7950` to `0.7888`.

The predefined validation selection therefore chose alpha zero in both the
matched and practical comparisons. ModernBERT is rejected as an initial CV
component. The next architecture screen is the matched seed-42 ELECTRA polarity
pilot; ELECTRA must show standalone or ensemble value before entering CV.

## 27. ELECTRA polarity pilot

A matched seed-42 `google/electra-base-discriminator` polarity pilot used the
same all-class SemEval fitting data, weighted cross-entropy, frozen RoBERTa
aspect probabilities, validation partition, and interpolation grid as the
ModernBERT screen. Its smoke check verified finite loss, an optimizer update,
FP32 parameters, four output logits, and supported token-type inputs. The best
standalone checkpoint was epoch 40; the complete run took 15.2 minutes. The
historical test split was not accessed.

| ELECTRA weight | Matched seed-42 pair F1 | Three-seed DeBERTa blend pair F1 |
|---:|---:|---:|
| 0.00 | **0.7950** | 0.7863 |
| 0.25 | 0.7919 | **0.7919** |
| 0.50 | 0.7795 | 0.7857 |
| 0.75 | 0.7869 | 0.7807 |
| 1.00 | 0.7678 | 0.7678 |

ELECTRA was stronger than ModernBERT alone (`0.7678` versus `0.7395`) and, more
importantly, supplied complementary errors to the practical three-seed DeBERTa
ensemble. A 25% ELECTRA blend improved validation pair F1 by `0.0057`, from
`0.7863` to `0.7919`. Gold-aspect macro-F1 increased from `0.7645` to `0.7884`,
neutral F1 from `0.7089` to `0.7179`, and conflict F1 from `0.5455` to `0.6250`.

This clears the pilot's minimum pair-gain signal without a minority-class
regression. ELECTRA is therefore the architecture-diverse challenger for the
initial grouped CV. The CV comparison should train fold-specific RoBERTa aspect,
all-class DeBERTa polarity, and all-class ELECTRA polarity models, choosing the
blend weight and aspect thresholds only on each fold's inner selection data.

## 28. ELECTRA-DeBERTa grouped cross-validation

The selected ELECTRA recipe was tested with five grouped outer folds over the
original fitting partition. Identical normalized texts remained together; each
outer training partition had a grouped inner selection split. Fold-specific
RoBERTa aspect, all-class DeBERTa polarity, and all-class ELECTRA polarity
models were trained with seed 42. SemEval rows were added only to fitting data,
and the historical validation and test partitions were not accessed.

| Pooled OOF system | Pair F1 | Aspect F1 | Polarity F1 | Exact-set accuracy |
|---|---:|---:|---:|---:|
| DeBERTa baseline | 0.7387 | **0.8828** | 0.8117 | 0.6625 |
| Inner-selected DeBERTa/ELECTRA blend | **0.7577** | 0.8826 | **0.8354** | **0.6794** |

The blend improved pair F1 by `0.0190` and won all five outer folds. Selected
ELECTRA weights were `0.50`, `0.50`, `0.50`, `0.25`, and `0.75`. Pair-level
neutral F1 rose from `0.5493` to `0.5756`, conflict F1 from `0.3596` to
`0.4621`, negative F1 from `0.6847` to `0.7065`, and positive F1 from `0.8320`
to `0.8399`. The paired review-bootstrap 95% interval for pair-F1 improvement
was `[+0.0100, +0.0279]`.

This passes every predefined advancement criterion: gain at least `0.005`, wins
in at least four folds, and no neutral/conflict regression. ELECTRA therefore
advances to three-seed confirmation with the same frozen folds and candidate
grid. The pooled `0.7577` is out-of-fold development evidence, not directly
comparable to historical-test rankings.

## 29. ELECTRA-DeBERTa three-seed confirmation

The CV baseline and ELECTRA recipe were repeated with training seeds 17 and 73
on the same five folds, then combined with seed 42 into fold-wise three-seed
ensembles. Checkpoints, alpha, and aspect thresholds were selected using each
fold's inner partition. The historical validation and test partitions were not
accessed.

| Three-seed OOF system | Pair F1 | Aspect F1 | Polarity F1 | Exact-set accuracy |
|---|---:|---:|---:|---:|
| DeBERTa baseline | **0.7667** | **0.8904** | 0.8356 | **0.6934** |
| DeBERTa/ELECTRA blend | 0.7637 | 0.8902 | **0.8356** | 0.6920 |

The blend regressed pair F1 by `0.0030` and won only one of five ensemble folds.
Its paired bootstrap interval was `[-0.0086, +0.0026]`. Conflict pair F1 rose
slightly from `0.4469` to `0.4532`, while neutral fell from `0.5960` to `0.5906`.
Individual-seed blend deltas were `+0.0190`, `+0.0054`, and `-0.0011` for seeds
42, 17, and 73 respectively.

The confirmation gate therefore failed. ELECTRA's apparent one-seed advantage
was not preserved after seed ensembling, so it is not promoted into the final
recipe. The fold-matched three-seed DeBERTa baseline remains preferred. The
roadmap proceeds to controlled external-data sampling rather than further
ELECTRA expansion.

## 30. External minority sampling status

The controlled `k = 1, 2, 4` external minority-sampling CV was launched after
the ELECTRA confirmation, then explicitly aborted before any fold completed.
It has no interpretable result and is excluded from comparisons. Partial
artifacts are retained only for provenance. At the user's direction, the
roadmap skips this experiment and proceeds to temperature-scaled interpolation.

## 31. Temperature-scaled interpolation CV

The first loader-heavy attempt was intentionally interrupted with exit status
`130` after two folds so checkpoint loading could be optimized. The resumed
implementation preserved those completed folds, finished all five folds with
exit status `0`, and did not access the historical test partition.

Temperature scaling did not improve the three-seed DeBERTa/ELECTRA
interpolation. The raw blend scored `0.7637` pooled pair F1 and the scaled blend
scored `0.7632`, a delta of `-0.0005`; scaling won zero of five folds. The paired
review bootstrap interval was `[-0.0025, +0.0014]`. Gold-aspect conflict F1 rose
slightly from `0.5200` to `0.5217`, but neutral F1 fell from `0.6483` to `0.6447`
and polarity micro-F1 fell from `0.8356` to `0.8344`.

This fails the shared advancement gate and confirms that unequal logit scale was
not the reason the ELECTRA blend failed three-seed confirmation. Keep the raw
DeBERTa baseline and proceed to the clause-level aspect-conditioned evidence
pilot. Full results are in
`artifacts/experiments/temperature-scaled-interpolation-cv-v1/report.md`.

## 32. Clause-level polarity evidence pilot

The validation-only seed-42 pilot compared the matched all-class standard
DeBERTa polarity model with whole-sentence and clause-aggregated two-evidence
heads. The historical test partition was not accessed. A first attempt failed
only during checkpoint reload because a compact FP16 encoder was paired with an
FP32 head; the corrected run reused the completed checkpoint and finished with
exit status `0`.

| System | Pair F1 | Gold-aspect accuracy | Macro F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|---:|
| Standard four-class DeBERTa | **0.7950** | **0.8794** | **0.7939** | 0.7317 | **0.6250** |
| Whole-sentence evidence | 0.7747 | 0.8667 | 0.7387 | **0.7595** | 0.4000 |
| Clause-level evidence | 0.7702 | 0.8508 | 0.7392 | 0.7381 | 0.4444 |

Clause aggregation regressed pair F1 by `0.0248` and conflict F1 by `0.1806`
against the standard model. It also failed to outperform the whole-sentence
evidence control. This rejects the proposed weakly supervised evidence
architecture at the pilot stage; do not spend five-fold CV or additional seeds
on it. The roadmap proceeds to the aspect-backbone comparison.

## 33. Aspect-backbone probability blending pilot

The validation-only seed-42 aspect pilot froze the same three-seed all-class
SemEval DeBERTa polarity predictions for every comparison. Existing RoBERTa and
DeBERTa aspect checkpoints were reused. One missing ModernBERT aspect model was
trained before the user clarified a preference for inference-only blending; no
historical test data was accessed.

| Aspect system | Pair F1 | Aspect micro-F1 | Aspect exact set | Overall exact set |
|---|---:|---:|---:|---:|
| RoBERTa | 0.7764 | 0.8882 | 0.8140 | 0.7171 |
| DeBERTa | 0.7700 | 0.8795 | 0.7907 | 0.6899 |
| ModernBERT | 0.7578 | 0.8634 | 0.7713 | 0.6860 |
| 50% RoBERTa + 50% DeBERTa | **0.7923** | **0.9042** | **0.8566** | **0.7481** |
| 75% RoBERTa + 25% ModernBERT | 0.7829 | 0.8843 | 0.8217 | 0.7326 |

The useful gain came from inference-only blending of the two already available
RoBERTa and DeBERTa aspect models, not from ModernBERT training. The 50/50 blend
improved pair F1 by `0.0159` and aspect micro-F1 by `0.0160` over RoBERTa alone.
It was therefore evaluated once on the historical test using the validation-
selected weight and thresholds unchanged. Pair F1 fell to `0.7660`, aspect
micro-F1 to `0.8674`, polarity micro-F1 to `0.8296`, and overall exact-set
accuracy to `0.6783`. The matched all-class SemEval DeBERTa composition remains
better at `0.7744` pair F1 and `0.8736` aspect micro-F1. The aspect blend is
rejected rather than advanced to grouped CV; its validation gain did not
generalize.

## 34. Roadmap experiments 11 and 12 status

Disagreement-driven data acquisition (Experiment 11) is skipped for now at the
user's direction because it requires a permitted new review pool and independent
human annotation. No synthetic or assistant-generated labels substitute for
that requirement.

Experiment 12 was stopped at the user's request on 2026-09-13 before completion
to redirect compute toward overall F1. Both launcher and training process were
verified exited; partial artifacts are retained and cannot establish a result.
The interrupted setup used grouped-CV ensemble distillation. It reused the
confirmed fold-specific three-seed DeBERTa teacher checkpoints and trains one
seed-42 DeBERTa student for each distillation weight `0.25`, `0.50`, and `0.75`
inside each fold. The hard-label seed-42 fold checkpoints are reused as controls.
Only fitting-input teacher logits supervise students; inner selections choose
weights and checkpoints, and outer folds remain held out. Fresh-data evaluation
will remain marked unavailable, and the repeatedly inspected historical test is
not treated as fresh data.

## 35. F1-focused overnight roadmap

The new priority is overall complete-pair micro-F1, with an eight-hour compute
budget. Minority-class regressions are reported but no longer veto a higher-F1
candidate. Distillation and automatic progression through the old roadmap are
disabled. The first implementation completed in 4.79 hours but its comparison
was invalidated: offline mode combined with the inherited
`fix_mistral_regex=True` option changed DeBERTa tokenization. The reused baseline
scored 0.5137 instead of 0.7667. Disabling the rewrite restored fold-1 inner
gold-aspect polarity accuracy from 0.5297 to 0.8960, exactly matching the saved
checkpoint's class metrics. No apparent gain from this attempt is valid.

The sequence is uncertainty-aware aspect selection using existing fold models,
weaker polarity class weighting (inverse-frequency exponents 0.5 and 0), an
optional lower learning rate (1e-5 versus 2e-5), and three-seed confirmation of
the inner-selected recipe with a small baseline/challenger interpolation grid.
Confirmation receives priority over the optional learning-rate stage.

The runner preserves the five frozen folds, existing all-class SemEval fitting
data and exclusions, and baseline seed-matched aspect predictions for training
checkpoint selection. Final compositions use the frozen three-seed aspects.
Checkpoint, decoder, and blend choices use inner selection only. The historical
test is not evaluated. Promotion requires +0.005 pooled pair F1, at least four
fold wins, and a positive group-bootstrap interval after three-seed evaluation;
incomplete and seed-42-only results remain screens.

Entry point: `scripts/run_f1_roadmap.py --budget-hours 8`.
Invalidated artifacts: `artifacts/experiments/f1-roadmap-v1/`, including the
superseding `INVALIDATED.json`. Original models and raw results are preserved.
Corrected artifacts use `artifacts/experiments/f1-roadmap-v2/`, an explicitly
unmodified DeBERTa tokenizer, persisted tokenizer fingerprints, and a mandatory
five-fold reproduction gate before training. The gate rejects an absolute
baseline pair-F1 difference above 0.001 or prediction-set disagreement above
0.005. A corrected attempt is limited to the remaining three hours of compute,
including the smoke test; incomplete stages are deferred, not called winners.
The corrected runner was launched in `NLP:5.1`, pane `%49`, launcher PID
`3204709`, after 174 passing unit tests and shell/whitespace checks. Its log is
`artifacts/experiments/f1-roadmap-v2/run.log`; it closes automatically and queues
a completion notification without launching further work.

The corrected run finished successfully after 1.92 hours. It reproduced the
recorded three-seed baseline exactly on every fold before training. The
three-seed uncertainty-aware decoder scored `0.7672` pooled OOF pair F1 versus
`0.7667` for ordinary decoding, a delta of only `+0.0005`; it won three of five
folds and its normalized-text-group bootstrap interval was
`[-0.0033, +0.0042]`. This is effectively a tie and fails the advancement gate.

The weaker-weight seed-42 screen was more promising:

| Polarity weighting | Pair F1 | Aspect F1 | Polarity F1 | Exact set | Delta vs seed-42 control | Fold wins | 95% group CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Full inverse-frequency control | 0.7419 | 0.8903 | 0.8093 | 0.6683 | — | — | — |
| Half weighting (`exponent=0.5`) | **0.7646** | 0.8890 | **0.8319** | **0.6876** | **+0.0227** | **5/5** | **[+0.0120, +0.0335]** |
| No class weighting (`exponent=0`) | 0.7610 | 0.8894 | 0.8300 | 0.6818 | +0.0191 | **5/5** | **[+0.0078, +0.0303]** |

Half weighting improved gold-aspect polarity accuracy from `0.8234` to `0.8475`
and macro-F1 from `0.6856` to `0.7191`. Its composed-pair class F1 changes were
`+0.0168` positive, `+0.0170` negative, `+0.0153` neutral, and `+0.0770`
conflict. Removing weights entirely produced the largest conflict gain
(`+0.1171`) but lower overall pair F1 than half weighting.

This is not a new winner yet. Both challengers are single-seed screens; the
half-weight model's `0.7646` remains `0.0021` below the existing three-seed
baseline at `0.7667`. Inner selection chose no weighting for fold 1 and half
weighting for folds 2–5. The lower-learning-rate stage was skipped and the ten
required seed-17/73 confirmation fits were deferred because they could not fit
inside the remaining budget. The correct next experiment, if resumed, is only
the preselected three-seed weighting confirmation—not another broad screen.
Full results are in `artifacts/experiments/f1-roadmap-v2/report.md`.

That confirmation was resumed on 2026-09-13 with the fold recipe frozen above.
It trains only the missing seeds 17 and 73 for the selected arm on each fold
(ten fits total), reuses the seed-42 screens, and has no fixed runtime deadline.
It does not revisit decoder/arm selection, evaluate the historical test, or
launch another experiment automatically. The job is running in tmux window
`NLP:5`, pane `%50`, launcher PID `3471803`; its attempt directory is
`artifacts/experiments/f1-roadmap-v2/attempts/confirmation-ZKT9qXKc/` and its
live log is `artifacts/experiments/f1-roadmap-v2/confirmation.log`. The pane
closes and queues this thread after writing the final exit status. Rankings stay
unchanged until all confirmation metrics are available.

The confirmation finished successfully. The frozen recipe reached **`0.7761`
pooled OOF pair F1**, compared with `0.7667` for the matched three-seed control:
`+0.0094`. It improved four of five outer folds and its 10,000-sample
normalized-text-group bootstrap interval was **`[+0.0008, +0.0179]`**. It
therefore passes every predeclared advancement condition and becomes the new
grouped-CV winner. Aspect F1 remained `0.8904`; polarity F1 rose from `0.8356`
to `0.8418`; exact-set accuracy rose from `0.6934` to `0.6954`.

The pair-class F1 changes were `+0.0066` positive, `+0.0127` negative,
`-0.0057` neutral, and `+0.0242` conflict. Gold-aspect polarity accuracy rose
from `0.8499` to `0.8574`, and macro-F1 rose from `0.7275` to `0.7364`. The
separate interpolation/decoder variant scored `0.7748`, but its bootstrap lower
bound was slightly negative (`-0.0002`), so the simpler confirmed training
recipe is retained. This is grouped-CV development evidence; it has not been
evaluated on the historical test or trained as one full-data deployable model.
Detailed protocol: `docs/experiments-proposal/F1_OVERNIGHT_ROADMAP.md`.

## 36. DeBERTa-v3-large LoRA polarity pilot

The seed-42 `microsoft/deberta-v3-large` pilot adapted all 48 attention query
and value projections across 24 layers with rank-16 LoRA, alpha 32, and dropout
0.05. The pretrained backbone remained frozen; the adapters, pooler, and
four-class polarity head contributed 2,626,564 trainable parameters. CPU and
GPU preflights verified finite updates, unchanged frozen weights, checkpoint
reload equivalence, and all five fold inputs. The successful ten-minute
benchmark selected microbatch 16 without gradient accumulation.

Each fold trained its polarity model on approximately 1,818 contest annotations
from the established training partition plus 1,021 external all-class SemEval
annotations. Inner selection retained no class weighting for fold 1 and
half-strength inverse-frequency weighting for folds 2–5. Existing three-seed
RoBERTa aspect outputs remained frozen.

| Evaluation | Pair F1 | Aspect F1 | Polarity F1 | Exact set |
|---|---:|---:|---:|---:|
| Pooled grouped-CV OOF | **0.7800** | 0.8893 | 0.8476 | 0.7050 |
| Locked historical test | 0.7727 | 0.8734 | **0.8561** | 0.6899 |

The OOF result improved the matched original seed-42 system by `+0.0381`, won
all five folds, and had a normalized-text-group bootstrap interval of
`[+0.0254, +0.0511]`. Against the original three-seed DeBERTa comparison it
improved by `+0.0133`, won three folds, and had interval
`[+0.0013, +0.0250]`. It is also `+0.0039` above the confirmed weaker-weight
three-seed system's `0.7761`, but that immediate comparison has no dedicated
paired interval and LoRA remains a single-seed pilot. Treat `0.7800` as the
highest OOF point estimate, not a multi-seed confirmation.

The earlier live values near `0.81`–`0.85` were inner checkpoint-selection
scores, not outer-fold generalization. The correct held-out fold pair F1 values
were `0.7824`, `0.8039`, `0.7948`, `0.7576`, and `0.7612`; the pooled OOF metric
is the authoritative CV result.

The historical-test protocol froze all five LoRA checkpoints, their inner-
selected decoders, the existing fold-specific three-seed RoBERTa aspect
checkpoints, and a 3-of-5 pair-voting rule before reading labels. It performed
inference only and covered all 258 reviews. The resulting `0.7727` pair F1 was
`+0.0111` above the confirmed class-weight CV committee at `0.7616`, but
`-0.0017` below the full-data historical-test leader at `0.7744`. Gold-aspect
polarity accuracy was `0.8576` with macro-F1 `0.7052`; class F1 was `0.9368`
positive, `0.8497` negative, `0.6471` neutral, and `0.3871` conflict. The larger
backbone therefore improved overall polarity strength but did not solve
conflict recall.

The model generalized with only a `0.0073` pair-F1 drop from pooled OOF to the
historical test, but it is a five-fold committee rather than one full-data
deployable checkpoint. The `0.7744` all-class SemEval model remains the
historical-test and deployable leader. Artifacts are under
`artifacts/experiments/deberta-v3-large-lora-seed42-v1/`; the locked test report
is in its `test/` subdirectory.

## 37. UWB-style count and TF-IDF ensemble

The constrained feature-based architecture from UWB's SemEval 2014 system was
recreated with count and TF-IDF representations, one-vs-rest aspect classifiers,
and aspect-conditioned polarity classifiers. Count, TF-IDF, and their combined
representation were evaluated alone and interpolated with the frozen
DeBERTa-v3-large LoRA transformer outputs. Sparse-model hyperparameters and
separate aspect/polarity interpolation weights were selected inside each outer
fold. The experiment used the established five grouped folds and did not read
the historical or private test labels.

| System | Pair F1 | Aspect micro | Aspect macro | Polarity micro | Polarity macro | Exact set |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | **0.7800** | **0.8893** | **0.8882** | 0.8476 | 0.7314 | **0.7050** |
| Count | 0.5339 | 0.7738 | 0.7450 | 0.6456 | 0.4752 | 0.4434 |
| TF-IDF | 0.5369 | 0.7784 | 0.7519 | 0.6470 | 0.4786 | 0.4454 |
| Count plus TF-IDF | 0.5324 | 0.7677 | 0.7418 | 0.6468 | 0.4848 | 0.4458 |
| Transformer plus count | 0.7787 | 0.8877 | 0.8848 | 0.8484 | 0.7332 | 0.7045 |
| Transformer plus TF-IDF | 0.7764 | 0.8841 | 0.8803 | **0.8489** | **0.7343** | 0.6992 |
| Transformer plus combined | 0.7750 | 0.8831 | 0.8798 | 0.8461 | 0.7309 | 0.6987 |

No ensemble passed the promotion rules. Relative to the transformer, count
changed pair F1 by `-0.0014` with one of five fold wins and bootstrap interval
`[-0.0057, +0.0031]`; TF-IDF changed it by `-0.0036` with interval
`[-0.0078, +0.0006]`; the combined representation changed it by `-0.0051` with
interval `[-0.0098, -0.0005]`. Inner selection usually assigned no weight to
the sparse polarity outputs. The small polarity improvements from two blends
were outweighed by weaker aspect prediction, so no historical-test evaluation
was launched.

The standalone result is much less negative than its end-to-end pair F1 first
suggests. On metrics comparable to the original UWB constrained ten-fold
results, this implementation obtained `77.84%` aspect-category F1 versus
UWB's `77.51%`, and `66.93%` gold-aspect polarity accuracy versus UWB's
`66.69%`. UWB evaluated aspect detection and supplied-aspect polarity as
separate subtasks; it did not report the stricter end-to-end pair F1 used here.
Its stronger unconstrained entry also added LDA topics, word clusters, sentiment
lexicons, SentiWordNet, and representations learned from a large external review
corpus. The experiment therefore reproduced the constrained UWB baseline
reasonably well, but showed that its lexical evidence is not sufficiently
complementary to the modern transformer. Full results are in
`artifacts/experiments/uwb-tfidf-ensemble-v1/report.md`.

## Overall conclusion

The meaningful progression in untouched or increasingly rigorous pair-level
evaluation was:

`0.6234` zero-shot -> `0.6725` single-label baseline -> `0.7223` corrected
conditioned pipeline -> `0.7448` previous separate ensemble -> `0.7544` new
aspect ensemble plus existing polarity ensemble -> `0.7630` RoBERTa aspect
plus mixed RoBERTa/DeBERTa polarity ensemble -> `0.7744` all-class SemEval
DeBERTa polarity. The later LoRA CV committee reached `0.7727`, close to but not
above that historical-test leader.

The largest gain came from correcting the task formulation, not changing the loss.
Multilabel aspect prediction plus aspect-conditioned polarity was essential. Loss
weighting and focal loss produced smaller, architecture-dependent effects.

The strongest validation-locked complete system from the backbone comparison
remains the three-seed RoBERTa multilabel aspect ensemble composed with an equal
blend of the three-seed RoBERTa and DeBERTa weighted-CE polarity ensembles at
`0.7630` historical-test pair F1. The later all-class SemEval experiment
produced a higher observed score of `0.7744` from augmented DeBERTa polarity,
but validation selected its mixed candidate instead (`0.7680` on test), and
uncertainty intervals included zero. The earlier
joint-versus-separate protocol still selected the joint focal model; later
experiments do not retroactively change that earlier locked decision.
The grouped-CV LoRA pilot later produced the highest OOF point estimate at
`0.7800` and generalized to `0.7727` on historical test. It establishes
DeBERTa-v3-large LoRA as a strong polarity direction, but it neither exceeds
the `0.7744` historical-test point estimate nor supplies a single full-data
checkpoint.

Additional neutral/conflict reweighting was also selected on validation, but it
did not improve the strongest observed test result. It traded a large neutral
gain for a larger conflict loss, reducing pair F1 from `0.7544` to `0.7357`.
Post-hoc calibration and explicit positive/negative evidence modeling were then
tested. Neither established a better complete system: calibrated evidence
matched the baseline's test pair F1 but reduced conflict F1, while evidence-only
decoding fell to `0.7322` pair F1 and `0.2963` conflict F1.

Future architecture and loss comparisons should continue using grouped
cross-validation or repeated validation splits to make selection more stable.
The completed learning curve found a statistically credible 75% to 100% gain
for complete pair prediction,
so additional high-quality annotation is now better supported than further
oversampling of existing examples. Aspect-only and polarity-only gains remain
inconclusive. A genuinely new final test partition would still be required for
another unbiased model-selection claim.

The later multi-seed conflict isolation also did not support promotion. Although
synthetic conflict examples improved the three-seed point estimates over matched
repetition, the uncertainty interval included zero and only one of three seeds
improved conflict F1. Synthetic clause joining is therefore not an established
replacement for natural labeled data.

SemEval augmentation provided the clearest recent direction. Including all
four polarities preserved broad classification quality and reached the highest
observed pair F1. Keeping only neutral and conflict improved those two classes
but harmed positive and negative enough to remove the overall gain. Future data
augmentation should therefore preserve representative majority examples while
improving minority coverage, rather than changing the training distribution to
minority-only external data.

The later full-combined joint DeBERTa pilot remained competitive on validation
but reached only `0.7520` historical-test pair F1. Post-hoc decoder tuning raised
validation pair F1 to `0.8019` while reducing test pair F1 to `0.7437`, providing
a direct example of decoder overfitting on the repeatedly used validation split.
Future composition and calibration choices should be selected inside grouped CV.
The saved all-class/minority DeBERTa interpolation likewise produced no overall
pair-F1 gain: 25% minority weight improved conflict and macro-F1 but tied the
all-class endpoint at `0.7863`. This does not justify carrying the extra
minority ensemble into the initial CV comparison.
ModernBERT also failed the architecture-diversity screen: its standalone pair
F1 was `0.7395`, and its best DeBERTa interpolation selected zero ModernBERT
weight. It should not be included in grouped CV unless a materially different
training recipe is justified independently.
ELECTRA did show useful diversity: 25% ELECTRA weight improved the practical
DeBERTa validation composition by `0.0057` and improved both neutral and
conflict gold-aspect F1. It replaces ModernBERT and the minority-only DeBERTa
as the challenger in the initial grouped-CV comparison.
The grouped CV then supported that decision: the inner-selected ELECTRA blend
improved pooled pair F1 by `0.0190`, won all five folds, improved both minority
classes, and had a bootstrap interval above zero. The next stage is confirmation
with training seeds 17 and 73 on the same fold assignments.
That confirmation reversed the pilot conclusion: the three-seed ELECTRA blend
scored `0.7637` versus `0.7667` for DeBERTa alone and won only one fold. ELECTRA
is therefore rejected for promotion despite the strong seed-42 screen.
The later UWB-style sparse experiment reached results close to the original
constrained system on comparable isolated aspect and gold-aspect polarity
metrics, but all three transformer blends reduced end-to-end pair F1. It is a
successful historical-baseline reproduction, not a new ensemble candidate.

## Related reports

- `docs/REPORT.md`
- `docs/OVERALL_F1_COMPARISON.md`
- `artifacts/ensemble/heldout-20-percent/report.md`
- `artifacts/ensemble/full-dataset/report.md`
- `artifacts/experiments/multilabel-conditioned-v1/error_analysis.md`
- `artifacts/experiments/absa-imbalance-joint-v1/report.md`
- `artifacts/experiments/multilabel-aspect-v1/report.md`
- `artifacts/experiments/multilabel-aspect-old-polarity-v1/report.md`
- `artifacts/experiments/multilabel-aspect-old-polarity-v1/polarity_error_audit.md`
- `artifacts/experiments/polarity-reweighting-v1/report.md`
- `artifacts/experiments/polarity-evidence-v1/report.md`
- `artifacts/experiments/polarity-evidence-v1/raw_evidence_only.json`
- `artifacts/experiments/deberta-v3-current-best-v1/report.md`
- `artifacts/experiments/deberta-v3-three-systems-v1/systems/deberta-only/report.md`
- `artifacts/experiments/deberta-v3-three-systems-v1/systems/roberta-aspect-deberta-polarity/report.md`
- `artifacts/experiments/deberta-v3-three-systems-v1/systems/roberta-aspect-mixed-polarity/report.md`
- `artifacts/experiments/deberta-v3-three-systems-v1/systems/roberta-aspect-mixed-polarity/aspect_error_audit.md`
- `artifacts/experiments/deberta-v3-three-systems-v1/systems/roberta-aspect-mixed-polarity/misc_conservatism_audit.md`
- `artifacts/experiments/deberta-v3-three-systems-v1/systems/roberta-aspect-mixed-polarity/polarity_error_audit.md`
- `artifacts/experiments/aspect-misc-oversampling-cv-v1/report.md`
- `artifacts/experiments/oof-label-audit-v1/report.md`
- `artifacts/experiments/oof-label-audit-v1/review_cases.csv`
- `artifacts/experiments/learning-curve-v1/report.md`
- `artifacts/experiments/conflict-augmentation-multiseed-v1/report.md`
- `artifacts/experiments/semeval14-deberta-polarity-3seed-v1/report.md`
- `artifacts/experiments/semeval14-minority-deberta-polarity-3seed-v1/report.md`
- `artifacts/experiments/semeval14-full-joint-deberta-seed42-v1/report.md`
- `artifacts/experiments/semeval14-full-joint-deberta-decoder-v1/report.md`
- `artifacts/experiments/semeval14-full-joint-deberta-decoder-v1/test/report.md`
- `artifacts/experiments/deberta-logit-interpolation-v1/report.md`
- `artifacts/experiments/modernbert-polarity-seed42-v1/report.md`
- `artifacts/experiments/electra-polarity-seed42-v1/report.md`
- `artifacts/experiments/electra-deberta-grouped-cv-v1/report.md`
- `artifacts/experiments/electra-deberta-three-seed-confirmation-v1/report.md`
- `artifacts/experiments/f1-roadmap-v2/report.md`
- `artifacts/experiments/deberta-v3-large-lora-seed42-v1/report.md`
- `artifacts/experiments/deberta-v3-large-lora-seed42-v1/test/report.md`
- `artifacts/experiments/uwb-tfidf-ensemble-v1/report.md`
