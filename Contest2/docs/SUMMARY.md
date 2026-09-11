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

## Overall conclusion

The meaningful progression in untouched or increasingly rigorous pair-level
evaluation was:

`0.6234` zero-shot -> `0.6725` single-label baseline -> `0.7223` corrected
conditioned pipeline -> `0.7448` previous separate ensemble -> `0.7544` new
aspect ensemble plus existing polarity ensemble -> `0.7630` RoBERTa aspect
plus mixed RoBERTa/DeBERTa polarity ensemble.

The largest gain came from correcting the task formulation, not changing the loss.
Multilabel aspect prediction plus aspect-conditioned polarity was essential. Loss
weighting and focal loss produced smaller, architecture-dependent effects.

The strongest observed complete system is now the three-seed RoBERTa
multilabel aspect ensemble composed with an equal blend of the three-seed
RoBERTa and DeBERTa weighted-CE polarity ensembles. This system also had the
highest validation pair F1 among the three backbone combinations, so its test
result follows the experiment's locked selection. The earlier
joint-versus-separate protocol still selected the joint focal model; later
experiments do not retroactively change that earlier locked decision.

Additional neutral/conflict reweighting was also selected on validation, but it
did not improve the strongest observed test result. It traded a large neutral
gain for a larger conflict loss, reducing pair F1 from `0.7544` to `0.7357`.
Post-hoc calibration and explicit positive/negative evidence modeling were then
tested. Neither established a better complete system: calibrated evidence
matched the baseline's test pair F1 but reduced conflict F1, while evidence-only
decoding fell to `0.7322` pair F1 and `0.2963` conflict F1.

A future experiment should use grouped cross-validation or repeated validation
splits to make architecture selection more stable. A genuinely new final test
partition would be required for another unbiased model-selection claim.

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
