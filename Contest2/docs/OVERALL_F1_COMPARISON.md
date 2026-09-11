# Overall Pair-F1 Comparison

## Metric and scope

“Overall F1” means micro-F1 over complete
`(id, aspectCategory, polarity)` predictions. A prediction is correct only when
both the aspect and polarity match a gold pair for the same review ID.

The tables include every final system or explicitly reported candidate in the
saved experiment reports. Per-epoch checkpoints are omitted: they are training
history, not separately selected systems. Validation, test, combined holdout,
and in-sample results are kept separate because they are not interchangeable.

## Held-out test results, ranked

| Rank | Experiment | System | Overall pair micro-F1 | Selection note |
|---:|---|---|---:|---|
| 1 | DeBERTa three-system | RoBERTa aspect + mixed RoBERTa/DeBERTa polarity | **0.762987** | Validation-selected current winner; exactly reproduced |
| 2 | DeBERTa three-system | RoBERTa aspect + DeBERTa polarity | 0.761600 | Separate validation lock |
| 3 | New aspect composition | RoBERTa three-seed aspect + RoBERTa three-seed polarity | 0.754358 | Validation-selected in that experiment |
| 3 | Evidence model | Original weighted-CE ensemble | 0.754358 | Baseline repeated for comparison |
| 3 | Evidence model | Calibrated evidence model | 0.754358 | Not the validation-locked winner |
| 6 | Evidence model | Evidence four-class head | 0.751189 | Analysis candidate |
| 7 | DeBERTa three-system | DeBERTa aspect + DeBERTa polarity | 0.748815 | Separate validation lock |
| 8 | Evidence model | Calibrated original | 0.744849 | Validation-locked evidence-experiment winner |
| 9 | Imbalance/joint experiment | Separate weighted-CE ensemble | 0.744783 | Highest observed test score in that experiment, not its locked winner |
| 10 | Reweighting | Neutral 1.5x, conflict 2x ensemble | 0.735669 | Validation-locked reweighting winner |
| 11 | Evidence model | Raw evidence-only decoder | 0.732200 | Auxiliary evidence logits only |
| 12 | Imbalance/joint experiment | Joint focal-gamma-2 ensemble | 0.724409 | Validation-locked winner in that experiment |
| 13 | Conditioned pipeline | One-seed multilabel aspect + conditioned polarity | 0.722311 | Validation-selected |
| 14 | Legacy baseline | Single-label aspect/polarity | 0.672474 | Cannot emit multiple pairs per review |

The reweighting experiment's baseline is also `0.754358`; it is the same
RoBERTa three-seed composition already shown above, not another model. The
current winner rerun regenerated the same 300 predictions and reproduced
`0.762987` exactly.

The original zero-shot pipeline reported `0.6234`, but its evaluation predates
the common 80/10/10 experiment protocol, so it is a historical reference rather
than a strictly comparable test entry.

## Validation comparisons by experiment

### Early and formulation experiments

| Experiment | System | Validation pair micro-F1 |
|---|---|---:|
| Custom two-model RoBERTa | Separate single-label models | 0.6318 |
| Multilabel-conditioned | Corrected conditioned pipeline | 0.7523 |

The early Hugging Face aspect, polarity, and multitask models reported row-level
accuracy or a generic best metric, not overall pair F1. The standalone
single-task runs likewise reported task-specific micro-F1 only.

### Separate polarity-loss screen, seed 42

| Loss | Validation pair micro-F1 |
|---|---:|
| Weighted cross-entropy | **0.773994** |
| Weighted focal, gamma 1 | 0.761610 |
| Weighted focal, gamma 2 | 0.767802 |
| Class-balanced focal, beta 0.999, gamma 2 | 0.755418 |

### Joint architecture screen, seed 42

| Joint polarity loss | Validation pair micro-F1 |
|---|---:|
| Weighted cross-entropy | 0.775701 |
| Weighted focal, gamma 2 | **0.783570** |

### Three-seed confirmation

| Architecture and loss | Seed | Validation pair micro-F1 |
|---|---:|---:|
| Separate weighted CE | 17 | 0.7523 |
| Separate weighted CE | 42 | 0.7740 |
| Separate weighted CE | 73 | 0.7570 |
| Separate weighted-CE ensemble | ensemble | 0.755418 |
| Joint focal, gamma 2 | 17 | 0.7664 |
| Joint focal, gamma 2 | 42 | 0.7836 |
| Joint focal, gamma 2 | 73 | 0.7607 |
| Joint focal-gamma-2 ensemble | ensemble | **0.785047** |

### New aspect ensemble with existing polarity models

| Polarity candidate | Validation pair micro-F1 |
|---|---:|
| Previous one-seed conditioned polarity | 0.757098 |
| Three-seed weighted-CE polarity | **0.765891** |

### Neutral/conflict reweighting screen, seed 42

| Candidate | Validation pair micro-F1 |
|---|---:|
| Neutral 1.25x, conflict 1.5x | 0.755832 |
| Neutral 1.5x, conflict 2x | **0.786936** |
| Neutral 2x, conflict 2x | 0.778125 |
| Neutral 1.5x, conflict 3x | 0.771384 |
| Neutral 2x, conflict 3x | 0.765163 |

After three-seed confirmation, the original ensemble scored `0.765891` and the
reweighted ensemble scored `0.777605`. The reweighted model was locked, but its
test F1 fell to `0.735669`.

### Positive/negative evidence experiment

| System | Validation pair micro-F1 | Test pair micro-F1 |
|---|---:|---:|
| Original weighted-CE ensemble | 0.765891 | **0.754358** |
| Calibrated original | **0.778295** | 0.744849 |
| Evidence model, four-class head | 0.765891 | 0.751189 |
| Calibrated evidence model | 0.772093 | **0.754358** |
| Raw evidence-only decoder | 0.765900 | 0.732200 |

The calibrated original was selected on validation. Calibration and evidence
did not establish a test improvement over the original model.

### DeBERTa experiments

| System | Validation pair micro-F1 | Test pair micro-F1 |
|---|---:|---:|
| DeBERTa seed-42 pilot | 0.766823 | Not evaluated |
| DeBERTa aspect + DeBERTa polarity | 0.774803 | 0.748815 |
| RoBERTa aspect + DeBERTa polarity | 0.776911 | 0.761600 |
| RoBERTa aspect + mixed RoBERTa/DeBERTa polarity | **0.779014** | **0.762987** |

The mixed system was locked by validation pair F1 before test inference.

## Evaluations that are not overall pair F1

These results answer narrower questions and must not be inserted into the
overall ranking:

| Experiment | Metric | Best reported result |
|---|---|---:|
| Early RoBERTa aspect | Row-level validation accuracy | about 0.7350 |
| Early RoBERTa polarity | Row-level validation accuracy | about 0.8044 |
| Early RoBERTa multitask | Generic best training metric | 0.7492 |
| Long single-task aspect | Validation aspect micro-F1 | 0.7923 |
| Long single-task polarity | Validation polarity micro-F1 | 0.8496 |
| Three-seed multilabel aspect | Test aspect micro-F1 | 0.877419 |
| Polarity oversampling CV reference | Gold-aspect polarity macro-F1 | 0.689644 |
| Polarity oversampling CV loss-control | Gold-aspect polarity macro-F1 | 0.685535 |
| Polarity oversampling CV oversampling | Gold-aspect polarity macro-F1 | 0.686531 |
| Polarity oversampling CV sampling-corrected | Gold-aspect polarity macro-F1 | 0.686042 |
| Corrected RoBERTa polarity ensemble | Gold-aspect polarity macro-F1 | 0.708798 |
| Corrected DeBERTa polarity ensemble | Gold-aspect polarity macro-F1 | 0.718703 |
| Corrected mixed polarity ensemble | Gold-aspect polarity macro-F1 | **0.721454** |

## Non-generalization diagnostics

| Evaluation | Data scope | Overall pair micro-F1 |
|---|---|---:|
| Best-checkpoint single-label ensemble | Validation + test combined holdout | 0.676548 |
| Best-checkpoint single-label ensemble | Full labeled training dataset | 0.835134 |

The combined holdout contains validation-selection bias. The full-dataset score
is in-sample and is not evidence that the single-label system outperforms the
current winner.

## Conclusion

The highest reproducible held-out test result is **0.762987 overall pair
micro-F1** from the RoBERTa multilabel aspect ensemble combined with the mixed
RoBERTa/DeBERTa polarity ensemble. Its lead over RoBERTa-aspect plus DeBERTa
polarity is only `0.001387`, and its lead over the previous RoBERTa-only
composition is `0.008629`.

Because the same test partition has been examined by several later experiments,
these are strongest observed results rather than a fresh unbiased model ranking.
