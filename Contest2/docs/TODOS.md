# Experiment roadmap

Detailed scopes, procedures, and interpretation rules for all 12 experiments
are in [the experiment proposal index](experiments-proposal/README.md).

Planned follow-up to the SemEval augmentation experiments. This document records
proposed work; unchecked items have not been completed or launched by this roadmap.

## Objective and baseline

Improve complete `(id, aspectCategory, polarity)` pair micro-F1 through
complementary models, logit interpolation, and grouped cross-validation (CV).
Track neutral/conflict performance alongside overall performance.

- Highest observed historical-test pair F1: **0.7744**, using the frozen
  three-seed RoBERTa aspect ensemble and all-class SemEval DeBERTa polarity.
- Validation selected the all-class augmented mixed-polarity candidate, which
  scored **0.7680** on the historical test partition.
- Minority-only augmentation improved neutral/conflict gold-aspect F1 but
  reduced overall performance. It is a potential ensemble component.
- The historical test partition has been inspected repeatedly. Use it only for
  explicitly labeled historical comparisons, not for tuning the new roadmap.

## 0. Correct joint DeBERTa supervision

- [ ] Repair the partial-annotation problem before interpreting the joint pilot.
  Filtering SemEval to neutral/conflict removed 32 known present aspect targets
  across 30 retained reviews. The current joint candidate builder treats these
  omitted aspects as absent, creating incorrect training targets.
- [ ] Recover complete aspect-presence labels for retained SemEval reviews from
  the unfiltered annotations. Supervise polarity only for their neutral/conflict
  annotations; preserve full supervision for original contest rows.
- [ ] Introduce separate aspect-presence and polarity-supervision masks. Test
  that omitted positive/negative annotations remain present aspects and receive
  no polarity-loss gradient.
- [ ] Restart the corrected seed-42 pilot in a new artifact directory. Mark the
  earlier version as having incomplete aspect supervision; retain its artifacts.
- [ ] Keep the pilot validation-only. Compare with the separate minority
  DeBERTa seed-42 model and assess whether joint training merits a CV arm.

## 1. Interpolate existing DeBERTa models

- [ ] Cache validation polarity logits from the all-class and minority-only
  three-seed DeBERTa ensembles. Align by review ID, candidate aspect, and polarity
  label order; verify text and split provenance before combining outputs.
- [ ] Average seed logits within each component, then evaluate:

  ```text
  combined_logits = (1 - alpha) * all_class_logits + alpha * minority_logits
  alpha in {0, 0.25, 0.5, 0.75, 1}
  ```

- [ ] Select alpha and aspect thresholds on validation pair F1. Include both
  endpoints so the selected result can exclude an unhelpful component. For exact
  score ties, prefer less minority-model weight.
- [ ] Report standalone and blended metrics, baseline errors corrected, correct
  predictions broken, and class-specific error overlap. Label this exploratory
  screening because the existing validation set has guided previous experiments.

## 2. Screen different polarity backbones

- [ ] Train `answerdotai/ModernBERT-base` first and
  `google/electra-base-discriminator` second, using seed 42 and all-class SemEval
  augmentation. Keep the aspect-conditioned four-class task and weighted
  cross-entropy loss.
- [ ] Start with the DeBERTa reference settings: learning rate `2e-5`, maximum
  length 256, effective batch size 16, weight decay 0.01, warmup ratio 0.1,
  maximum 50 epochs, validation every 5 epochs, and patience of 3 evaluations.
  Record any required compatibility or memory adjustments.
- [ ] Measure throughput and peak memory for ten minutes before extrapolating
  runtime. Preserve FP32 trainable weights with mixed-precision computation and
  verify optimizer updates and inference compatibility before a full run.
- [ ] Evaluate each model alone and interpolated with all-class DeBERTa using
  the same five-value alpha grid. Select the backbone whose best validation
  blend has the highest pair F1; break ties by standalone pair F1, then lower
  measured inference cost. Do not require a standalone win to retain a useful
  ensemble component.

The hypothesis is complementary errors, not guaranteed superiority. DeBERTa-v3
already uses ELECTRA-style pretraining, so ELECTRA is not our first test of
replaced-token detection. Sources: [ModernBERT model card](https://huggingface.co/answerdotai/ModernBERT-base),
[ELECTRA model card](https://huggingface.co/google/electra-base-discriminator),
[DeBERTa-v3 model card](https://huggingface.co/microsoft/deberta-v3-base).

## 3. Five-fold CV with selection inside each fold

- [ ] Freeze five approximately stratified outer folds over the original contest
  fitting partition. Group identical normalized texts across IDs, preserve all
  annotation rows, and audit near duplicates before freezing groups. Balance
  aspect, polarity, and multi-aspect coverage where possible.
- [ ] Reserve a fixed grouped inner validation subset from each outer training
  partition. Fit weights on the remaining data; choose checkpoints, interpolation
  weights, and aspect thresholds using inner validation only. Score locked
  choices on the outer holdout.
- [ ] Add cleaned SemEval rows only to each fold's fitting data. Verify no
  external exact/near duplicate crosses into inner validation or outer holdout.
- [ ] Train these four components per fold, initially with seed 42:

  | Component | Fitting data |
  |---|---|
  | RoBERTa multilabel aspect | Original contest rows |
  | Baseline DeBERTa polarity | Original rows + all-class SemEval |
  | Specialist DeBERTa polarity | Original rows + minority-only SemEval |
  | Selected new backbone polarity | Original rows + all-class SemEval |

- [ ] Budget **20 fits**. Share each fold's aspect model across polarity
  comparisons. Do not reuse existing full-fit checkpoints that have seen outer
  holdout reviews. Existing split helpers may be reused after adding text-group
  handling and leakage checks.
- [ ] Predefine three recipes: baseline alone, baseline plus minority DeBERTa,
  and baseline plus the selected new backbone. Tune alpha only on inner
  validation. Use all-class DeBERTa as the baseline component in each blend.
- [ ] Report pooled out-of-fold pair F1, individual fold scores, gold-aspect
  polarity accuracy/macro-F1, per-class precision/recall/F1, exact-set accuracy,
  selected alphas, and 10,000-sample paired review-group bootstrap intervals.
  Keep gold-aspect polarity metrics distinct from the official evaluator's
  sentiment metrics.
- [ ] Consider a corrected joint DeBERTa arm only after its pilot is useful;
  it adds five fits. Do not automatically expand the initial 20-fit comparison.

## 4. Confirm the strongest recipe

- [ ] Advance a recipe if pooled pair F1 improves by at least **0.005**, it wins
  on at least **four of five folds**, and neither neutral nor conflict
  gold-aspect F1 decreases by more than **0.02** against the matched baseline.
- [ ] Repeat the baseline and winning recipe's required components with seeds
  17 and 73 on the same folds. Share matched aspect models and reuse completed
  fold/seed components where provenance permits.
- [ ] Report seed variation separately from review-bootstrap uncertainty. An
  interval crossing zero is promising but unconfirmed evidence, not a decisive
  improvement.
- [ ] Freeze the selected recipe before final fitting and historical comparison.
  Repeated use of CV results also makes them development evidence; a new final
  test partition is needed for a fresh unbiased performance claim.

## Execution and verification

- [ ] Use resumable manifests with source hashes, split/group assignments,
  tokenizer/model revisions, seeds, checkpoint provenance, and locked selection.
  Save reusable logits and lightweight metrics/reports for every condition.
- [ ] Test split isolation, complete multi-aspect labels, joint supervision masks,
  logit alignment, interpolation endpoints, and selection independence from
  outer holdout labels. Run the repository test suite with the training environment.
- [ ] Launch GPU work sequentially through one orchestrator after current jobs
  finish, with separate artifact directories per experiment/fold/seed.
- [ ] Follow repository tmux conventions: visible logs, recorded process IDs and
  commands, attempt-specific status to avoid stale failure notifications,
  automatic closure, and completion notification to the originating thread.
- [ ] Update `docs/SUMMARY.md` after completed stages with observed results,
  validation selection, uncertainty, and any changes to the plan.

Defer class-specific interpolation weights and learned stacking until a global
blend demonstrates useful performance. The immediate sequence is: correct joint
labels, blend existing models, screen ModernBERT and ELECTRA, then confirm with CV.
