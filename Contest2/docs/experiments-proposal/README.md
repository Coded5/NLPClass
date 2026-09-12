# Model improvement experiment proposals

Status: proposed. These documents specify future work; creating them does not
launch experiments. Follow the order below. Experiments 7–12 are conditional
follow-ups to the findings from experiments 1–6.

| Order | Proposal | Purpose |
|---|---|---|
| 1 | [Joint DeBERTa](JOINT_DEBERTA.md) | Correct supervision and test joint prediction |
| 2 | [DeBERTa interpolation](DEBERTA_LOGIT_INTERPOLATION.md) | Combine existing complementary models |
| 3 | [ModernBERT polarity](MODERNBERT_POLARITY.md) | Screen architecture diversity |
| 4 | [ELECTRA polarity](ELECTRA_POLARITY.md) | Screen another encoder and its ensemble value |
| 5 | [Grouped cross-validation](GROUPED_CROSS_VALIDATION.md) | Compare recipes across review partitions |
| 6 | [Three-seed confirmation](THREE_SEED_CONFIRMATION.md) | Measure sensitivity to training randomness |
| 7 | [External-data sampling](EXTERNAL_DATA_SAMPLING.md) | Find a middle ground between augmentation policies |
| 8 | [Temperature-scaled interpolation](TEMPERATURE_SCALED_INTERPOLATION.md) | Account for differing logit scales |
| 9 | [Clause evidence](CLAUSE_LEVEL_EVIDENCE.md) | Recognize opposing sentiment about the same aspect |
| 10 | [Aspect backbone ensemble](ASPECT_BACKBONE_ENSEMBLE.md) | Improve aspect detection |
| 11 | [Disagreement-driven data](DISAGREEMENT_DRIVEN_DATA.md) | Test the value of targeted additional labels |
| 12 | [Distillation and fresh evaluation](ENSEMBLE_DISTILLATION.md) | Reduce inference cost and test generalization |

## Shared evaluation protocol

The primary metric is official overall pair micro-F1 for complete
`(id, aspectCategory, polarity)` sets. Report aspect micro-F1, per-review exact-set
accuracy, and gold-aspect polarity accuracy, macro-F1, and per-class
precision/recall/F1 with support. Gold-aspect polarity evaluates every supplied
aspect and must be distinguished from the official evaluator's sentiment metric.

The historical-test score of 0.7744 is the highest observed all-class SemEval
DeBERTa composition score. Its experiment selected the mixed candidate on
validation, which scored 0.7680 on historical test. Neither score is the target
used to select new CV candidates: compare against a baseline trained on matching
folds, seeds, and supervision.

Use original contest fitting data for CV. Group IDs and identical normalized
texts together; preserve every valid annotation. Audit near duplicates before
freezing groups. Add external data only to fitting partitions and exclude its
matches to heldout groups. Keep historical validation/test and contest submission
data outside CV selection. Existing validation may be used for explicitly
exploratory pilots.

Every outer fold has a fitting partition, an inner selection partition, and an
outer scoring partition. Model training uses fitting data. Checkpoints,
interpolation weights, sampling settings, and thresholds use inner selection
only. Outer scores are revealed after those choices are locked. Do not load
full-data trained models as fold baselines.

Use five frozen outer folds with a grouped 10% inner selection partition from
each outer training partition. Fix split seed 42 and initially training seed 42.
Approximately balance aspects, polarities, and multi-aspect reviews. Save group
membership, sizes, label counts, and all exclusions.

For CV advancement, require at least +0.005 pooled pair F1, wins on at least four
of five folds, and no decrease greater than 0.02 in either neutral or conflict
gold-aspect F1. Report 10,000 paired bootstrap replicates resampling review/text
groups; preserve all annotations and match systems within replicates. An interval
crossing zero is inconclusive, even when the point-estimate gate passes. This
bootstrap does not account fully for model fitting or candidate-selection
uncertainty. Repeated CV comparisons remain development evidence.

## Shared training and artifacts

For polarity pilots, start with weighted cross-entropy, learning rate 2e-5,
maximum length 256, effective batch 16, weight decay 0.01, warmup ratio 0.1,
50 maximum epochs, validation every five epochs, patience three evaluations,
and minimum improvement 0.001. Compute training weights from fitting rows only.
Declare any exception in its proposal and manifest.

Save source hashes, model/tokenizer revisions, split/group assignments, seed,
loss settings, checkpoints, aligned logits, predictions, locked selections,
metrics, timings, and a report. Label logits by review ID, aspect, and explicit
class order; reject mismatches rather than silently relying on array position.

Before long runs, verify a finite forward/backward optimizer update, parameter
dtypes, checkpoint reload, and inference. Keep trainable master weights in FP32
when using FP16 autocast with GradScaler. Measure ten minutes of actual throughput
and peak GPU memory for new architectures before estimating total time.

Use sequential GPU jobs, resumable manifests, and separate artifacts per
experiment/fold/seed. Follow AGENTS.md: visible tmux logs, automatic closure,
attempt-specific status, and completion notification. Preserve earlier artifacts.
Update the summary only with verified results and limitations.

