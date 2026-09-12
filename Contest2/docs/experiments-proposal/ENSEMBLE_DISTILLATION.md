# Experiment 12: Ensemble Distillation and Fresh-Data Evaluation

## Scope

Compress a validated ensemble into one aspect-conditioned polarity model while
retaining the selected aspect component. Measure total pipeline cost. Evaluate
the frozen teacher and student on genuinely new labeled reviews when available.

## What this finds out

Can a single student retain most of the teacher's pair-F1 gain at lower latency
and memory cost? Does the established improvement persist beyond repeatedly used
development data?

## How to run

1. Freeze the teacher recipe from CV/seed confirmation. Use DeBERTa-v3-base as the
   first single polarity student and train an identical hard-label-only control.
2. Generate teacher logits for fitting inputs only. For CV experiments the teacher
   must itself be fold-specific and have no outer-holdout exposure. External
   unlabeled inputs require a separate permitted, deduplicated fitting pool.
3. Minimize `(1 - lambda) * CE + lambda * T^2 * KL(teacher_T || student_T)`
   with lambda in {0.25, 0.5, 0.75} and distillation temperature T=2. Use
   weighted CE from fitting labels and an unweighted soft-target KL term.
   Select lambda, checkpoint, and thresholds on inner validation.
4. Compare teacher, distilled student, and hard-label student on matched folds.
   Use identical aspect predictions and measure full-pipeline batch-1 latency,
   batch throughput, peak memory, checkpoint size, and pair/class metrics.
5. Advance a student if pair F1 is within 0.005 of the teacher, neither minority
   F1 loses more than 0.02, and full-pipeline median latency improves by at least
   25% on the same hardware and workload.
6. Before reading fresh labels, freeze the teacher/student choices and evaluation
   protocol. Use independently annotated, deduplicated reviews with full aspect
   sets, natural class prevalence, and reported source coverage.

## Interpretation and next action

Matching the teacher with lower cost supports deployment-oriented adoption.
Beating the hard-label control but missing the teacher quantifies partial
knowledge transfer. A small polarity speedup may have little practical value if
aspect inference dominates; report full-pipeline measurements.

Fresh-data regression indicates domain shift or development overfitting. Report
minority support and uncertainty before drawing conclusions from rare conflict
cases. Do not tune on a fresh final set after seeing results; use new development
data for subsequent changes. If no fresh labeled set is available, mark this
stage incomplete rather than presenting historical test as fresh.

## Verification and deliverables

Test teacher/student class alignment, detached teacher outputs, KL direction,
temperature scaling, and fitting-only teacher access. Save the student recipe,
accuracy-cost comparison, reproducible latency workload, and frozen fresh-data
evaluation manifest.

