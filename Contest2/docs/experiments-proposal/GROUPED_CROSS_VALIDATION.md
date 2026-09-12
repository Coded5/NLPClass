# Experiment 5: Five-Fold Grouped Cross-Validation

## Scope

Compare baseline all-class DeBERTa, its minority-DeBERTa blend, and its blend
with the selected new backbone. The primary budget is 20 fits: four components
per fold, one training seed. Use the shared protocol in [README](README.md).

## What this finds out

Are the pilot gains consistent across different reviews? Does a recipe improve
performance when interpolation and thresholds are selected without access to
the reviews being scored?

## How to run

1. Freeze five grouped outer folds using split seed 42 over original contest
   fitting data. Keep identical normalized texts together even under different
   IDs. Audit near duplicates and preserve complete annotation sets.
2. Within each outer training fold, reserve a grouped approximately stratified
   10% inner selection partition. Fit on the remainder. Save all group assignments.
3. Add external SemEval rows only to fitting data after duplicate checks against
   heldout groups. Never augment selection or scoring partitions.
4. Train seed 42 for four components: original-data RoBERTa aspect, all-class
   DeBERTa polarity, minority DeBERTa polarity, and all-class new-backbone polarity.
   Share the aspect model across all three recipes.
5. Select checkpoints, alpha, and aspect thresholds on inner validation. Freeze
   them before outer inference. Do not reuse full-fit model checkpoints.
6. Pool outer predictions and compute all shared metrics, fold deltas, selected
   alphas, and paired group-bootstrap intervals. Add corrected joint DeBERTa
   only as an explicitly separate five-fit extension if its pilot warrants it.

## Interpretation and next action

Apply the shared advancement gate: pooled pair gain at least 0.005, wins in four
of five folds, and no neutral/conflict F1 loss above 0.02. Passing this gate
justifies seed confirmation; it is not itself a definitive win.

Large fold variation indicates composition sensitivity or sparse-class effects.
An alpha frequently equal to zero suggests unreliable ensemble value. A pooled
gain driven by one fold should not advance under this rule.

## Verification and deliverables

Test complete outer coverage, group disjointness, external duplicate exclusions,
and checkpoint provenance. Verify that changing outer labels cannot change
training or selected alpha/thresholds. Save per-fold predictions and selections,
pooled metrics, group-bootstrap results, and a fit-count/runtime ledger.

