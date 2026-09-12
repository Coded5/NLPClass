# Experiment 7: Moderate External-Data Sampling

## Scope

Train separate DeBERTa polarity models retaining all four SemEval polarities.
Change only how often external neutral/conflict rows are sampled. Use this
follow-up if the existing experiments show a useful minority-versus-majority
tradeoff. Do not change aspect training in this experiment.

## What this finds out

Can moderate minority emphasis improve neutral/conflict while preserving the
overall gains from external positive/negative examples? This tests the middle
ground between all-class and minority-only augmentation.

## How to run

1. Use the frozen grouped folds and full cleaned SemEval additions in fitting
   partitions. Start with seed 42.
2. Assign sampling weight 1 to all original contest rows and external
   positive/negative rows. Assign weight k to external neutral/conflict rows,
   with k in {1, 2, 4}.
3. Sample with replacement for exactly the natural fitting row count each epoch.
   Apply this sampler to k=1 as well, creating a matched sampling baseline.
   Include the existing shuffle baseline as context, clearly labeled as a
   different sampling process.
4. Compute weighted-CE class weights once from the natural all-class fitting
   rows. Freeze them across k to isolate sampling from loss-weight changes.
5. Keep optimizer-step budget, seed policy, effective batch size, and validation
   schedule equal. Log realized source/class sample counts.
6. Choose k and decoding on inner validation only; evaluate on the outer fold.
   Keep the same fold-specific aspect model across conditions.

## Interpretation and next action

An interior k improving pair F1 while retaining minority gains supports moderate
sampling. Higher minority recall with worsening precision or majority F1 means
the tradeoff remains unresolved. If k=1 wins, extra sampling is unnecessary.

Use the shared CV advancement gate before seed confirmation. Selection among
several k values must happen inside each fold; reporting the best outer score
as if k were fixed would overstate improvement.

## Verification and deliverables

Test reproducible sampling, unchanged class weights and update counts, and no
sampling from heldout partitions. Save realized exposure counts, k selections,
class confusion matrices, and paired deltas against the matched k=1 control.

