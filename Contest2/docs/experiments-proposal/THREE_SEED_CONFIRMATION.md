# Experiment 6: Three-Seed Confirmation

## Scope

Repeat only the baseline and strongest recipe that passed grouped CV. Retain
the same fold assignments and candidate definition; add training seeds 17 and
73 to existing seed-42 results.

## What this finds out

Is the gain stable across initialization and minibatch order? Does it persist
after averaging multiple seeds, or was the initial benefit tied to one fit?

## How to run

1. Freeze the winning recipe and candidate grid before revealing added-seed outer
   scores. Reuse seed-42 artifacts only after validating their manifests.
2. For each fold and additional seed, train the matched RoBERTa aspect component,
   baseline DeBERTa polarity, and the winning additional polarity component.
   For a joint finalist, train its joint model alongside the matched baseline.
   This normally adds 30 fits for two additional seeds across five folds.
3. Select checkpoints and decoding on inner validation for each seed. Evaluate
   paired baseline/candidate outer predictions.
4. Also construct a three-seed ensemble within each fold: average polarity logits
   within each backbone, average aspect probabilities for separate aspect models,
   then select interpolation and thresholds on inner validation.
5. Report each seed's pooled delta, each fold's ensemble delta, aggregate class
   metrics, runtime, and inference cost. Bootstrap review groups for paired
   prediction uncertainty and report training-seed variation separately.

## Interpretation and next action

Require the shared CV gate for the three-seed ensemble and a positive pooled
pair delta in at least two of three individual seeds. Call the result supported
within this development protocol only if its paired ensemble interval excludes
zero; otherwise call it promising but uncertain.

If averaging seeds removes the benefit, the extra backbone may be compensating
for instability rather than adding reliable complementary information. Prefer
the simpler baseline when gains are negligible relative to inference cost.

## Verification and deliverables

Check matching folds, seeds, and aspect components; equal weighting within each
backbone; and no checkpoint trained on an outer group. Save ensemble composition,
per-seed and ensemble metrics, bootstrap intervals, and a fixed final recipe.
This stage does not authorize treating historical test as a fresh evaluation.

