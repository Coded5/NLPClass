# Experiment 11: Disagreement-Driven Data Acquisition

## Scope

Acquire and independently annotate additional restaurant reviews after the model
comparison stabilizes. Compare targeted selection with random selection from the
same new source. Execution requires an available permitted text pool and human
annotation capacity.

## What this finds out

Does labeling examples where strong models disagree improve performance more
than labeling an equal number of randomly chosen reviews? Are unresolved errors
caused by missing examples, ambiguous task definitions, or limited context?

## How to run

1. Freeze the current models and evaluation groups. Obtain a new unlabeled review
   pool with suitable usage rights and remove overlaps with all evaluation data.
   Keep any fresh final-test pool separate from acquisition.
2. Create a short annotation rubric covering complete aspect sets, target-specific
   sentiment, conflict, neutral, and insufficient context. Use two independent
   annotators and adjudication for disagreements.
3. Allocate 200 review-level annotation slots: 100 selected by ensemble
   disagreement and 100 randomly sampled from the remaining same-source pool.
   Annotate all aspects in every review, not only the model's proposed aspect.
4. Rank targeted examples by disagreement between aligned polarity distributions
   over aspects detected by at least one model. Stratify by predicted aspect
   and cap near-duplicate text clusters; freeze the ranking rule in advance.
5. Keep annotators blind to predictions and selection arm where practical.
   Record confidence and unresolved cases; never use assistant guesses as gold.
6. Train matched baseline-plus-targeted and baseline-plus-random models with equal
   seed schedules and fixed hyperparameters. Score on unchanged evaluation groups.

## Interpretation and next action

A larger targeted-data gain supports another acquisition round. Equal gains
suggest general coverage matters more than disagreement targeting. Worse targeted
performance may reflect unrepresentative difficult cases or annotation uncertainty.

Report agreement, adjudication rate, cost per review, source/class distributions,
and model gains. Annotator disagreement does not automatically establish that
existing dataset labels are wrong. Keep unresolved labels out of training.

## Verification and deliverables

Save acquisition manifests, overlap checks, blinded worksheets, adjudication
records, and matched exposure counts. Do not silently replace human review with
synthetic labeling if annotation capacity is unavailable; leave this experiment
pending and pursue independent work.

