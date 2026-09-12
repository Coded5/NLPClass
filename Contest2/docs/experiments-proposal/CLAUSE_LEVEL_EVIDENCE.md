# Experiment 9: Clause-Level Aspect-Conditioned Evidence

## Scope

Test a polarity architecture that aggregates evidence from clauses while keeping
aspect predictions fixed. This is a new representation experiment, distinct from
the earlier evidence model that processed the whole sentence.

## What this finds out

Can the model retain positive and negative evidence about the same aspect across
contrastive clauses, instead of allowing one sentiment to dominate? Can it avoid
false conflict when the opposing sentiments concern different aspects?

## How to run

1. Use all-class fitting data and grouped CV. Freeze a deterministic segmentation
   rule before training: punctuation and contrast markers such as but, however,
   though, and although. Retain conjunctions, offsets, and original sentence order;
   use the whole sentence when no split applies.
2. Encode each clause with the candidate aspect using a shared DeBERTa encoder.
   Produce positive/negative evidence probabilities per clause.
3. Aggregate each evidence channel by maximum over clauses. Convert the two
   aggregate channels to positive-only, negative-only, both, and neither scores,
   normalized over the four polarities.
4. Train from sentence/aspect labels: positive=(1,0), negative=(0,1),
   conflict=(1,1), neutral=(0,0). These are an operational evidence assumption;
   do not invent gold clause labels.
5. Compare against a whole-sentence two-evidence-head control with the same
   encoder/loss, and the standard four-class DeBERTa control. Use the same data,
   splits, seed, and stopping rules. Start with one seed, then CV only if useful.
6. Evaluate contrastive sentences, same-aspect opposition, cross-aspect opposition,
   negation, and no-contrast examples. Define diagnostic subsets before examining
   candidate outcomes; use manual aspect-scope tags only for evaluation.

## Interpretation and next action

Conflict gains with stable overall pair F1 support the evidence hypothesis.
False conflict increases on cross-aspect sentences indicate weak target binding.
Gains only on the hand-selected contrast subset do not establish a general win.

Clause scores are weakly supervised diagnostics, not validated explanations.
If segmentation or missing context causes regressions, audit these before adding
complexity. Advance only under the shared CV gate.

## Verification and deliverables

Test negation preservation, no-clause fallback, aspect conditioning, output
normalization, and absent human clause labels. Save clause offsets/evidence,
subset metrics, false-conflict rates, errors, and comparisons with both controls.

