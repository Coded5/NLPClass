# Experiment 1: Corrected Joint DeBERTa

## Scope

Run a validation-only seed-42 pilot with a shared DeBERTa-v3-base encoder,
aspect-presence head, and four-class polarity head. Use the current minority
augmentation condition. Do not interpret the earlier pilot as a clean
architecture comparison until its supervision has been corrected.

## What this finds out

Does learning aspect presence and polarity together improve complete pair
prediction? Can a shared representation help conflict and neutral without
damaging negative classification or aspect recall?

The inspected minority data omitted 32 known present aspects across 30 retained
SemEval reviews. The existing joint builder interprets omitted aspects as absent.
Correcting this is a prerequisite, not an experimental accuracy improvement.

## How to run

1. Join retained external IDs to the cleaned, unfiltered SemEval annotations.
   Recover the complete aspect-presence set for each retained review.
2. Represent aspect presence and polarity supervision independently. Original
   contest rows receive their original full supervision. External neutral/conflict
   aspects receive both losses. External positive/negative aspects receive
   positive aspect-presence supervision but no polarity loss. Truly absent
   aspects receive negative aspect supervision only.
3. Calculate aspect weights from complete presence labels and polarity weights
   from supervised polarity targets. A batch with no supervised polarity targets
   must produce a finite zero polarity loss.
4. Use weighted focal polarity loss, gamma 2, plus the existing weighted binary
   aspect loss. Retain the shared training defaults and seed 42.
5. Train in a fresh directory and choose the checkpoint and thresholds on
   validation pair F1. Compare against the minority separate DeBERTa seed-42
   pipeline on exactly the same validation reviews.

## Interpretation and next action

Report aspect, pair, exact-set, gold-aspect polarity, and pair-level class metrics.
A better polarity score with worse pair F1 suggests aspect detection is limiting
the joint system. A minority gain accompanied by a negative-class regression is
a tradeoff, not an automatic promotion.

Advance to an optional five-fold arm if pair F1 exceeds the matched separate
pilot and negative gold-aspect F1 falls by no more than 0.02. A single pilot win
is preliminary. Joint RoBERTa's original-data result is historical context, not
a controlled backbone comparison because its data differ.

## Verification and deliverables

Test missing-but-present aspects, absent candidates, no-supervision batches,
masked gradients, complete duplicate-text annotation sets, and checkpoint reload.
Save the repaired supervision manifest and label-repair audit alongside the
validation report. No real historical-test file should be opened by this pilot.

