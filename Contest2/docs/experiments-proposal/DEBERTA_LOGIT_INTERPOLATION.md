# Experiment 2: Existing DeBERTa Logit Interpolation

## Scope

Perform inference and decoding with existing all-class and minority-only SemEval
DeBERTa three-seed polarity checkpoints. Use the same frozen RoBERTa aspect
predictions. No new model fitting is needed for this exploratory pilot.

## What this finds out

Can the minority model's neutral/conflict strengths complement the broader
accuracy of the all-class model? Does a partial mixture improve on both standalone
components, rather than simply reproducing the stronger endpoint?

## How to run

1. Verify identical validation reviews and complete label alignment. Cache logits
   for all five aspect candidates per review for each checkpoint.
2. Average seed logits equally within each training condition.
3. Evaluate `z = (1 - alpha) * z_all + alpha * z_minority` for alpha values
   0, 0.25, 0.5, 0.75, and 1. Apply argmax after interpolation.
4. First compare these alphas with the all-class baseline's frozen aspect
   thresholds to isolate polarity changes.
5. Then tune aspect thresholds separately for each alpha on validation, using
   the existing search. Select by pair F1; exact ties prefer smaller alpha.
   Report both the fixed-threshold diagnostic and retuned composition.
6. Save per-class error corrections and newly introduced errors, including
   baseline wrong/candidate correct and the reverse on gold aspects.

## Interpretation and next action

An interior alpha beating both endpoints suggests useful complementarity.
Selecting alpha zero means the minority component adds no measured value;
selecting one favors the specialist alone. Disagreement by itself is not enough:
a component must correct errors without introducing too many new ones.

Treat gains on the repeatedly used validation set as screening evidence. Carry
the recipe, alpha grid, and selection rule into grouped CV; do not choose a
weight from historical-test results. Stop this branch if no blend helps and
error overlap shows little corrective value.

## Verification and deliverables

Test alpha endpoints, class ordering, missing/duplicate keys, and invariance to
record ordering. Save cached logits, the complete alpha sweep, selected thresholds,
error-overlap tables, and timing. Include standalone endpoints in every report.

