# Experiment 8: Temperature-Scaled Logit Interpolation

## Scope

Calibrate a fixed two-component polarity ensemble using saved fold logits.
No encoder retraining is required. Keep model checkpoints and candidate pool fixed.

## What this finds out

Does one model dominate raw interpolation because its logits have a larger
scale? Can better-calibrated probabilities improve blending or confidence quality?

## How to run

1. Use the baseline and selected component logits from the CV or seed-confirmation
   experiments. Each fold supplies inner-selection and outer-scoring logits.
2. Compare raw interpolation against temperature-scaled interpolation:
   `z = (1 - alpha) * z_A / T_A + alpha * z_B / T_B`.
3. Fit one positive temperature per component by minimizing unweighted
   gold-aspect negative log-likelihood on inner selection only. Optimize log
   temperature within T in [0.25, 4], with a fixed solver and seed.
4. Fix those temperatures, then choose alpha from {0, 0.25, 0.5, 0.75, 1}
   and aspect thresholds using inner-selection pair F1. Apply the same
   alpha/threshold search to the raw control.
5. Evaluate locked choices on outer folds. Report pair F1, gold-aspect NLL,
   Brier score, a fixed 10-bin confidence calibration error, and class metrics.

## Interpretation and next action

Positive scalar temperature cannot change a standalone model's argmax; it can
change ensemble predictions. For two unconstrained positive temperatures and
continuous alpha, relative scaling largely reparameterizes the blend weight.
Therefore do not claim new predictive information or architecture diversity.

Better NLL with unchanged F1 can still be useful for confidence reporting.
A raw-versus-scaled F1 gain may reflect better effective weighting on the coarse
grid; report effective relative weights. Reject calibration if outer NLL worsens
or improvements disappear across folds.

## Verification and deliverables

Test T=1 equivalence, positive bounds, standalone argmax invariance, and that
outer labels never enter fitting. Save temperatures, raw and effective alphas,
proper scoring metrics, reliability-bin data, and paired pair-F1 differences.
Use the shared gate for an accuracy promotion claim.

