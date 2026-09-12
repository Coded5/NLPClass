# Experiment 10: Aspect Backbone Comparison and Ensemble

## Scope

Improve multilabel aspect prediction after the polarity recipe is stable.
Compare RoBERTa, DeBERTa-v3, and ModernBERT aspect encoders while freezing the
same fold-specific polarity predictions for every candidate.

## What this finds out

Has aspect detection become the main limit on pair F1? Can complementary
aspect probabilities recover missed secondary aspects without excessive false
positives, particularly for anecdotes/miscellaneous?

## How to run

1. Quantify the gap between predicted-aspect and gold-aspect polarity performance
   using existing outer predictions. Use the gap as a diagnostic, not as a
   deployable oracle result.
2. Train each aspect backbone on original contest fitting rows with complete
   five-label multihot targets. Use the existing RoBERTa multilabel loss,
   positive-weight policy, and training recipe consistently; record adjustments.
   Keep external aspect augmentation out of this first comparison.
3. Start with seed 42 on a pilot selection split, then advance the best challenger
   into five-fold CV against RoBERTa.
4. Compare standalone models and probability blends
   `p = (1 - alpha) * p_roberta + alpha * p_candidate`, using the five-value grid.
   Tune five aspect thresholds on inner validation pair F1 with fixed polarity.
   Retain the current fallback selecting the highest-probability aspect.
5. Report aspect precision/recall/F1 per category, pair F1, exact-set accuracy,
   predicted aspect counts, and single-versus-multi-aspect performance.

## Interpretation and next action

Higher aspect F1 without higher pair F1 can mean newly emitted aspects receive
incorrect polarity. Improved recall with reduced exact-set accuracy can mean
overprediction. Judge the complete pipeline as well as the aspect component.

If the gain concentrates on one ambiguous category and varies across folds,
require further confirmation. Advance under the shared gate and then assess
whether the added encoder's inference cost is justified.

## Verification and deliverables

Test multi-aspect targets, identical-text group isolation, threshold selection,
fallback behavior, and frozen polarity alignment. Save category-level error
audits and the exact selected aspect ensemble recipe.

