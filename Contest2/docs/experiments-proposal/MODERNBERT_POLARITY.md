# Experiment 3: ModernBERT Polarity and Interpolation

## Scope

Fine-tune `answerdotai/ModernBERT-base` for aspect-conditioned four-class polarity.
Use all-class SemEval augmentation, seed 42, and the existing fixed validation
partition. Keep the RoBERTa aspect component fixed for the pilot.

## What this finds out

Does ModernBERT produce useful different errors from DeBERTa? Its standalone
score and its value in an ensemble are separate questions. This tests the
pretrained model package, including tokenizer and pretraining, rather than
isolating the architecture alone.

## How to run

1. Pin the model/tokenizer revision and implement the sequence-classification
   adapter. Encode review text paired with the candidate aspect. Respect the
   model's supported input fields and verify pooling/classification behavior.
2. Run a training and inference smoke check, then measure ten minutes of
   throughput and peak memory. Reduce physical batch size with compensating
   accumulation if needed to retain effective batch 16.
3. Train with the shared polarity defaults and all-class fitting data. Select the
   checkpoint using validation pair F1 with the common aspect model.
4. Compare standalone ModernBERT against all-class DeBERTa seed 42 for a matched
   seed comparison. Also evaluate its practical blend with the existing
   three-seed DeBERTa ensemble, explicitly recording that unequal model count.
5. Sweep alpha in {0, 0.25, 0.5, 0.75, 1}, where alpha weights ModernBERT, and
   select validation composition as in the interpolation proposal.

## Interpretation and next action

A weaker standalone model can still advance if its blend improves pair F1.
A strong standalone score with no blending benefit suggests similar error
patterns. If both standalone and blended scores regress, do not immediately
expand to multiple seeds or a large hyperparameter search.

Compare the best blend against the ELECTRA pilot using validation pair F1;
break ties by standalone pair F1 and then inference cost. The selected candidate
enters CV as a freshly trained model in each fold.

## Verification and deliverables

Verify text/aspect encoding, output label order, finite optimizer updates,
checkpoint reload, and pair coverage. Save standalone and blended metrics,
classwise corrections/regressions, memory, throughput, and inference latency.

Reference: [ModernBERT model card](https://huggingface.co/answerdotai/ModernBERT-base).

