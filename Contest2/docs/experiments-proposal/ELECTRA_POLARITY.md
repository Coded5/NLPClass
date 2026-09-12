# Experiment 4: ELECTRA Polarity and Interpolation

## Scope

Fine-tune `google/electra-base-discriminator` with an aspect-conditioned four-class
classification head, all-class SemEval fitting data, and seed 42. Use the same
aspect component and pilot selection partition as ModernBERT.

## What this finds out

Can ELECTRA improve standalone polarity or provide complementary predictions
for DeBERTa? This is supervised fine-tuning of an existing discriminator encoder,
not training a new generator or performing replaced-token pretraining.

DeBERTa-v3 already uses ELECTRA-style pretraining. An ELECTRA result therefore
cannot by itself establish that replaced-token detection caused an improvement.

## How to run

1. Load the discriminator encoder through the sequence-classification adapter.
   Confirm that each review/aspect pair produces exactly four polarity logits.
2. Pin revisions and run finite forward/backward, optimizer, and checkpoint
   smoke checks. Benchmark ten minutes under the same length and effective batch
   settings used for the ModernBERT pilot.
3. Train weighted cross-entropy using the shared defaults. Use fitting data only
   for class weights, and validation only for stopping and threshold selection.
4. Compare ELECTRA alone with all-class DeBERTa seed 42 and ModernBERT seed 42.
   Record architecture-dependent runtime and parameter differences.
5. Blend ELECTRA with the existing DeBERTa ensemble using the five-value alpha
   grid. Retain endpoint scores, fixed-threshold diagnostics, retuned pair F1,
   and classwise corrections versus regressions.

## Interpretation and next action

Promote based on ensemble utility as well as standalone performance. If ELECTRA
beats ModernBERT's best validation blend, choose ELECTRA for the new-backbone
CV arm. Break ties by standalone pair F1 and then measured inference cost.

A failure under one fixed recipe does not prove ELECTRA cannot work; it means
this controlled pilot offers insufficient reason to allocate more compute.
Avoid selecting a candidate from historical-test outcomes.

## Verification and deliverables

Check discriminator versus token-discrimination head loading, explicit polarity
ordering, input-pair handling, and reproducible caching. Save a comparison table
covering validation pair F1, gold-aspect class metrics, runtime, and blend alpha.

References: [ELECTRA model card](https://huggingface.co/google/electra-base-discriminator)
and [DeBERTa-v3 model card](https://huggingface.co/microsoft/deberta-v3-base).

