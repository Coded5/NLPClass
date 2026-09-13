# Current Top Models

Last updated: 2026-09-13.

The first F1-focused roadmap run was invalidated by a tokenizer mismatch. The
corrected attempt reproduced the baseline exactly and completed its seed-42
screen. Its frozen three-seed confirmation succeeded: fold 1 uses no class
weighting and folds 2–5 use exponent `0.5`.
The subsequent seed-42 DeBERTa-v3-large LoRA pilot produced the highest pooled
OOF point estimate and completed a locked historical-test evaluation.
Distillation was stopped. See
[the protocol](experiments-proposal/F1_OVERNIGHT_ROADMAP.md). Model rankings
below include only complete evaluations at their stated evaluation level.

The primary metric is overall micro-F1 over complete
`(id, aspectCategory, polarity)` pairs. `Exact set` is per-review accuracy: a
review is correct only when its complete predicted pair set equals gold. The
polarity F1 column is the official evaluator's polarity micro-F1; `Gold-pol.
accuracy` evaluates polarity only on supplied gold aspects.

## Historical-test ranking

| Rank | Model recipe | Pair F1 | Aspect F1 | Polarity F1 | Exact set | Gold-pol. accuracy |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Three-seed RoBERTa multilabel aspect + three-seed all-class SemEval DeBERTa polarity | **0.7744** | **0.8736** | 0.8318 | 0.6822 | 0.8544 |
| 2 | Five-fold RoBERTa aspect + seed-42 DeBERTa-v3-large LoRA polarity committee | 0.7727 | 0.8734 | **0.8561** | 0.6899 | **0.8576** |
| 3 | Three-seed RoBERTa aspect + equal RoBERTa/all-class SemEval DeBERTa polarity blend | 0.7680 | **0.8736** | 0.8349 | **0.6938** | 0.8449 |
| 4 | Validation-locked 50/50 RoBERTa-DeBERTa aspect blend + all-class SemEval DeBERTa polarity | 0.7660 | 0.8674 | 0.8296 | 0.6783 | — |
| 5 | Three-seed RoBERTa aspect + equal original RoBERTa/DeBERTa polarity blend | 0.7630 | 0.8734 | 0.8231 | 0.6860 | 0.8449 |
| 6 | Three-seed RoBERTa aspect + three-seed original-data DeBERTa polarity | 0.7616 | **0.8736** | 0.8229 | 0.6860 | 0.8418 |

The `0.7744` system is the highest score observed on the repeatedly inspected
historical test. It was not the validation-selected candidate in that
experiment, and its bootstrap interval against the original DeBERTa system
included zero. Treat it as the leading deployable candidate, not a fresh
unbiased performance claim. The LoRA committee came within `0.0017` pair F1
and improved polarity F1 by `0.0243`, but it uses five fold-specific large
models rather than one full-data deployable checkpoint.

## Grouped-CV ranking

| Rank | Model recipe | Pooled OOF pair F1 | Aspect F1 | Polarity F1 | Exact set |
|---:|---|---:|---:|---:|---:|
| 1 | Seed-42 DeBERTa-v3-large LoRA polarity + frozen three-seed RoBERTa aspects | **0.7800** | 0.8893 | **0.8476** | **0.7050** |
| 2 | Frozen fold-specific weaker-weight DeBERTa, three seeds + three-seed RoBERTa aspect | 0.7761 | **0.8904** | 0.8418 | 0.6954 |
| 3 | Three-seed RoBERTa aspect + three-seed all-class SemEval DeBERTa polarity | 0.7667 | **0.8904** | 0.8356 | 0.6934 |
| 4 | Three-seed RoBERTa aspect + DeBERTa/ELECTRA polarity blend | 0.7637 | 0.8902 | 0.8356 | 0.6920 |
| 5 | Temperature-scaled DeBERTa/ELECTRA polarity blend | 0.7632 | **0.8904** | 0.8344 | 0.6915 |

These are pooled predictions from fold-specific models, not single deployable
checkpoints. LoRA has the highest point estimate, `+0.0039` above the confirmed
weaker-weight system, but it is only a seed-42 pilot and has no paired interval
against that immediate incumbent. The weaker-weight system remains the
confirmed multi-seed result: it improved the prior leader by `+0.0094`, won four
of five folds, and passed its predeclared group-bootstrap gate with interval
`[+0.0008, +0.0179]`.

### Confirmed class-weight experiment

Reducing DeBERTa polarity class-weight strength to exponent `0.5` scored
`0.7646` seed-42 pooled OOF pair F1 versus `0.7419` for the matched seed-42
control: `+0.0227`, five of five fold wins, with group-bootstrap interval
`[+0.0120, +0.0335]`. The frozen confirmation with seeds 17 and 73 reached
`0.7761` versus the three-seed baseline at `0.7667`. Pair-class conflict F1
improved by `0.0242`, while neutral fell by `0.0057`. No-weight CE scored
`0.7610` in the seed-42 screen and was retained only for fold 1 by inner
selection.

The three-seed uncertainty-aware decoder reached `0.7672`, only `+0.0005` over
the baseline with interval `[-0.0033, +0.0042]`; treat it as a tie.

For grouped-CV development comparisons, LoRA is the point-estimate leader and
the weaker-weight recipe is the confirmed multi-seed leader. For a currently
deployable full-data model, use the three-seed RoBERTa aspect ensemble with the
three-seed all-class SemEval DeBERTa polarity ensemble.
Do not include the
RoBERTa polarity blend merely because it was selected on the old validation
split: it reduced historical-test pair F1 from `0.7744` to `0.7680`.

## Validation-only ranking

These results guided experiments but are not interchangeable with OOF or
historical-test scores.

| Candidate | Validation pair F1 | Decision |
|---|---:|---|
| Tuned full-combined joint DeBERTa decoder | **0.8019** | Rejected: fell to 0.7437 on historical test |
| Standard seed-42 all-class DeBERTa polarity with frozen RoBERTa aspects | 0.7950 | Strong control; one seed only |
| 50/50 RoBERTa-DeBERTa aspect probability blend with frozen polarity | 0.7923 | Rejected: fell to 0.7660 on historical test |
| Seed-42 DeBERTa/ELECTRA pilot blend | 0.7919 | Rejected after three-seed confirmation |
| Three-seed all-class SemEval DeBERTa polarity | 0.7863 | Retained |
| Clause-level evidence | 0.7702 | Rejected: below standard control and worse conflict F1 |

## Rejected ensemble components

- ELECTRA improved the seed-42 grouped-CV screen, but the three-seed blend fell
  from `0.7667` to `0.7637` and won only one of five folds.
- Temperature scaling reduced that blend again, from `0.7637` to `0.7632`.
- ModernBERT polarity scored `0.7395` alone and selected zero interpolation
  weight with DeBERTa.
- Minority-only SemEval augmentation improved neutral/conflict balance but did
  not improve overall pair F1.
- Whole-sentence and clause-level evidence heads scored `0.7747` and `0.7702`
  against their matched standard DeBERTa control at `0.7950`.
- ModernBERT aspect was weaker alone (`0.7578` pair F1). Its best blend reached
  `0.7829`, below the inference-only RoBERTa-DeBERTa aspect blend at `0.7923`.
- The 50/50 RoBERTa-DeBERTa aspect blend's validation gain did not generalize:
  historical-test pair F1 was `0.7660`, below the incumbent `0.7744`, while
  aspect micro-F1 fell from `0.8736` to `0.8674`.

## Artifact pointers

- Historical augmented comparison:
  `artifacts/experiments/semeval14-deberta-polarity-3seed-v1/`
- Grouped three-seed confirmation:
  `artifacts/experiments/electra-deberta-three-seed-confirmation-v1/`
- Previous locked winner:
  `artifacts/experiments/deberta-v3-three-systems-v1/`
- Aspect-backbone probability blend:
  `artifacts/experiments/aspect-backbone-seed42-v1/`
- Corrected F1-focused screen:
  `artifacts/experiments/f1-roadmap-v2/`
- DeBERTa-v3-large LoRA CV and locked test evaluation:
  `artifacts/experiments/deberta-v3-large-lora-seed42-v1/`
- Full experiment narrative: `docs/SUMMARY.md`
- Exhaustive earlier pair-F1 table: `docs/OVERALL_F1_COMPARISON.md`

The aspect blend was evaluated with its validation-selected weight and
thresholds locked. It is excluded from the recommended ranking because it did
not improve the historical-test incumbent and has no grouped-CV confirmation.
