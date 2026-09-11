# RoBERTa ensemble evaluation

## Scope

- Evaluated rows: 631
- Unique review IDs: 516
- Dataset: 20% heldout union (10% validation + 10% untouched test)
- Ensemble: aspect prediction from the aspect model plus polarity prediction from the polarity model

The validation half was used to select the best checkpoint, so the combined 20% score has selection bias; only the test half is strictly untouched. The single-label architecture also cannot recover every pair for multi-aspect reviews.

## Selected checkpoints

| Task | Run | Best epoch | Validation micro-F1 |
|---|---|---:|---:|
| aspect | `artifacts/training/roberta-aspect-exp1` | 10 | 0.792321 |
| polarity | `artifacts/training/roberta-polarity-exp2` | 17 | 0.849624 |

## Evaluation results

| Target | Precision | Recall | F1 | Exact-set accuracy |
|---|---:|---:|---:|---:|
| aspect | 0.893411 | 0.730586 | 0.803836 | 0.703488 |
| polarity | 0.850775 | 0.805505 | 0.827521 | 0.800388 |
| overall pair | 0.751938 | 0.614897 | 0.676548 | 0.593023 |

## Output files

- `predictions.csv`: combined row-level predictions
- `gold.csv`: evaluation copy of the labeled input
- `metrics.json`: structured metrics
- `official_evaluation.txt`: output from the supplied evaluator
- `selection.json`: checkpoint selection provenance
