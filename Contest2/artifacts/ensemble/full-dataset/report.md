# RoBERTa ensemble full-dataset evaluation

## Scope

- Evaluated rows: 3156
- Unique review IDs: 2584
- Dataset: complete labeled training dataset (in-sample)
- Ensemble: aspect prediction from the aspect model plus polarity prediction from the polarity model

This is a training-set diagnostic, not an unbiased generalization estimate. The models are single-label classifiers; repeated text for multi-aspect reviews therefore receives the same prediction and cannot recover every distinct gold pair.

## Selected checkpoints

| Task | Run | Best epoch | Validation micro-F1 |
|---|---|---:|---:|
| aspect | `artifacts/training/roberta-aspect-exp1` | 10 | 0.792321 |
| polarity | `artifacts/training/roberta-polarity-exp2` | 17 | 0.849624 |

## Full-dataset results

| Target | Precision | Recall | F1 | Exact-set accuracy |
|---|---:|---:|---:|---:|
| aspect | 0.976006 | 0.799620 | 0.879052 | 0.787539 |
| polarity | 0.970201 | 0.918652 | 0.943723 | 0.916796 |
| overall pair | 0.927245 | 0.759670 | 0.835134 | 0.765480 |

## Output files

- `predictions.csv`: combined row-level predictions
- `gold.csv`: evaluation copy of the labeled input
- `metrics.json`: structured metrics
- `official_evaluation.txt`: output from the supplied evaluator
- `selection.json`: checkpoint selection provenance
