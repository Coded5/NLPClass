# Grouped polarity oversampling cross-validation

Five grouped outer folds estimate gold-aspect polarity performance. Every outer
training set contains its own grouped checkpoint-selection split.

| Configuration | Accuracy | Macro F1 | Neutral F1 | Conflict F1 |
|---|---:|---:|---:|---:|
| reference | 0.806338 | 0.689644 | 0.656425 | 0.428571 |
| loss-control | 0.820775 | 0.685535 | 0.666667 | 0.384365 |
| oversampling | 0.811268 | 0.686531 | 0.653555 | 0.425287 |
| sampling-corrected | 0.804577 | 0.686042 | 0.662651 | 0.413965 |

Promotion gate winner: `reference`.

The historical test split was not used by this experiment.
