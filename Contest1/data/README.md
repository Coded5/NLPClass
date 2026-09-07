# Local data

The data files are intentionally excluded from Git because they are large or
answer-bearing. `manifest.sha256` records the exact local files used during the
experiments.

| Path | Purpose |
|---|---|
| `../train/train.src.tok` | Canonical punctuation-preserving corpus used by the final models |
| `train.src.tok` | Differently normalized corpus used by the original notebook |
| `devv_eval.csv` | Allowed development evaluation set |
| `devv_test.csv` | Forbidden answer-bearing test partition; never use for evaluation |
| `test_set_no_answer.csv` | Contest input without answers |
| `legacy/dev_set.csv` | Earlier punctuation-preserving combined development CSV |
| `legacy/test_set_no_answer.csv` | Earlier punctuation-preserving test input |

The two training corpora and the two test-input variants are not byte-identical
and must not be substituted without rebuilding dependent artifacts.
