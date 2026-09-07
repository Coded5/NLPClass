# Contest 1: Hint-Constrained Language Modeling

This project predicts a complete next word from a text context and a literal
first-character hint. The strongest system uses step-258k GPT-2 candidate
generation, frozen MiniLM embeddings, listwise MLP rerankers, and calibrated
prediction priors from the earlier full-word rankers.

## Result

The final tuned grouped-OOF development result on 65,619 non-contaminated
alphanumeric-hint examples is **63.3368% top-1 accuracy**. Assuming all 7,903
non-alphanumeric hints are correct gives an aggregate of **67.2778%** over
73,522 examples.

These are post-selection development results, not an untouched holdout
estimate. See [`reports/neural-reranker.md`](reports/neural-reranker.md) and
[`LOGS.md`](LOGS.md) for the full methodology and chronology.

## Layout

```text
src/contest1/   Python package and command-line tools
tests/          Model-download-free and integration tests
notebooks/      Exploratory and probability-analysis notebooks
reports/        Small, commit-safe result summaries
data/           Local datasets plus a tracked hash manifest
train/          Canonical punctuation-preserving training corpus
artifacts/      Generated checkpoints, caches, predictions, and reports
outputs/        Generated submission/model outputs
```

Large datasets and generated artifacts remain local and are intentionally
ignored by Git. Their identities are recorded in `data/manifest.sha256` and in
artifact provenance files.

## Setup

Use Python 3.11 and select the CUDA-enabled PyTorch wheel appropriate for the
host:

```bash
python -m pip install -e .
# Optional MLflow support:
python -m pip install -e '.[tracking]'
```

The existing uv-managed environment can run the project without installation
by prefixing commands with `PYTHONPATH=src`.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Main commands

```bash
contest1-train-gpt2 --help
contest1-comprehensive-eval --help
contest1-rerank --help
contest1-neural-rerank --help
contest1-pipeline --help
```

Equivalent module invocation:

```bash
python -m contest1.neural_rerank_experiment --help
```

In an editable installation, default paths resolve from the project root rather
than the caller's working directory. Set `CONTEST1_PROJECT_ROOT` to the checkout
or external data root when using a non-editable installation.

## Data policy

- `train/train.src.tok` is the corpus used by the final GPT-2 and reranker
  artifacts.
- `data/train.src.tok` is a differently normalized corpus retained for the
  original notebook. It is not interchangeable with the final corpus.
- `data/devv_eval.csv` is the allowed development file.
- **Do not inspect or evaluate against `data/devv_test.csv`.** Evaluation code
  rejects its filename, canonical path, symlink target, and recorded content
  fingerprint.
- `data/test_set_no_answer.csv` is the answer-free contest input.
