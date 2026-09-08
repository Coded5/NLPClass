# Zero-shot Aspect-based Sentiment Classifier

This project classifies restaurant reviews into one or more `aspectCategory` and `polarity` pairs with a zero-shot LLM. It defaults to the local Ollama model `qwen2.5:7b`, checkpoints every review, and can resume after model, network, or process failures.

The workflow only splits `data/contest2_train.csv`. It does not read, split, or modify `data/contest2_test.csv`.

## Labels

- Aspects: `food`, `price`, `service`, `ambience`, `anecdotes/miscellaneous`
- Polarities: `positive`, `negative`, `neutral`, `conflict`

## Setup

Python 3.10 or newer is required. The pandas dependency is used by the official `scripts/evaluate.py` evaluator.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For the default local model:

```bash
ollama pull qwen2.5:7b
ollama serve
```

## 1. Create the 90:10 split

```bash
zeroshot-classifier split
```

Rows sharing an ID are kept together to prevent the same review leaking across partitions. IDs are stratified by their complete set of `(aspectCategory, polarity)` labels with seed 42. Because one ID can own multiple rows, the row ratio is as close to 90:10 as group boundaries permit.

Generated files are written outside `data/`:

```text
artifacts/splits/train.csv
artifacts/splits/test.csv
artifacts/splits/manifest.json
```

Use `--overwrite` to intentionally regenerate an existing split.

## 2. Run classification on the complete dataset

```bash
zeroshot-classifier run
```

By default, this classifies every unique ID in `data/contest2_train.csv`. The file currently contains 3,156 annotation rows across 2,584 unique review IDs. Each review is sent once and the model may return several aspect/polarity pairs.

Defaults:

- Backend: `ollama`
- Model: `qwen2.5:7b`
- Ollama URL: `http://localhost:11434`
- Temperature: `0`
- Attempts per item: `3`

Only review text is sent to the model. Gold labels are never included in the prompt, so classification remains zero-shot even though the complete labeled dataset is used for evaluation.

Progress is committed to `artifacts/runs/default/checkpoint.sqlite3`. Running the same command again skips completed IDs. A run refuses to resume if its input, model, backend, URL, temperature, or prompt changed; use a different `--run-dir` in that case.

After repairing an unavailable model or persistent response error, retry terminal failures with:

```bash
zeroshot-classifier run --retry-failed
```

The current valid predictions are atomically exported after every run or interruption:

```text
artifacts/runs/default/predictions.csv
```

### Different Ollama model

```bash
zeroshot-classifier run \
  --model llama3.1:8b \
  --run-dir artifacts/runs/llama3.1-8b
```

### OpenAI-compatible API

The adapter works with APIs exposing `/v1/chat/completions` and structured JSON responses.

```bash
export OPENAI_API_KEY='...'
zeroshot-classifier run \
  --backend openai-compatible \
  --model your-model \
  --base-url https://api.example.com/v1 \
  --run-dir artifacts/runs/your-model
```

For a differently named key variable, pass `--api-key-env VARIABLE_NAME`. The key is read from the environment and is never written to the manifest or checkpoint.

## 3. Evaluate

```bash
zeroshot-classifier evaluate
```

This command executes `scripts/evaluate.py` directly. Its aspect, sentiment, and overall precision/recall/F1 report is the source of truth and is saved to:

```text
artifacts/runs/default/official_evaluation.txt
```

It also writes `metrics.json` with a machine-readable reproduction of the evaluator's set-based metrics, coverage, and supplemental exact-match accuracy:

- Aspect accuracy: the complete set of aspects for an ID must match.
- Polarity accuracy: the complete set of polarities for an ID must match.
- Overall accuracy: the complete set of `(aspectCategory, polarity)` pairs for an ID must match. Both labels must be correct.

Missing predictions are included as failures in exact-match accuracy and coverage. This is important because the official evaluator only includes gold IDs that appear in the prediction file.

### Evaluate only the held-out 10%

The generated split remains available for a smaller validation run:

```bash
zeroshot-classifier run \
  --input artifacts/splits/test.csv \
  --run-dir artifacts/runs/heldout

zeroshot-classifier evaluate \
  --gold artifacts/splits/test.csv \
  --predictions artifacts/runs/heldout/predictions.csv \
  --metrics-output artifacts/runs/heldout/metrics.json \
  --report-output artifacts/runs/heldout/official_evaluation.txt
```

Custom paths can be supplied when using another run directory:

```bash
zeroshot-classifier evaluate \
  --predictions artifacts/runs/your-model/predictions.csv \
  --metrics-output artifacts/runs/your-model/metrics.json \
  --report-output artifacts/runs/your-model/official_evaluation.txt
```

## Train two RoBERTa classifiers

Install the optional training dependencies, then run the sequential trainer:

```bash
python -m pip install -e '.[training]'
python scripts/train_roberta.py \
  --train-ratio 0.8 \
  --eval-ratio 0.1 \
  --test-ratio 0.1 \
  --epochs 4 \
  --run-dir artifacts/training/roberta-two-model
```

Each global epoch trains the aspect model for one epoch, saves its state and
releases its GPU memory, then does the same for the polarity model. The official
`scripts/evaluate.py` evaluator runs on the evaluation split after both models
finish the epoch. The best paired checkpoint is selected by overall micro-F1,
and the held-out test split is evaluated once at the end.

Training automatically resumes from a compatible run directory. Checkpoints
include model, optimizer, scheduler, mixed-precision, random-number-generator,
epoch, and batch progress. Use a different `--run-dir` after changing the data
or training configuration. Metrics and reports are logged to the MLflow
experiment `contest2-roberta-two-model`; the default tracking store is
the SQLite database `artifacts/mlflow.db`.

Run `python scripts/train_roberta.py --help` for model, batch size, gradient
accumulation, checkpoint interval, mixed precision, and MLflow options.

The multilabel ABSA experiment groups annotations by review, predicts all aspects,
and then predicts polarity conditioned on each detected aspect. It uses the persisted
80/10/10 split and writes an isolated comparison report and checkpoints:

```bash
uv run --extra training python scripts/train_multilabel_absa.py
```

Run the staged class-imbalance and candidate-conditioned joint ABSA experiment:

```bash
uv run --extra training python scripts/run_joint_absa_experiment.py
```

The resumable output is written to
`artifacts/experiments/absa-imbalance-joint-v1/`. To train one controlled
variant instead, use `scripts/train_absa_variant.py` with `--architecture`,
`--polarity-loss`, and a distinct `--run-dir`; focal variants also accept
`--focal-gamma` and `--class-balance-beta`.

Outputs are stored in `artifacts/experiments/multilabel-conditioned-v1/`. The run
is resumable and logs to the `contest2-roberta-multilabel-conditioned` MLflow
experiment.

To train the two classifiers in separate processes, give each task its own run
directory:

```bash
python scripts/train_roberta.py --task aspect --epoch 100 \
  --run-dir artifacts/training/roberta-aspect
python scripts/train_roberta.py --task polarity --epoch 100 \
  --run-dir artifacts/training/roberta-polarity
```

Single-task evaluation still runs the official evaluator. The untrained field
is copied from the evaluation gold rows so checkpoint selection uses only the
selected task's official micro-F1. Single-task contest output contains `id`,
`text`, and only the selected prediction column.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
