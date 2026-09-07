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

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
