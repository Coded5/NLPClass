# Zero-shot financial-news classifiers

The classifiers send the entire financial-news test set to an LLM in batches, validate structured predictions, save a resumable checkpoint, and write a scikit-learn classification report.

Install the Python dependencies first:

```bash
python -m pip install -r requirements.txt
```

## DeepSeek V4 Flash

Set the API key in the environment and run the DeepSeek classifier:

```bash
export DEEPSEEK_API_KEY="your-api-key"
python deepseek_zero_shot_classifier.py
```

DeepSeek results are written to `results-deepseek-v4-flash/`. The API key is read only from `DEEPSEEK_API_KEY` and is never written to a checkpoint or result file.

## Ollama

Start Ollama and download the default model:

```bash
ollama serve
ollama pull qwen2.5:7b
```

Classify all 4,117 rows:

```bash
python zero_shot_classifier.py
```

Results are written to `results/`:

- `predictions.csv`: source text, true label, predicted label, and correctness
- `classification_report.txt`: scikit-learn text report
- `classification_report.json`: machine-readable report
- `checkpoint.json`: predictions used to resume an interrupted run

Use another model with `--model MODEL`. A quick end-to-end check can be run with `--limit 20 --output-dir results-smoke`. Use `--restart` when changing inference settings while reusing an output directory.
