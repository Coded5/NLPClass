# Zero-shot clickbait classifier

This project uses the OpenAI API to classify headlines without training examples. It can
also evaluate predictions against the labeled test split of
[`christinacdl/clickbait_detection_dataset`](https://huggingface.co/datasets/christinacdl/clickbait_detection_dataset),
where label `1` is clickbait and label `0` is not clickbait.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your API key in `.env`:

```dotenv
OPENAI_API_KEY=your-api-key-here
```

The default model is `gpt-4o-mini`. Set `OPENAI_MODEL` in `.env` or pass `--model`
to use another model that supports structured outputs.

## Classify headlines

```bash
python clickbait_classifier.py predict \
  "Scientists discover a new species in the Pacific" \
  "You won't believe what happened next"
```

For multiple headlines, provide a UTF-8 file containing one headline per line:

```bash
python clickbait_classifier.py predict --file headlines.txt
```

The command prints each label and model-reported confidence as JSON.

## Evaluate on the Hugging Face dataset

The default evaluation shuffles and classifies 100 test examples to limit accidental API
spend:

```bash
python clickbait_classifier.py evaluate
```

Change the sample count or evaluate the complete test split:

```bash
python clickbait_classifier.py evaluate --max-samples 500
python clickbait_classifier.py evaluate --max-samples 0
```

Evaluation reports accuracy, precision, recall, F1, and the confusion matrix. The complete
test split contains 3,787 API-billed headlines; review model pricing before using
`--max-samples 0`. `--batch-size` controls how many headlines are included in each request.

## Tests

The tests use a fake API client and do not require a key:

```bash
python -m unittest discover -s tests
```
