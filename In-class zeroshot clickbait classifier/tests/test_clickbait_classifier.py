import json
import unittest
from types import SimpleNamespace

from clickbait_classifier import ZeroShotClickbaitClassifier, calculate_metrics


class FakeResponses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        headlines = json.loads(kwargs["input"])["headlines"]
        classifications = [
            {
                "id": item["id"],
                "label": "CLICKBAIT" if "shocking" in item["text"] else "NOT_CLICKBAIT",
                "confidence": 0.9,
            }
            for item in headlines
        ]
        return SimpleNamespace(
            output_text=json.dumps({"classifications": classifications})
        )


class ZeroShotClickbaitClassifierTests(unittest.TestCase):
    def test_predict_preserves_order_and_batches_requests(self):
        responses = FakeResponses()
        client = SimpleNamespace(responses=responses)
        classifier = ZeroShotClickbaitClassifier(
            model="test-model", batch_size=2, client=client
        )

        predictions = classifier.predict(
            ["A factual headline", "A shocking discovery", "Another report"]
        )

        self.assertEqual(2, len(responses.calls))
        self.assertEqual(
            ["NOT_CLICKBAIT", "CLICKBAIT", "NOT_CLICKBAIT"],
            [prediction.label for prediction in predictions],
        )
        self.assertEqual("test-model", responses.calls[0]["model"])
        self.assertEqual("json_schema", responses.calls[0]["text"]["format"]["type"])

    def test_predict_rejects_empty_headlines(self):
        classifier = ZeroShotClickbaitClassifier(
            client=SimpleNamespace(responses=FakeResponses())
        )

        with self.assertRaises(ValueError):
            classifier.predict([""])


class MetricsTests(unittest.TestCase):
    def test_calculate_metrics(self):
        metrics = calculate_metrics([1, 1, 0, 0], [1, 0, 1, 0])

        self.assertEqual(0.5, metrics["accuracy"])
        self.assertEqual(0.5, metrics["precision"])
        self.assertEqual(0.5, metrics["recall"])
        self.assertEqual(0.5, metrics["f1"])
        self.assertEqual(
            {
                "true_negative": 1,
                "false_positive": 1,
                "false_negative": 1,
                "true_positive": 1,
            },
            metrics["confusion_matrix"],
        )


if __name__ == "__main__":
    unittest.main()
