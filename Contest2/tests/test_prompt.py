import unittest

from zeroshot_classifier.prompt import ResponseValidationError, parse_response


class PromptTests(unittest.TestCase):
    def test_parses_fenced_response_and_orders_predictions(self):
        raw = '''```json
        {"predictions":[
          {"aspectCategory":"service","polarity":"negative"},
          {"aspectCategory":"food","polarity":"positive"}
        ]}
        ```'''

        predictions = parse_response(raw)

        self.assertEqual(
            [(prediction.aspect, prediction.polarity) for prediction in predictions],
            [('food', 'positive'), ('service', 'negative')],
        )

    def test_rejects_invalid_label(self):
        with self.assertRaises(ResponseValidationError):
            parse_response(
                '{"predictions":[{"aspectCategory":"parking","polarity":"positive"}]}'
            )

    def test_rejects_two_polarities_for_one_aspect(self):
        with self.assertRaises(ResponseValidationError):
            parse_response(
                '{"predictions":['
                '{"aspectCategory":"food","polarity":"positive"},'
                '{"aspectCategory":"food","polarity":"negative"}'
                ']}'
            )


if __name__ == '__main__':
    unittest.main()
