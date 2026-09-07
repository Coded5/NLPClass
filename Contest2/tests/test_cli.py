import unittest
from pathlib import Path
from unittest.mock import patch

from zeroshot_classifier.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_run_and_evaluate_default_to_the_complete_training_dataset(self):
        parser = build_parser()

        run_args = parser.parse_args(['run'])
        evaluate_args = parser.parse_args(['evaluate'])

        self.assertEqual(run_args.input, Path('data/contest2_train.csv'))
        self.assertEqual(evaluate_args.gold, Path('data/contest2_train.csv'))

    def test_protected_contest_test_file_is_rejected_before_reading(self):
        protected = Path(__file__).resolve().parents[1] / 'data' / 'contest2_test.csv'
        with patch('zeroshot_classifier.cli.write_split') as write_split:
            status = main(['split', '--input', str(protected)])
        self.assertEqual(status, 1)
        write_split.assert_not_called()


if __name__ == '__main__':
    unittest.main()
