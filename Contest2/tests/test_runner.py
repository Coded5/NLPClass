import csv
import tempfile
import unittest
from pathlib import Path

from zeroshot_classifier.backends import LLMBackend, TransientBackendError
from zeroshot_classifier.checkpoint import ResumeMismatchError, RunLock, RunLockedError, RunStore
from zeroshot_classifier.data import InputItem
from zeroshot_classifier.runner import RunnerSettings, run_classification


class FakeBackend(LLMBackend):
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, schema):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RunnerTests(unittest.TestCase):
    def test_retries_failures_and_resumes_without_repeating_completed_items(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            database = run_dir / 'checkpoint.sqlite3'
            manifest = {'input': 'test', 'model': 'fake'}
            items = [InputItem('1', 'Great food.', 0), InputItem('2', 'Slow staff.', 1)]
            first_backend = FakeBackend(
                [
                    TransientBackendError('temporary'),
                    '{"predictions":[{"aspectCategory":"food","polarity":"positive"}]}',
                    'not json',
                    'still not json',
                ]
            )
            with RunStore(database, manifest) as store:
                summary = run_classification(
                    items,
                    first_backend,
                    store,
                    run_dir,
                    RunnerSettings(max_attempts=2, initial_backoff=0),
                    sleep=lambda _: None,
                )
            self.assertEqual(summary['completed'], 1)
            self.assertEqual(summary['failed'], 1)
            self.assertEqual(first_backend.calls, 4)

            second_backend = FakeBackend(
                ['{"predictions":[{"aspectCategory":"service","polarity":"negative"}]}']
            )
            with RunStore(database, manifest) as store:
                summary = run_classification(
                    items,
                    second_backend,
                    store,
                    run_dir,
                    RunnerSettings(max_attempts=1, retry_failed=True),
                    sleep=lambda _: None,
                )
            self.assertEqual(summary['completed'], 2)
            self.assertEqual(summary['failed'], 0)
            self.assertEqual(second_backend.calls, 1)
            with (run_dir / 'predictions.csv').open(newline='', encoding='utf-8') as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual(len(predictions), 2)

    def test_rejects_resume_with_different_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            database = Path(temporary_dir) / 'checkpoint.sqlite3'
            with RunStore(database, {'model': 'one'}):
                pass
            with self.assertRaises(ResumeMismatchError):
                RunStore(database, {'model': 'two'})

    def test_run_lock_rejects_a_second_process_lock(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            lock_path = Path(temporary_dir) / '.run.lock'
            with RunLock(lock_path):
                with self.assertRaises(RunLockedError):
                    with RunLock(lock_path):
                        pass

    def test_interrupted_attempt_still_counts_toward_persistent_limit(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            database = run_dir / 'checkpoint.sqlite3'
            manifest = {'model': 'fake'}
            item = InputItem('1', 'Review', 0)
            with RunStore(database, manifest) as store:
                store.register_items([item])
                store.begin_attempt(item.item_id)

            backend = FakeBackend(
                ['{"predictions":[{"aspectCategory":"food","polarity":"positive"}]}']
            )
            with RunStore(database, manifest) as store:
                summary = run_classification(
                    [item],
                    backend,
                    store,
                    run_dir,
                    RunnerSettings(max_attempts=1),
                    sleep=lambda _: None,
                )

            self.assertEqual(backend.calls, 0)
            self.assertEqual(summary['failed'], 1)


if __name__ == '__main__':
    unittest.main()
