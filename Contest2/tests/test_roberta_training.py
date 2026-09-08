import csv
import random
import tempfile
import unittest
from pathlib import Path

from zeroshot_classifier.checkpoint import ResumeMismatchError
from zeroshot_classifier.roberta_training import (
    TrainingConfig,
    _prepare_splits,
    _restore_rng,
    _restore_optimizer_state,
    config_from_args,
    validate_config,
)


class RobertaTrainingTests(unittest.TestCase):
    def test_optimizer_resume_remaps_state_by_parameter_name(self):
        parameter_a = object()
        parameter_b = object()

        class FakeModel:
            @staticmethod
            def named_parameters():
                return [('b', parameter_b), ('a', parameter_a)]

        class FakeOptimizer:
            param_groups = [{'params': [parameter_b, parameter_a]}]
            loaded = None

            @staticmethod
            def state_dict():
                return {'state': {}, 'param_groups': [{'params': [0, 1]}]}

            def load_state_dict(self, state):
                self.loaded = state

        optimizer = FakeOptimizer()
        checkpoint = {
            'model': {'a': object(), 'b': object()},
            'optimizer_param_names': ['a', 'b'],
            'optimizer': {
                'state': {0: {'marker': 'A'}, 1: {'marker': 'B'}},
                'param_groups': [{'params': [0, 1], 'lr': 2e-5}],
            },
        }

        _restore_optimizer_state(optimizer, FakeModel(), checkpoint)

        self.assertEqual(optimizer.loaded['state'][0]['marker'], 'B')
        self.assertEqual(optimizer.loaded['state'][1]['marker'], 'A')

    def test_rng_restore_moves_remapped_checkpoint_tensors_to_cpu(self):
        class FakeTensor:
            def __init__(self):
                self.on_cpu = False

            def detach(self):
                return self

            def cpu(self):
                self.on_cpu = True
                return self

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def set_rng_state_all(states):
                self.assertTrue(all(state.on_cpu for state in states))

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def set_rng_state(state):
                self.assertTrue(state.on_cpu)

        _restore_rng(
            FakeTorch(),
            {
                'python': random.getstate(),
                'torch': FakeTensor(),
                'cuda': [FakeTensor()],
            },
        )

    def test_configurable_ratios_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, 'sum to 1'):
            validate_config(
                TrainingConfig(train_ratio=0.8, eval_ratio=0.15, test_ratio=0.1)
            )

    def test_cli_accepts_split_and_epoch_configuration(self):
        config = config_from_args(
            [
                '--train-ratio', '0.7',
                '--eval-ratio', '0.15',
                '--test-ratio', '0.15',
                '--epochs', '7',
            ]
        )

        self.assertEqual(config.train_ratio, 0.7)
        self.assertEqual(config.eval_ratio, 0.15)
        self.assertEqual(config.test_ratio, 0.15)
        self.assertEqual(config.epochs, 7)

    def test_cli_accepts_singular_epoch_alias(self):
        config = config_from_args(['--epoch', '100', '--task', 'aspect'])

        self.assertEqual(config.epochs, 100)
        self.assertEqual(config.task, 'aspect')
        self.assertEqual(config.mlflow_tracking_uri, 'sqlite:///artifacts/mlflow.db')

    def test_cli_accepts_evaluation_interval(self):
        config = config_from_args(['--evaluation-interval', '5'])

        self.assertEqual(config.evaluation_interval, 5)

    def test_evaluation_interval_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, 'evaluation-interval'):
            validate_config(TrainingConfig(evaluation_interval=0))

    def test_persisted_three_way_split_has_no_id_leakage(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / 'train.csv'
            self._write_dataset(source)
            config = TrainingConfig(input=source, run_dir=root / 'run')

            splits, manifest = _prepare_splits(config)
            ids = {
                name: {row.item_id for row in rows}
                for name, rows in splits.items()
            }

            self.assertFalse(ids['train'] & ids['eval'])
            self.assertFalse(ids['train'] & ids['test'])
            self.assertFalse(ids['eval'] & ids['test'])
            self.assertEqual(sum(manifest['ids'].values()), 50)
            self.assertTrue((config.run_dir / 'splits' / 'eval.csv').is_file())

    def test_resume_rejects_changed_training_configuration(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / 'train.csv'
            self._write_dataset(source)
            run_dir = root / 'run'
            _prepare_splits(TrainingConfig(input=source, run_dir=run_dir))

            with self.assertRaises(ResumeMismatchError):
                _prepare_splits(
                    TrainingConfig(input=source, run_dir=run_dir, epochs=5)
                )

    @staticmethod
    def _write_dataset(path):
        labels = (
            ('food', 'positive'),
            ('price', 'negative'),
            ('service', 'neutral'),
            ('ambience', 'conflict'),
            ('anecdotes/miscellaneous', 'positive'),
        )
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('id', 'text', 'aspectCategory', 'polarity'))
            for index in range(50):
                aspect, polarity = labels[index % len(labels)]
                writer.writerow((index, f'Review {index}', aspect, polarity))


if __name__ == '__main__':
    unittest.main()
