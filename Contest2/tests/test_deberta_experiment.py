import unittest

from zeroshot_classifier.deberta_experiment import (
    config_from_args,
    gradient_accumulation,
    projected_minutes,
    load_tokenizer,
    should_continue,
)


class DebertaExperimentTests(unittest.TestCase):
    def test_defaults_lock_model_seed_and_ten_minute_benchmark(self):
        config = config_from_args([])

        self.assertEqual(config.model, 'microsoft/deberta-v3-base')
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.benchmark_seconds_per_stage * 2, 600)

    def test_batch_fallback_preserves_effective_batch(self):
        self.assertEqual(gradient_accumulation(16, 8), 2)
        self.assertEqual(gradient_accumulation(16, 4), 4)
        self.assertEqual(gradient_accumulation(16, 2), 8)

    def test_runtime_projection_covers_both_stages(self):
        self.assertEqual(projected_minutes(10, 100, 200, 2), 1.0)

    def test_feasibility_requires_both_finite_stages_and_time_limit(self):
        benchmark = {
            'stages': {
                'aspect': {'finite_loss': True, 'optimizer_steps': 2},
                'polarity': {'finite_loss': True, 'optimizer_steps': 2},
            },
            'projected_one_seed_max_minutes': 90,
        }

        self.assertTrue(should_continue(benchmark, 120))
        benchmark['stages']['polarity']['finite_loss'] = False
        self.assertFalse(should_continue(benchmark, 120))

    def test_invalid_candidate_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            config_from_args(['--candidate-batch-sizes', '3'])

    def test_tokenizer_enables_transformers_regex_compatibility_fix(self):
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model, **kwargs):
                return model, kwargs

        model, kwargs = load_tokenizer(
            type('Transformers', (), {'AutoTokenizer': AutoTokenizer}), 'example/model'
        )

        self.assertEqual(model, 'example/model')
        self.assertTrue(kwargs['use_fast'])
        self.assertTrue(kwargs['fix_mistral_regex'])


if __name__ == '__main__':
    unittest.main()
