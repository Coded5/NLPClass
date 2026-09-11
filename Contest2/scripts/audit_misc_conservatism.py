#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / 'src'
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from zeroshot_classifier.absa_training import group_aspect_targets
from zeroshot_classifier.data import read_labeled_csv
from zeroshot_classifier.deberta_three_experiments import (
    ThreeExperimentConfig,
    _aspect_inference_config,
    _roberta_aspect_paths,
)
from zeroshot_classifier.io_utils import atomic_write_json
from zeroshot_classifier.labels import ASPECTS
from zeroshot_classifier.multilabel_aspect_experiment import (
    _predict_checkpoint_probabilities,
)


MISC = ASPECTS.index('anecdotes/miscellaneous')


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def value(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        'minimum': ordered[0], 'p10': value(0.10), 'p25': value(0.25),
        'median': value(0.50), 'p75': value(0.75), 'p90': value(0.90),
        'maximum': ordered[-1], 'mean': mean(ordered),
    }


def summarize(rows: list[object], seed_probabilities: list[list[list[float]]]) -> dict[str, object]:
    reviews = group_aspect_targets(rows)
    ensemble = [
        mean(seed[index][MISC] for seed in seed_probabilities)
        for index in range(len(reviews))
    ]
    records = []
    for index, review in enumerate(reviews):
        gold_misc = MISC in review.aspects
        records.append({
            'id': review.item_id,
            'gold_misc': gold_misc,
            'gold_aspect_count': len(review.aspects),
            'gold_aspects': [ASPECTS[item] for item in review.aspects],
            'ensemble_probability': ensemble[index],
            'seed_probabilities': [seed[index][MISC] for seed in seed_probabilities],
            'text': review.text,
        })

    groups = {}
    selectors = {
        'gold_single': lambda item: item['gold_misc'] and item['gold_aspect_count'] == 1,
        'gold_multi': lambda item: item['gold_misc'] and item['gold_aspect_count'] > 1,
        'gold_all': lambda item: item['gold_misc'],
        'nongold': lambda item: not item['gold_misc'],
    }
    for name, selector in selectors.items():
        groups[name] = quantiles([
            item['ensemble_probability'] for item in records if selector(item)
        ])

    sweep = []
    for step in range(5, 100, 5):
        threshold = step / 100
        tp = sum(item['gold_misc'] and item['ensemble_probability'] >= threshold for item in records)
        fp = sum(not item['gold_misc'] and item['ensemble_probability'] >= threshold for item in records)
        fn = sum(item['gold_misc'] and item['ensemble_probability'] < threshold for item in records)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        sweep.append({
            'threshold': threshold, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
        })

    for item in records:
        item['predicted_at_0.8'] = item['ensemble_probability'] >= 0.8
    return {'groups': groups, 'threshold_sweep': sweep, 'records': records}


def main() -> int:
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError('Miscellaneous score audit requires CUDA')
    config = ThreeExperimentConfig()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.roberta_model, use_fast=True, local_files_only=True,
    )
    inference_config = _aspect_inference_config(config, config.roberta_model)
    checkpoints = _roberta_aspect_paths(config)
    output = {}
    for split in ('eval', 'test'):
        rows = read_labeled_csv(config.split_dir / f'{split}.csv')
        probabilities = [
            _predict_checkpoint_probabilities(
                torch, transformers, tokenizer, inference_config, checkpoint,
                rows, f'{split} miscellaneous audit seed {index + 1}/{len(checkpoints)}',
            )
            for index, checkpoint in enumerate(checkpoints)
        ]
        output[split] = summarize(rows, probabilities)

    path = (
        config.run_dir / 'systems/roberta-aspect-mixed-polarity'
        / 'misc_conservatism_scores.json'
    )
    atomic_write_json(path, output)
    print(json.dumps({
        split: {
            'groups': result['groups'],
            'threshold_sweep': result['threshold_sweep'],
        }
        for split, result in output.items()
    }, indent=2))
    print(f'Wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
