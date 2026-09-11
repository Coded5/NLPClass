#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / 'src'
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from zeroshot_classifier.data import read_labeled_csv
from zeroshot_classifier.deberta_experiment import load_tokenizer
from zeroshot_classifier.deberta_three_experiments import (
    ThreeExperimentConfig,
    _aspect_inference_config,
    _aspect_probabilities,
    _decode,
    _deberta_polarity_paths,
    _joint_config,
    _polarity_logits,
    _roberta_aspect_paths,
    _roberta_polarity_paths,
    average_backbone_logits,
)
from zeroshot_classifier.evaluation import read_predictions
from zeroshot_classifier.io_utils import atomic_write_json, file_sha256
from zeroshot_classifier.joint_absa_experiment import _write_evaluation


def main() -> int:
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError('Current-winner reevaluation requires CUDA')

    config = ThreeExperimentConfig()
    system_dir = config.run_dir / 'systems/roberta-aspect-mixed-polarity'
    output_dir = config.run_dir / 'reevaluations/current-winner-20260911'
    locked_path = system_dir / 'locked_selection.json'
    locked = json.loads(locked_path.read_text(encoding='utf-8'))
    thresholds = locked['thresholds']
    rows = read_labeled_csv(config.split_dir / 'test.csv')

    roberta_tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.roberta_model, use_fast=True, local_files_only=True,
    )
    deberta_tokenizer = load_tokenizer(transformers, config.deberta_model)

    aspect_paths = _roberta_aspect_paths(config)
    roberta_polarity_paths = _roberta_polarity_paths(config)
    deberta_polarity_paths = _deberta_polarity_paths(config)
    aspect_probabilities = _aspect_probabilities(
        torch, transformers, roberta_tokenizer,
        _aspect_inference_config(config, config.roberta_model),
        aspect_paths, rows, 'Rerun RoBERTa aspects',
    )
    polarity_config = _joint_config(config, config.roberta_model, output_dir)
    roberta_logits = _polarity_logits(
        torch, transformers, roberta_tokenizer, polarity_config,
        roberta_polarity_paths, rows, 'Rerun RoBERTa polarity',
    )
    deberta_logits = _polarity_logits(
        torch, transformers, deberta_tokenizer,
        _joint_config(config, config.deberta_model, output_dir),
        deberta_polarity_paths, rows, 'Rerun DeBERTa polarity',
    )
    mixed_logits = average_backbone_logits(torch, roberta_logits, deberta_logits)
    predictions, used_thresholds = _decode(
        rows, aspect_probabilities, mixed_logits, thresholds,
    )
    metrics = _write_evaluation(
        polarity_config, output_dir, rows, predictions,
    )

    original_path = system_dir / 'test/predictions.csv'
    original = read_predictions(original_path)
    comparison = {
        'system': 'roberta-aspect-mixed-polarity',
        'device': str(torch.device('cuda')),
        'rows': len(rows),
        'thresholds': used_thresholds,
        'predictions_identical_to_original': predictions == original,
        'prediction_sets_identical_to_original': set(predictions) == set(original),
        'rerun_prediction_count': len(predictions),
        'original_prediction_count': len(original),
        'pair_micro_f1': metrics['source_of_truth_metrics']['overall']['micro']['f1'],
        'checkpoints': {
            'roberta_aspect': [str(path) for path in aspect_paths],
            'roberta_polarity': [str(path) for path in roberta_polarity_paths],
            'deberta_polarity': [str(path) for path in deberta_polarity_paths],
        },
        'checkpoint_sha256': {
            str(path): file_sha256(path)
            for path in aspect_paths + roberta_polarity_paths + deberta_polarity_paths
        },
        'locked_selection': str(locked_path),
        'test_split_sha256': file_sha256(config.split_dir / 'test.csv'),
    }
    atomic_write_json(output_dir / 'comparison.json', comparison)
    print(json.dumps(comparison, indent=2))
    print(f'Wrote reevaluation to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
