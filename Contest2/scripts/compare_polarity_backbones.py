#!/usr/bin/env python3
from __future__ import annotations

import json

from zeroshot_classifier.data import read_labeled_csv
from zeroshot_classifier.deberta_experiment import load_tokenizer
from zeroshot_classifier.deberta_three_experiments import (
    ThreeExperimentConfig,
    _deberta_polarity_paths,
    _joint_config,
    _polarity_logits,
    _roberta_polarity_paths,
    average_backbone_logits,
)
from zeroshot_classifier.labels import ASPECTS, POLARITIES


def class_metrics(gold: list[int], predicted: list[int]) -> dict[str, object]:
    confusion = [[0] * len(POLARITIES) for _ in POLARITIES]
    for truth, guess in zip(gold, predicted):
        confusion[truth][guess] += 1
    classes = {}
    f1_values = []
    for index, label in enumerate(POLARITIES):
        true_positive = confusion[index][index]
        predicted_count = sum(row[index] for row in confusion)
        gold_count = sum(confusion[index])
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / gold_count if gold_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes[label] = {
            'precision': precision, 'recall': recall, 'f1': f1,
            'gold': gold_count, 'predicted': predicted_count,
        }
        f1_values.append(f1)
    return {
        'rows': len(gold),
        'accuracy': sum(a == b for a, b in zip(gold, predicted)) / len(gold),
        'macro_f1': sum(f1_values) / len(f1_values),
        'classes': classes,
        'confusion_matrix': confusion,
    }


def main() -> int:
    import torch
    import transformers

    config = ThreeExperimentConfig()
    rows = read_labeled_csv(config.split_dir / 'test.csv')
    roberta_tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.roberta_model, use_fast=True, local_files_only=True,
    )
    deberta_tokenizer = load_tokenizer(transformers, config.deberta_model)
    roberta_logits = _polarity_logits(
        torch, transformers, roberta_tokenizer,
        _joint_config(config, config.roberta_model, config.run_dir),
        _roberta_polarity_paths(config), rows, 'RoBERTa polarity comparison',
    )
    deberta_logits = _polarity_logits(
        torch, transformers, deberta_tokenizer,
        _joint_config(config, config.deberta_model, config.run_dir),
        _deberta_polarity_paths(config), rows, 'DeBERTa polarity comparison',
    )

    # _polarity_logits emits five aspect candidates per unique review. Select
    # only each annotated gold aspect for a like-for-like polarity comparison.
    review_ids = []
    seen = set()
    for row in rows:
        if row.item_id not in seen:
            seen.add(row.item_id)
            review_ids.append(row.item_id)
    review_index = {item_id: index for index, item_id in enumerate(review_ids)}
    indices = [
        review_index[row.item_id] * len(ASPECTS) + ASPECTS.index(row.aspect)
        for row in rows
    ]
    gold = [POLARITIES.index(row.polarity) for row in rows]
    results = {}
    logits_by_model = {
        'roberta': roberta_logits,
        'deberta': deberta_logits,
        'equal_logit_mix': average_backbone_logits(torch, roberta_logits, deberta_logits),
    }
    for name, logits in logits_by_model.items():
        predicted_all = logits.argmax(dim=-1).tolist()
        predicted = [predicted_all[index] for index in indices]
        results[name] = class_metrics(gold, predicted)

    output = config.run_dir / 'polarity_backbone_comparison.json'
    output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))
    print(f'Wrote {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
