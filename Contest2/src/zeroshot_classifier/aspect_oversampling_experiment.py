from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from .absa_training import ExperimentConfig as TrainingConfig, group_aspect_targets, train_stage
from .checkpoint import RunLock
from .data import LabeledRow, read_labeled_csv
from .deberta_experiment import load_tokenizer
from .deberta_three_experiments import (
    ThreeExperimentConfig,
    _decode,
    _deberta_polarity_paths,
    _joint_config,
    _polarity_logits,
    _roberta_polarity_paths,
    average_backbone_logits,
)
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .joint_absa_experiment import _write_evaluation as write_pipeline_evaluation
from .labels import ASPECTS
from .multilabel_aspect_experiment import (
    AspectExperimentConfig,
    _export_compact_checkpoint,
    _predict_checkpoint_probabilities,
    _target_vectors,
    _write_evaluation,
    aspect_predictions,
    calculate_aspect_metrics,
    mean_probabilities,
    tune_thresholds,
)
from .roberta_training import _prepare_mlflow_tracking_uri, _seed_everything


LOGGER = logging.getLogger(__name__)
MISC = 'anecdotes/miscellaneous'
CONFIGURATIONS = ('reference', 'sampling-control', 'targeted-2x', 'targeted-3x', 'targeted-4x')
MULTIPLIERS = {'reference': None, 'sampling-control': 1.0, 'targeted-2x': 2.0,
               'targeted-3x': 3.0, 'targeted-4x': 4.0}


@dataclass(frozen=True)
class OversamplingConfig:
    split_dir: Path = Path('artifacts/training/roberta-aspect-exp1/splits')
    run_dir: Path = Path('artifacts/experiments/aspect-misc-oversampling-cv-v1')
    optimizer_dir: Path = Path('/tmp/contest2-aspect-misc-oversampling-cv-v1')
    model: str = 'FacebookAI/roberta-base'
    folds: int = 5
    seed: int = 42
    final_seeds: tuple[int, ...] = (42, 43, 44)
    epochs: int = 50
    evaluation_interval: int = 5
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
    max_length: int = 256
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    checkpoint_steps: int = 100
    log_steps: int = 10
    mixed_precision: str = 'auto'
    micro_f1_tolerance: float = 0.005
    mlflow_tracking_uri: str = 'sqlite:////home/kami/Projects/NLP/Contest2/artifacts/mlflow.db'
    mlflow_experiment: str = 'contest2-aspect-misc-oversampling-cv-v1'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run targeted multi-aspect miscellaneous oversampling.')
    for name, field in OversamplingConfig.__dataclass_fields__.items():
        default = field.default; option = f'--{name.replace("_", "-")}'
        if name == 'final_seeds':
            parser.add_argument(option, type=int, nargs='+', default=list(default))
        elif isinstance(default, Path):
            parser.add_argument(option, type=Path, default=default)
        else:
            parser.add_argument(option, type=type(default), default=default)
    return parser


def config_from_args(argv: list[str] | None = None) -> OversamplingConfig:
    values = vars(build_parser().parse_args(argv)); values['final_seeds'] = tuple(values['final_seeds'])
    config = OversamplingConfig(**values)
    if config.folds < 2 or not config.final_seeds or len(set(config.final_seeds)) != len(config.final_seeds):
        raise ValueError('Folds and unique final seeds are required')
    if config.micro_f1_tolerance < 0:
        raise ValueError('Micro-F1 tolerance must be nonnegative')
    return config


def is_target_review(aspects: Sequence[int]) -> bool:
    return len(aspects) > 1 and ASPECTS.index(MISC) in aspects


def sampling_weights(rows: Sequence[LabeledRow], configuration: str) -> list[float] | None:
    if configuration not in CONFIGURATIONS:
        raise ValueError(f'Unknown configuration: {configuration}')
    multiplier = MULTIPLIERS[configuration]
    if multiplier is None:
        return None
    return [multiplier if is_target_review(review.aspects) else 1.0
            for review in group_aspect_targets(rows)]


def grouped_aspect_folds(rows: Sequence[LabeledRow], folds: int, seed: int) -> list[set[str]]:
    reviews = group_aspect_targets(rows)
    if folds < 2 or folds > len(reviews):
        raise ValueError('Fold count must be between 2 and the review count')
    feature_count = len(ASPECTS) + 2
    features = {}
    for review in reviews:
        vector = [int(index in review.aspects) for index in range(len(ASPECTS))]
        vector.extend((int(is_target_review(review.aspects)), len(review.aspects) - 1))
        features[review.item_id] = vector
    totals = [sum(value[index] for value in features.values()) for index in range(feature_count)]
    targets = [max(value / folds, 1.0) for value in totals]
    rng = random.Random(seed); tie = {review.item_id: rng.random() for review in reviews}
    ordered = sorted(reviews, key=lambda review: (
        -sum(value / max(total, 1) for value, total in zip(features[review.item_id], totals)),
        -len(review.aspects), tie[review.item_id], review.item_id,
    ))
    assignments = [set() for _ in range(folds)]; counts = [[0] * feature_count for _ in range(folds)]
    target_size = len(reviews) / folds
    for review in ordered:
        vector = features[review.item_id]
        destination = min(range(folds), key=lambda fold: (
            sum(((counts[fold][i] + vector[i]) / targets[i]) ** 2 for i in range(feature_count)),
            ((len(assignments[fold]) + 1) / target_size) ** 2, len(assignments[fold]), fold,
        ))
        assignments[destination].add(review.item_id)
        counts[destination] = [left + right for left, right in zip(counts[destination], vector)]
    return assignments


def subgroup_metrics(rows: Sequence[LabeledRow], predictions: Sequence[tuple[str, str]]) -> dict[str, Any]:
    reviews = group_aspect_targets(rows); predicted = set(predictions)
    targets = [review.item_id for review in reviews if is_target_review(review.aspects)]
    hits = sum((item_id, MISC) in predicted for item_id in targets)
    return {'support': len(targets), 'correct': hits, 'recall': hits / len(targets) if targets else 0.0}


def extended_metrics(rows: Sequence[LabeledRow], predictions: Sequence[tuple[str, str]]) -> dict[str, Any]:
    metrics = calculate_aspect_metrics(rows, predictions)
    metrics['multi_misc'] = subgroup_metrics(rows, predictions)
    return metrics


def eligible(candidate: dict[str, Any], reference: dict[str, Any], tolerance: float,
             fold_non_regressions: int, folds: int) -> bool:
    return (
        candidate['aspect']['micro']['f1'] >= reference['aspect']['micro']['f1'] - tolerance
        and candidate['aspect']['classes'][MISC]['f1'] >= reference['aspect']['classes'][MISC]['f1']
        and candidate['multi_misc']['recall'] > reference['multi_misc']['recall']
        and fold_non_regressions >= math.ceil(0.8 * folds)
    )


def _training_config(config: OversamplingConfig, seed: int, run_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        split_dir=config.split_dir, run_dir=run_dir, model=config.model, epochs=config.epochs,
        evaluation_interval=config.evaluation_interval, early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta, seed=seed, max_length=config.max_length,
        learning_rate=config.learning_rate, train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size, gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay, warmup_ratio=config.warmup_ratio,
        checkpoint_steps=config.checkpoint_steps, log_steps=config.log_steps,
        mixed_precision=config.mixed_precision, mlflow_tracking_uri=config.mlflow_tracking_uri,
        mlflow_experiment=config.mlflow_experiment, mlflow_run_name=f'aspect-oversampling-{seed}',
    )


def _probabilities(torch: Any, transformers: Any, tokenizer: Any, config: OversamplingConfig,
                   checkpoint: Path, rows: list[LabeledRow], label: str) -> list[list[float]]:
    inference = replace(AspectExperimentConfig(), model=config.model, split_dir=config.split_dir,
                        eval_batch_size=config.eval_batch_size)
    return _predict_checkpoint_probabilities(torch, transformers, tokenizer, inference, checkpoint, rows, label)


def _train_one(torch: Any, transformers: Any, mlflow: Any, tokenizer: Any,
               config: OversamplingConfig, name: str, seed: int,
               train_rows: list[LabeledRow], selection_rows: list[LabeledRow],
               output_dir: Path) -> tuple[Path, dict[str, Any]]:
    result_path = output_dir / 'result.json'; compact = output_dir / 'best/aspect.pt'
    if result_path.is_file() and compact.is_file():
        return compact, json.loads(result_path.read_text())
    scratch = config.optimizer_dir / output_dir.relative_to(config.run_dir)
    training = _training_config(config, seed, scratch); _seed_everything(torch, seed)
    with mlflow.start_run(run_name=f'{name}-seed-{seed}'):
        selection = train_stage(torch, transformers, mlflow, tokenizer, training, 'aspect',
                                train_rows, selection_rows, sampling_weights(train_rows, name))
    _export_compact_checkpoint(torch, scratch / 'best/aspect.pt', compact)
    result = {'configuration': name, 'seed': seed, 'selection': selection}
    atomic_write_json(result_path, result); shutil.rmtree(scratch)
    return compact, result


def _fold_rows(rows: list[LabeledRow], ids: set[str], include: bool) -> list[LabeledRow]:
    return [row for row in rows if (row.item_id in ids) == include]


def _run_cv(torch: Any, transformers: Any, mlflow: Any, tokenizer: Any,
            config: OversamplingConfig, rows: list[LabeledRow]) -> dict[str, Any]:
    folds = grouped_aspect_folds(rows, config.folds, config.seed); output = {}
    for name in CONFIGURATIONS:
        fold_results = []; pooled_predictions = []
        for index, heldout_ids in enumerate(folds, 1):
            outer_train = _fold_rows(rows, heldout_ids, False); heldout = _fold_rows(rows, heldout_ids, True)
            inner_ids = grouped_aspect_folds(outer_train, 10, config.seed + index)[0]
            fit = _fold_rows(outer_train, inner_ids, False); selection = _fold_rows(outer_train, inner_ids, True)
            fold_dir = config.run_dir / 'cv' / name / f'fold-{index}'
            fold_metrics_path = fold_dir / 'fold_metrics.json'
            if fold_metrics_path.is_file():
                fold_result = json.loads(fold_metrics_path.read_text())
                pooled_predictions.extend(tuple(value) for value in fold_result['predictions'])
                fold_results.append(fold_result)
                LOGGER.info('Reusing completed %s fold %d', name, index)
                continue
            checkpoint, training_result = _train_one(
                torch, transformers, mlflow, tokenizer, config, name, config.seed + index,
                fit, selection, fold_dir,
            )
            selection_probabilities = _probabilities(
                torch, transformers, tokenizer, config, checkpoint, selection, f'{name} fold {index} selection')
            thresholds = tune_thresholds(selection_probabilities, _target_vectors(selection))
            heldout_probabilities = _probabilities(
                torch, transformers, tokenizer, config, checkpoint, heldout, f'{name} fold {index} heldout')
            predictions = aspect_predictions(heldout, heldout_probabilities, thresholds)
            metrics = extended_metrics(heldout, predictions); pooled_predictions.extend(predictions)
            fold_result = {**training_result, 'thresholds': thresholds, 'metrics': metrics,
                           'predictions': [list(value) for value in predictions]}
            atomic_write_json(fold_metrics_path, fold_result); fold_results.append(fold_result)
            (fold_dir / 'best/aspect.pt').unlink(missing_ok=True)
        pooled = extended_metrics(rows, pooled_predictions)
        output[name] = {'pooled': pooled, 'folds': fold_results}
        atomic_write_json(config.run_dir / 'cv' / name / 'summary.json', output[name])
    reference = output['reference']; candidates = []
    for name in CONFIGURATIONS[2:]:
        non_regressions = sum(
            output[name]['folds'][i]['metrics']['multi_misc']['recall']
            >= reference['folds'][i]['metrics']['multi_misc']['recall']
            for i in range(config.folds)
        )
        output[name]['fold_non_regressions'] = non_regressions
        output[name]['eligible'] = eligible(output[name]['pooled'], reference['pooled'],
                                            config.micro_f1_tolerance, non_regressions, config.folds)
        if output[name]['eligible']:
            candidates.append(name)
    winner = max(candidates, key=lambda name: (
        output[name]['pooled']['multi_misc']['recall'],
        output[name]['pooled']['aspect']['micro']['f1'],
        output[name]['pooled']['aspect']['classes'][MISC]['f1'],
        output[name]['pooled']['exact_set_accuracy'],
    )) if candidates else 'reference'
    return {'fold_ids': [sorted(values) for values in folds], 'configurations': output,
            'eligible': candidates, 'winner': winner}


def _final_confirmation(torch: Any, transformers: Any, mlflow: Any, tokenizer: Any,
                        deberta_tokenizer: Any, config: OversamplingConfig, winner: str,
                        train_rows: list[LabeledRow], validation_rows: list[LabeledRow]) -> dict[str, Any]:
    if winner == 'reference':
        return {'promoted': False, 'locked_winner': 'reference', 'reason': 'No CV candidate passed'}
    checkpoints = []
    for seed in config.final_seeds:
        checkpoint, _ = _train_one(torch, transformers, mlflow, tokenizer, config, winner, seed,
                                   train_rows, validation_rows, config.run_dir / 'final' / winner / f'seed-{seed}')
        checkpoints.append(checkpoint)
    baseline_paths = [Path(f'artifacts/experiments/multilabel-aspect-v1/runs/seed-{seed}/best/aspect.pt')
                      for seed in config.final_seeds]
    aspect_config = replace(AspectExperimentConfig(), model=config.model,
                            split_dir=config.split_dir, eval_batch_size=config.eval_batch_size)
    def ensemble(paths: Sequence[Path], rows: list[LabeledRow], label: str) -> list[list[float]]:
        return mean_probabilities([_predict_checkpoint_probabilities(
            torch, transformers, tokenizer, aspect_config, path, rows,
            f'{label} seed {index + 1}/{len(paths)}') for index, path in enumerate(paths)])
    validation_aspects = {'reference': ensemble(baseline_paths, validation_rows, 'reference validation'),
                          winner: ensemble(checkpoints, validation_rows, 'candidate validation')}
    aspect_results = {}; aspect_thresholds = {}
    for name in ('reference', winner):
        thresholds = tune_thresholds(validation_aspects[name], _target_vectors(validation_rows))
        aspect_thresholds[name] = thresholds
        aspect_results[name] = {
            'validation': extended_metrics(validation_rows, aspect_predictions(validation_rows, validation_aspects[name], thresholds)),
        }
    three = ThreeExperimentConfig(); roberta_config = _joint_config(three, three.roberta_model, config.run_dir)
    deberta_config = _joint_config(three, three.deberta_model, config.run_dir)
    validation_logits = average_backbone_logits(torch,
        _polarity_logits(torch, transformers, tokenizer, roberta_config, _roberta_polarity_paths(three), validation_rows, 'validation RoBERTa polarity'),
        _polarity_logits(torch, transformers, deberta_tokenizer, deberta_config, _deberta_polarity_paths(three), validation_rows, 'validation DeBERTa polarity'))
    pipeline = {}; pair_thresholds = {}
    for name in ('reference', winner):
        validation_predictions, thresholds = _decode(validation_rows, validation_aspects[name], validation_logits)
        pair_thresholds[name] = thresholds
        pipeline[name] = {
            'validation': write_pipeline_evaluation(roberta_config, config.run_dir / 'final/evaluations' / name / 'validation', validation_rows, validation_predictions),
        }
    reference = aspect_results['reference']['validation']; candidate = aspect_results[winner]['validation']
    pair_reference = pipeline['reference']['validation']['source_of_truth_metrics']['overall']['micro']['f1']
    pair_candidate = pipeline[winner]['validation']['source_of_truth_metrics']['overall']['micro']['f1']
    promoted = (candidate['aspect']['micro']['f1'] >= reference['aspect']['micro']['f1'] - config.micro_f1_tolerance
                and candidate['aspect']['classes'][MISC]['f1'] >= reference['aspect']['classes'][MISC]['f1']
                and candidate['multi_misc']['recall'] > reference['multi_misc']['recall']
                and pair_candidate >= pair_reference - config.micro_f1_tolerance)
    locked = winner if promoted else 'reference'
    return {'candidate': winner, 'promoted': promoted, 'locked_winner': locked,
            'aspect_thresholds': aspect_thresholds, 'pair_thresholds': pair_thresholds,
            'aspect': aspect_results, 'pipeline': pipeline,
            'checkpoints': {winner: [str(path) for path in checkpoints],
                            'reference': [str(path) for path in baseline_paths]}}


def _final_test(torch: Any, transformers: Any, tokenizer: Any, deberta_tokenizer: Any,
                config: OversamplingConfig, final: dict[str, Any],
                test_rows: list[LabeledRow]) -> None:
    winner = final['candidate']; names = ('reference', winner)
    aspect_config = replace(AspectExperimentConfig(), model=config.model,
                            split_dir=config.split_dir, eval_batch_size=config.eval_batch_size)
    aspect_probabilities = {}
    for name in names:
        paths = [Path(value) for value in final['checkpoints'][name]]
        aspect_probabilities[name] = mean_probabilities([
            _predict_checkpoint_probabilities(
                torch, transformers, tokenizer, aspect_config, path, test_rows,
                f'{name} test seed {index + 1}/{len(paths)}',
            ) for index, path in enumerate(paths)
        ])
        predictions = aspect_predictions(test_rows, aspect_probabilities[name],
                                         final['aspect_thresholds'][name])
        final['aspect'][name]['test'] = extended_metrics(test_rows, predictions)
    three = ThreeExperimentConfig(); roberta_config = _joint_config(three, three.roberta_model, config.run_dir)
    test_logits = average_backbone_logits(torch,
        _polarity_logits(torch, transformers, tokenizer, roberta_config,
                         _roberta_polarity_paths(three), test_rows, 'test RoBERTa polarity'),
        _polarity_logits(torch, transformers, deberta_tokenizer,
                         _joint_config(three, three.deberta_model, config.run_dir),
                         _deberta_polarity_paths(three), test_rows, 'test DeBERTa polarity'))
    for name in names:
        predictions, _ = _decode(test_rows, aspect_probabilities[name], test_logits,
                                 final['pair_thresholds'][name])
        final['pipeline'][name]['test'] = write_pipeline_evaluation(
            roberta_config, config.run_dir / 'final/evaluations' / name / 'test',
            test_rows, predictions,
        )


def _report(results: dict[str, Any]) -> str:
    rows = []
    for name, value in results['cv']['configurations'].items():
        metrics = value['pooled']; rows.append(
            f"| {name} | {metrics['aspect']['micro']['f1']:.6f} | "
            f"{metrics['aspect']['classes'][MISC]['f1']:.6f} | {metrics['multi_misc']['recall']:.6f} | "
            f"{value.get('fold_non_regressions', '-')} | {value.get('eligible', name == 'reference')} |")
    final = results['final']; final_text = ''
    if 'aspect' in final:
        final_rows = []
        for name in ('reference', final['candidate']):
            aspect = final['aspect'][name]; pipeline = final['pipeline'][name]
            final_rows.append(f"| {name} | {aspect['validation']['aspect']['micro']['f1']:.6f} | "
                f"{aspect['validation']['multi_misc']['recall']:.6f} | "
                f"{pipeline['validation']['source_of_truth_metrics']['overall']['micro']['f1']:.6f} | "
                f"{aspect['test']['aspect']['micro']['f1']:.6f} | {aspect['test']['multi_misc']['recall']:.6f} | "
                f"{pipeline['test']['source_of_truth_metrics']['overall']['micro']['f1']:.6f} |")
        final_text = '''\n## Three-seed confirmation\n\n| System | Val aspect F1 | Val subgroup recall | Val pair F1 | Test aspect F1 | Test subgroup recall | Test pair F1 |\n|---|---:|---:|---:|---:|---:|---:|\n''' + '\n'.join(final_rows)
    return '''# Targeted multi-aspect miscellaneous oversampling\n\n## Train-only five-fold CV\n\n| Configuration | Aspect micro-F1 | Misc F1 | Multi-misc recall | Fold non-regressions | Eligible |\n|---|---:|---:|---:|---:|---|\n''' + '\n'.join(rows) + f"\n\nCV winner: `{results['cv']['winner']}`.\n" + final_text + f"\n\nLocked winner: `{final['locked_winner']}`. Test was not used for selection.\n"


def run_experiment(config: OversamplingConfig) -> None:
    import mlflow
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError('Aspect oversampling experiment requires CUDA')
    started = time.monotonic(); config.run_dir.mkdir(parents=True, exist_ok=True)
    with RunLock(config.run_dir / '.experiment.lock'):
        if (config.run_dir / 'results.json').is_file():
            LOGGER.info('Experiment is already complete'); return
        train_rows = read_labeled_csv(config.split_dir / 'train.csv')
        validation_rows = read_labeled_csv(config.split_dir / 'eval.csv')
        manifest = {**{key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
                       for key, value in asdict(config).items()}, 'configurations': list(CONFIGURATIONS),
                    'multipliers': MULTIPLIERS, 'split_sha256': {name: file_sha256(config.split_dir / f'{name}.csv')
                    for name in ('train', 'eval', 'test')}}
        manifest_path = config.run_dir / 'manifest.json'
        if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Experiment manifest mismatch')
        atomic_write_json(manifest_path, manifest)
        _prepare_mlflow_tracking_uri(config.mlflow_tracking_uri); mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, use_fast=True)
        deberta_tokenizer = load_tokenizer(transformers, ThreeExperimentConfig().deberta_model)
        cv = _run_cv(torch, transformers, mlflow, tokenizer, config, train_rows)
        atomic_write_json(config.run_dir / 'cv_results.json', cv)
        final = _final_confirmation(torch, transformers, mlflow, tokenizer, deberta_tokenizer,
                                    config, cv['winner'], train_rows, validation_rows)
        atomic_write_json(config.run_dir / 'pretest_lock.json', {
            'cv_winner': cv['winner'], 'final_winner': final['locked_winner'],
            'promoted': final['promoted'], 'test_accessed': False,
        })
        if 'aspect' in final:
            test_rows = read_labeled_csv(config.split_dir / 'test.csv')
            _final_test(torch, transformers, tokenizer, deberta_tokenizer,
                        config, final, test_rows)
        results = {'cv': cv, 'final': final, 'runtime_seconds': time.monotonic() - started,
                   'test_accessed': 'aspect' in final}
        atomic_write_json(config.run_dir / 'results.json', results)
        atomic_write_json(config.run_dir / 'locked_selection.json', {
            'cv_winner': cv['winner'], 'final_winner': final['locked_winner'],
            'promoted': final['promoted'], 'test_accessed': results['test_accessed']})
        atomic_write_text(config.run_dir / 'report.md', _report(results))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    try:
        run_experiment(config_from_args(argv))
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted; rerun the same command to resume'); return 130
    except Exception:
        LOGGER.exception('Aspect oversampling experiment failed'); return 1
    return 0
