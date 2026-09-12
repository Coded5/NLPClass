from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .aspect_oversampling_experiment import grouped_aspect_folds
from .data import LabeledRow, read_labeled_csv
from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .labels import ASPECTS, POLARITIES


MISC = 'anecdotes/miscellaneous'
ASSESSMENTS = (
    'clear model error',
    'ambiguous annotation',
    'suspected annotation error',
    'insufficient evidence',
)


@dataclass(frozen=True)
class AuditConfig:
    train: Path = Path('artifacts/training/roberta-aspect-exp1/splits/train.csv')
    aspect_run: Path = Path('artifacts/experiments/aspect-misc-oversampling-cv-v1')
    polarity_run: Path = Path('artifacts/experiments/polarity-oversampling-cv-v1')
    output_dir: Path = Path('artifacts/experiments/oof-label-audit-v1')
    sample_size: int = 100
    seed: int = 42
    assessments: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build a review set from saved out-of-fold ABSA errors.')
    parser.add_argument('--train', type=Path, default=AuditConfig.train)
    parser.add_argument('--aspect-run', type=Path, default=AuditConfig.aspect_run)
    parser.add_argument('--polarity-run', type=Path, default=AuditConfig.polarity_run)
    parser.add_argument('--output-dir', type=Path, default=AuditConfig.output_dir)
    parser.add_argument('--sample-size', type=int, default=AuditConfig.sample_size)
    parser.add_argument('--seed', type=int, default=AuditConfig.seed)
    parser.add_argument('--assessments', type=Path)
    return parser


def config_from_args(argv: list[str] | None = None) -> AuditConfig:
    config = AuditConfig(**vars(build_parser().parse_args(argv)))
    if config.sample_size < 1:
        raise ValueError('sample_size must be positive')
    return config


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f'Required artifact not found: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _id_key(item_id: str) -> tuple[int, int | str]:
    return (0, int(item_id)) if item_id.isdigit() else (1, item_id)


def _rows_by_id(rows: Sequence[LabeledRow]) -> dict[str, list[LabeledRow]]:
    grouped: dict[str, list[LabeledRow]] = defaultdict(list)
    for row in rows:
        grouped[row.item_id].append(row)
    return dict(grouped)


def _validate_partition(folds: Sequence[set[str]], expected: set[str], label: str) -> dict[str, int]:
    seen: set[str] = set()
    membership: dict[str, int] = {}
    for fold, values in enumerate(folds, 1):
        overlap = seen & values
        if overlap:
            raise ValueError(f'{label} fold membership overlaps: {sorted(overlap, key=_id_key)[:5]}')
        for item_id in values:
            membership[item_id] = fold
        seen.update(values)
    if seen != expected:
        missing = expected - seen
        extra = seen - expected
        raise ValueError(f'{label} fold coverage mismatch: missing={len(missing)} extra={len(extra)}')
    return membership


def _gold_aspects(rows_by_id: dict[str, list[LabeledRow]]) -> dict[str, set[str]]:
    return {item_id: {row.aspect for row in values} for item_id, values in rows_by_id.items()}


def load_aspect_predictions(
    run_dir: Path, train_path: Path, rows: Sequence[LabeledRow]
) -> tuple[dict[str, dict[str, set[str]]], dict[str, int], dict[str, Any]]:
    manifest = _load_json(run_dir / 'manifest.json')
    if manifest.get('split_sha256', {}).get('train') != file_sha256(train_path):
        raise ValueError('Aspect manifest train hash does not match audit input')
    ids = set(_rows_by_id(rows))
    folds = grouped_aspect_folds(rows, int(manifest['folds']), int(manifest['seed']))
    membership = _validate_partition(folds, ids, 'aspect')
    configurations = tuple(manifest['configurations'])
    predictions: dict[str, dict[str, set[str]]] = {}
    for configuration in configurations:
        by_id = {item_id: set() for item_id in ids}
        observed: set[str] = set()
        for fold in range(1, int(manifest['folds']) + 1):
            payload = _load_json(run_dir / 'cv' / configuration / f'fold-{fold}' / 'fold_metrics.json')
            expected_fold = folds[fold - 1]
            predicted_ids = {str(item_id) for item_id, _ in payload['predictions']}
            if predicted_ids != expected_fold:
                raise ValueError(f'Aspect {configuration} fold {fold} heldout coverage mismatch')
            observed.update(expected_fold)
            for item_id, aspect in payload['predictions']:
                item_id = str(item_id)
                if aspect not in ASPECTS:
                    raise ValueError(f'Unknown aspect prediction: {aspect}')
                by_id[item_id].add(aspect)
        if observed != ids:
            raise ValueError(f'Aspect {configuration} predictions do not cover every fold')
        predictions[configuration] = by_id
    return predictions, membership, manifest


def load_polarity_predictions(
    run_dir: Path, rows: Sequence[LabeledRow]
) -> tuple[dict[str, dict[tuple[str, str], str]], dict[str, int], dict[str, Any]]:
    manifest = _load_json(run_dir / 'manifest.json')
    train_path = Path(manifest['train']); validation_path = Path(manifest['validation'])
    if manifest.get('train_sha256') != file_sha256(train_path):
        raise ValueError('Polarity manifest train hash does not match its source file')
    if manifest.get('validation_sha256') != file_sha256(validation_path):
        raise ValueError('Polarity manifest validation hash does not match its source file')
    all_rows = read_labeled_csv(train_path) + read_labeled_csv(validation_path)
    all_ids = set(_rows_by_id(all_rows))
    folds = [set(values) for values in manifest['fold_ids']]
    all_membership = _validate_partition(folds, all_ids, 'polarity')
    train_ids = set(_rows_by_id(rows))
    expected_keys = {(row.item_id, row.aspect) for row in rows}
    expected_gold = {(row.item_id, row.aspect): row.polarity for row in rows}
    expected_text = {row.item_id: row.text for row in rows}
    configurations = tuple(manifest['configurations'])
    predictions: dict[str, dict[tuple[str, str], str]] = {}
    for configuration in configurations:
        values = _load_json(run_dir / 'oof' / f'{configuration}.json')
        by_key: dict[tuple[str, str], str] = {}
        for value in values:
            item_id = str(value['id'])
            if item_id not in train_ids:
                continue
            key = (item_id, value['aspect'])
            if key in by_key:
                if by_key[key] != value['predicted']:
                    raise ValueError(f'Conflicting duplicate polarity prediction: {key}')
                continue
            if value['aspect'] not in ASPECTS or value['predicted'] not in POLARITIES:
                raise ValueError(f'Unknown polarity label in {key}')
            if value.get('gold') != expected_gold[key] or value.get('text') != expected_text[item_id]:
                raise ValueError(f'Polarity source does not align with audit input: {key}')
            by_key[key] = value['predicted']
        if set(by_key) != expected_keys:
            raise ValueError(f'Polarity {configuration} coverage mismatch')
        predictions[configuration] = by_key
    return predictions, {item_id: all_membership[item_id] for item_id in train_ids}, manifest


def build_aspect_candidates(
    rows: Sequence[LabeledRow], predictions: dict[str, dict[str, set[str]]], membership: dict[str, int]
) -> list[dict[str, Any]]:
    grouped = _rows_by_id(rows); gold = _gold_aspects(grouped)
    configurations = tuple(predictions)
    candidates = []
    for item_id, values in grouped.items():
        reference = predictions['reference'][item_id]
        if reference == gold[item_id]:
            continue
        misc_error = MISC in (gold[item_id] ^ reference)
        if misc_error and MISC in gold[item_id] and len(gold[item_id]) > 1:
            bucket = 'aspect_multi_misc'
        elif misc_error:
            bucket = 'aspect_single_misc'
        else:
            bucket = 'aspect_other'
        candidates.append({
            'case_key': f'aspect:{item_id}', 'task': 'aspect', 'bucket': bucket,
            'id': item_id, 'text': values[0].text, 'gold_aspects': sorted(gold[item_id], key=ASPECTS.index),
            'reference_prediction': sorted(reference, key=ASPECTS.index),
            'configuration_predictions': {name: sorted(predictions[name][item_id], key=ASPECTS.index)
                                          for name in configurations},
            'error_persistence': sum(predictions[name][item_id] != gold[item_id] for name in configurations),
            'configuration_count': len(configurations), 'fold': membership[item_id],
            'supporting_text': '', 'assessment': 'insufficient evidence',
            'rationale': '', 'proposed_label': '', 'needs_user_adjudication': True,
        })
    return candidates


def build_polarity_candidates(
    rows: Sequence[LabeledRow], predictions: dict[str, dict[tuple[str, str], str]], membership: dict[str, int]
) -> list[dict[str, Any]]:
    configurations = tuple(predictions); candidates = []; seen: set[tuple[str, str, str]] = set()
    for row in rows:
        annotation = (row.item_id, row.aspect, row.polarity)
        if annotation in seen:
            continue
        seen.add(annotation)
        key = (row.item_id, row.aspect); reference = predictions['reference'][key]
        if reference == row.polarity:
            continue
        bucket = f'polarity_{row.polarity}' if row.polarity in {'neutral', 'conflict'} else 'polarity_other'
        candidates.append({
            'case_key': f'polarity:{row.item_id}:{row.aspect}', 'task': 'polarity', 'bucket': bucket,
            'id': row.item_id, 'text': row.text, 'aspect': row.aspect, 'gold_polarity': row.polarity,
            'reference_prediction': reference,
            'configuration_predictions': {name: predictions[name][key] for name in configurations},
            'error_persistence': sum(predictions[name][key] != row.polarity for name in configurations),
            'configuration_count': len(configurations), 'fold': membership[row.item_id],
            'supporting_text': '', 'assessment': 'insufficient evidence',
            'rationale': '', 'proposed_label': '', 'needs_user_adjudication': True,
        })
    return candidates


def select_cases(candidates: Sequence[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    quotas = (
        ('aspect_multi_misc', 20), ('aspect_single_misc', 15), ('aspect_other', 15),
        ('polarity_conflict', 20), ('polarity_neutral', 20), ('polarity_other', 10),
    )
    ranked = sorted(candidates, key=lambda value: (-value['error_persistence'], _id_key(value['id']), value['case_key']))
    selected: list[dict[str, Any]] = []; used_ids: set[str] = set(); used_keys: set[str] = set()
    for bucket, limit in quotas:
        matches = [value for value in ranked if value['bucket'] == bucket and value['id'] not in used_ids]
        for value in matches[:limit]:
            selected.append(dict(value)); used_ids.add(value['id']); used_keys.add(value['case_key'])
    if len(selected) < sample_size:
        for value in ranked:
            if value['case_key'] in used_keys or value['id'] in used_ids:
                continue
            selected.append(dict(value)); used_ids.add(value['id']); used_keys.add(value['case_key'])
            if len(selected) == sample_size:
                break
    return selected[:sample_size]


def apply_assessments(cases: list[dict[str, Any]], path: Path | None) -> None:
    if path is None:
        return
    values = _load_json(path)
    if not isinstance(values, list):
        raise ValueError('Assessments must be a JSON list')
    by_key = {value['case_key']: value for value in values}
    unknown = set(by_key) - {case['case_key'] for case in cases}
    if unknown:
        raise ValueError(f'Assessments contain unknown case keys: {sorted(unknown)[:5]}')
    for case in cases:
        if case['case_key'] not in by_key:
            continue
        assessment = by_key[case['case_key']]
        if assessment.get('assessment') not in ASSESSMENTS:
            raise ValueError(f"Invalid assessment for {case['case_key']}")
        for field in ('supporting_text', 'assessment', 'rationale', 'proposed_label', 'needs_user_adjudication',
                      'previous_assessment', 'review_revision', 'review_status'):
            if field in assessment:
                case[field] = assessment[field]


def _csv(cases: Sequence[dict[str, Any]]) -> str:
    fields = ('case_key', 'task', 'bucket', 'id', 'text', 'aspect', 'gold_aspects', 'gold_polarity',
              'reference_prediction', 'configuration_predictions', 'error_persistence', 'configuration_count',
              'fold', 'supporting_text', 'assessment', 'rationale', 'proposed_label', 'needs_user_adjudication',
              'previous_assessment', 'review_revision', 'review_status')
    output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader()
    for case in cases:
        writer.writerow({field: json.dumps(case.get(field), ensure_ascii=False) if isinstance(case.get(field), (dict, list))
                         else case.get(field, '') for field in fields})
    return output.getvalue()


def _report(candidates: Sequence[dict[str, Any]], cases: Sequence[dict[str, Any]]) -> str:
    full = Counter(value['bucket'] for value in candidates); selected = Counter(value['bucket'] for value in cases)
    assessments = Counter(value['assessment'] for value in cases)
    by_bucket = {
        bucket: Counter(value['assessment'] for value in cases if value['bucket'] == bucket)
        for bucket in {value['bucket'] for value in cases}
    }
    complete = all(value['rationale'] for value in cases)
    lines = [
        '# Out-of-fold ABSA label audit', '',
        'This diagnostic uses saved historical out-of-fold predictions and only IDs from the original training split.',
        'Aspect and polarity predictions came from different CV experiments, so their error rates are not directly comparable.', '',
        '## Coverage', '', '| Bucket | Full error pool | Reviewed sample |', '|---|---:|---:|',
    ]
    for bucket in ('aspect_multi_misc', 'aspect_single_misc', 'aspect_other',
                   'polarity_conflict', 'polarity_neutral', 'polarity_other'):
        lines.append(f'| {bucket} | {full[bucket]} | {selected[bucket]} |')
    lines.extend(['', '## First-pass assessments', '', '| Assessment | Count |', '|---|---:|'])
    for label in ASSESSMENTS:
        lines.append(f'| {label} | {assessments[label]} |')
    lines.extend(['', '| Bucket | Clear model error | Ambiguous | Suspected annotation error | Insufficient |',
                  '|---|---:|---:|---:|---:|'])
    for bucket in ('aspect_multi_misc', 'aspect_single_misc', 'aspect_other',
                   'polarity_conflict', 'polarity_neutral', 'polarity_other'):
        counts = by_bucket.get(bucket, Counter())
        lines.append(f"| {bucket} | {counts['clear model error']} | {counts['ambiguous annotation']} | "
                     f"{counts['suspected annotation error']} | {counts['insufficient evidence']} |")
    lines.extend(['', f'Assessment complete: **{str(complete).lower()}**.', '',
                  'The selected sample is deliberately enriched for difficult minority cases. Its category rates do not estimate dataset-wide label quality.', '',
                  '## Findings and next step', '',
                  f"The current assistant review assigns {assessments['clear model error']} of {len(cases)} cases to clear model error, "
                  f"{assessments['ambiguous annotation']} to ambiguous annotation, "
                  f"{assessments['suspected annotation error']} to suspected annotation error, and "
                  f"{assessments['insufficient evidence']} to insufficient evidence. These are provisional judgments, not adjudicated labels.", '',
                  '## Revision standard', '',
                  'The original 37/30/32/1 breakdown was too confident. Revision 2 preserves previous categories in the worksheet. '
                  'No annotation manual was located in the repository search, so these criteria are a provisional review rubric, not official dataset rules.', '',
                  '- Mentioning food or money alone does not establish a separately evaluated food or price aspect.',
                  '- Tie sentiment to its target; opposite sentiments about different aspects do not automatically constitute conflict.',
                  '- Lack of an explicit overall judgment does not invalidate anecdotes/miscellaneous: personal narrative may belong there.',
                  '- Questions, comparisons, and implicit evaluations can convey sentiment. Missing context does not automatically establish neutral.',
                  '- Use ambiguous annotation when plausible readings remain; this category does not mean that the gold label is wrong.',
                  '- Reserve suspected annotation error for a strong textual mismatch and keep any alternative as a hypothesis requiring adjudication.', '',
                  'Clear examples still show missed secondary aspects (IDs 3 and 181), contrasting evaluations of the same aspect '
                  '(IDs 763, 1842, and 1903), and sentiment transferred from ordering difficulty to food (ID 484). '
                  'These observations support a clause-to-aspect attribution weakness, but do not establish its prevalence or prove the cause of the performance plateau.', '',
                  '**Next step:** obtain the original annotation guidance and independently review the uncertain cases, ideally blinded to predictions. '
                  'Keep proposed alternatives out of training until adjudicated. This selected sample cannot establish dataset-wide label noise '
                  'or justify concluding that annotation quality caused the plateau.', '',
                  '## Provenance limitations', '',
                  '- Aspect OOF models used only the original training split.',
                  '- Polarity OOF models used the original training and validation splits; this audit filters their held-out predictions back to training IDs.',
                  '- Saved class probabilities were unavailable. Error persistence counts configurations that repeat an error and must not be described as confidence.',
                  '- The historical test split is not read by this audit.', ''])
    return '\n'.join(lines)


def run(config: AuditConfig) -> list[dict[str, Any]]:
    rows = read_labeled_csv(config.train); train_ids = set(_rows_by_id(rows))
    aspect_predictions, aspect_membership, aspect_manifest = load_aspect_predictions(
        config.aspect_run, config.train, rows
    )
    polarity_predictions, polarity_membership, polarity_manifest = load_polarity_predictions(config.polarity_run, rows)
    candidates = build_aspect_candidates(rows, aspect_predictions, aspect_membership)
    candidates.extend(build_polarity_candidates(rows, polarity_predictions, polarity_membership))
    cases = select_cases(candidates, config.sample_size); apply_assessments(cases, config.assessments)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        'train_sha256': file_sha256(config.train), 'train_ids': len(train_ids),
        'aspect_source_manifest': str(config.aspect_run / 'manifest.json'),
        'polarity_source_manifest': str(config.polarity_run / 'manifest.json'),
        'aspect_source_manifest_sha256': file_sha256(config.aspect_run / 'manifest.json'),
        'polarity_source_manifest_sha256': file_sha256(config.polarity_run / 'manifest.json'),
        'aspect_configurations': aspect_manifest['configurations'],
        'polarity_configurations': polarity_manifest['configurations'],
        'labels': {'aspects': list(ASPECTS), 'polarities': list(POLARITIES)},
        'selection': 'error persistence descending, review ID ascending; unique IDs across tasks',
        'historical_test_accessed': False,
    }
    normalized = {'aspect': {name: {item_id: sorted(labels, key=ASPECTS.index) for item_id, labels in values.items()}
                             for name, values in aspect_predictions.items()},
                  'polarity': {name: {f'{item_id}\x1f{aspect}': label for (item_id, aspect), label in values.items()}
                               for name, values in polarity_predictions.items()}}
    atomic_write_json(config.output_dir / 'manifest.json', manifest)
    atomic_write_json(config.output_dir / 'normalized_predictions.json', normalized)
    atomic_write_json(config.output_dir / 'full_error_pool.json', candidates)
    atomic_write_json(config.output_dir / 'review_cases.json', cases)
    atomic_write_text(config.output_dir / 'review_cases.csv', _csv(cases))
    atomic_write_text(config.output_dir / 'report.md', _report(candidates, cases))
    return cases


def main(argv: list[str] | None = None) -> int:
    run(config_from_args(argv)); return 0
