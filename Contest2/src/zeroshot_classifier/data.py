from __future__ import annotations

import csv
import io
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .io_utils import atomic_write_json, atomic_write_text, file_sha256
from .labels import ASPECTS, POLARITIES


REQUIRED_COLUMNS = ('id', 'text', 'aspectCategory', 'polarity')


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class LabeledRow:
    item_id: str
    text: str
    aspect: str
    polarity: str
    position: int

    @property
    def joint_label(self) -> str:
        return f'{self.aspect}\x1f{self.polarity}'


@dataclass(frozen=True)
class InputItem:
    item_id: str
    text: str
    position: int


@dataclass(frozen=True)
class SplitResult:
    train_rows: tuple[LabeledRow, ...]
    test_rows: tuple[LabeledRow, ...]
    train_ids: frozenset[str]
    test_ids: frozenset[str]


def read_labeled_csv(path: Path) -> list[LabeledRow]:
    if not path.is_file():
        raise DataValidationError(f'Dataset not found: {path}')

    rows: list[LabeledRow] = []
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise DataValidationError(f'Missing required columns: {", ".join(sorted(missing))}')

        for position, raw in enumerate(reader):
            item_id = raw['id'].strip()
            text = raw['text'].strip()
            aspect = raw['aspectCategory'].strip()
            polarity = raw['polarity'].strip()
            row_number = position + 2
            if not item_id:
                raise DataValidationError(f'Empty id at CSV row {row_number}')
            if not text:
                raise DataValidationError(f'Empty text at CSV row {row_number}')
            if aspect not in ASPECTS:
                raise DataValidationError(f'Invalid aspectCategory {aspect!r} at CSV row {row_number}')
            if polarity not in POLARITIES:
                raise DataValidationError(f'Invalid polarity {polarity!r} at CSV row {row_number}')
            rows.append(LabeledRow(item_id, text, aspect, polarity, position))

    if not rows:
        raise DataValidationError('Dataset contains no rows')
    _validate_id_texts(rows)
    return rows


def read_input_items(path: Path) -> list[InputItem]:
    if not path.is_file():
        raise DataValidationError(f'Input not found: {path}')

    items: list[InputItem] = []
    seen: dict[str, str] = {}
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        missing = {'id', 'text'} - set(reader.fieldnames or ())
        if missing:
            raise DataValidationError(f'Missing required columns: {", ".join(sorted(missing))}')
        for raw in reader:
            item_id = raw['id'].strip()
            text = raw['text'].strip()
            if not item_id or not text:
                raise DataValidationError('Input contains an empty id or text')
            if item_id in seen:
                if seen[item_id] != text:
                    raise DataValidationError(f'ID {item_id!r} maps to more than one text')
                continue
            seen[item_id] = text
            items.append(InputItem(item_id, text, len(items)))
    if not items:
        raise DataValidationError('Input contains no rows')
    return items


def stratified_group_split(
    rows: Sequence[LabeledRow],
    test_ratio: float = 0.1,
    seed: int = 42,
) -> SplitResult:
    if not 0 < test_ratio < 1:
        raise ValueError('test_ratio must be between 0 and 1')
    _validate_id_texts(rows)

    grouped: dict[str, list[LabeledRow]] = defaultdict(list)
    for row in rows:
        grouped[row.item_id].append(row)
    if len(grouped) < 2:
        raise DataValidationError('At least two unique IDs are required for a split')

    strata: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for item_id, group in grouped.items():
        signature = tuple(sorted(row.joint_label for row in group))
        strata[signature].append(item_id)

    randomizer = random.Random(seed)
    for signature in sorted(strata):
        randomizer.shuffle(strata[signature])

    target_test_groups = max(1, min(len(grouped) - 1, round(len(grouped) * test_ratio)))
    allocation = {signature: math.floor(len(ids) * test_ratio) for signature, ids in strata.items()}
    seats_left = target_test_groups - sum(allocation.values())
    ranked_signatures = sorted(
        strata,
        key=lambda signature: (
            -(len(strata[signature]) * test_ratio - allocation[signature]),
            signature,
        ),
    )
    for signature in ranked_signatures[:seats_left]:
        allocation[signature] += 1

    test_ids = {
        item_id
        for signature, ids in strata.items()
        for item_id in ids[:allocation[signature]]
    }
    test_ids = _ensure_joint_label_coverage(grouped, test_ids, test_ratio)
    test_ids = _improve_distribution(grouped, test_ids, test_ratio)
    train_ids = set(grouped) - test_ids
    train_rows = tuple(row for row in rows if row.item_id in train_ids)
    test_rows = tuple(row for row in rows if row.item_id in test_ids)
    return SplitResult(train_rows, test_rows, frozenset(train_ids), frozenset(test_ids))


def write_split(
    source_path: Path,
    output_dir: Path,
    test_ratio: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, object]:
    rows = read_labeled_csv(source_path)
    split = stratified_group_split(rows, test_ratio=test_ratio, seed=seed)
    train_path = output_dir / 'train.csv'
    test_path = output_dir / 'test.csv'
    manifest_path = output_dir / 'manifest.json'
    existing = [path for path in (train_path, test_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        names = ', '.join(str(path) for path in existing)
        raise FileExistsError(f'Split output already exists: {names}; pass --overwrite to replace it')

    manifest = {
        'source': str(source_path),
        'source_sha256': file_sha256(source_path),
        'seed': seed,
        'requested_test_ratio': test_ratio,
        'actual_test_row_ratio': len(split.test_rows) / len(rows),
        'actual_test_id_ratio': len(split.test_ids) / (len(split.train_ids) + len(split.test_ids)),
        'rows': {'all': len(rows), 'train': len(split.train_rows), 'test': len(split.test_rows)},
        'ids': {
            'all': len(split.train_ids) + len(split.test_ids),
            'train': len(split.train_ids),
            'test': len(split.test_ids),
        },
        'joint_label_distribution': {
            'all': _label_counts(rows),
            'train': _label_counts(split.train_rows),
            'test': _label_counts(split.test_rows),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(train_path, _rows_to_csv(split.train_rows))
    atomic_write_text(test_path, _rows_to_csv(split.test_rows))
    atomic_write_json(manifest_path, manifest)
    return manifest


def _validate_id_texts(rows: Iterable[LabeledRow]) -> None:
    texts: dict[str, str] = {}
    for row in rows:
        known_text = texts.setdefault(row.item_id, row.text)
        if known_text != row.text:
            raise DataValidationError(f'ID {row.item_id!r} maps to more than one text')


def _rows_to_csv(rows: Iterable[LabeledRow]) -> str:
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(REQUIRED_COLUMNS)
    for row in rows:
        writer.writerow((row.item_id, row.text, row.aspect, row.polarity))
    return output.getvalue()


def _label_counts(rows: Iterable[LabeledRow]) -> dict[str, int]:
    counts = Counter(f'{row.aspect}|{row.polarity}' for row in rows)
    return dict(sorted(counts.items()))


def _ensure_joint_label_coverage(
    grouped: dict[str, list[LabeledRow]],
    initial_test_ids: set[str],
    test_ratio: float,
) -> set[str]:
    test_ids = set(initial_test_ids)
    group_counts = {
        item_id: Counter(row.joint_label for row in group)
        for item_id, group in grouped.items()
    }
    total_counts = sum(group_counts.values(), Counter())
    test_counts = sum((group_counts[item_id] for item_id in test_ids), Counter())
    test_rows = sum(len(grouped[item_id]) for item_id in test_ids)
    target_rows = sum(len(group) for group in grouped.values()) * test_ratio

    missing_labels = sorted(
        label
        for label, count in total_counts.items()
        if count * test_ratio >= 0.5 and test_counts[label] == 0
    )
    for missing_label in missing_labels:
        protected_labels = {
            label
            for label, count in total_counts.items()
            if count * test_ratio >= 0.5 and test_counts[label] > 0
        }
        additions = [
            item_id
            for item_id in grouped.keys() - test_ids
            if group_counts[item_id][missing_label]
        ]
        best_swap: tuple[float, str, str, Counter[str], int] | None = None
        for added_id in additions:
            for removed_id in test_ids:
                candidate_counts = test_counts - group_counts[removed_id] + group_counts[added_id]
                if any(
                    candidate_counts[label] == 0
                    for label in protected_labels | {missing_label}
                ):
                    continue
                candidate_rows = test_rows - len(grouped[removed_id]) + len(grouped[added_id])
                error = _distribution_error(
                    candidate_counts,
                    total_counts,
                    candidate_rows,
                    target_rows,
                    test_ratio,
                )
                candidate = (error, added_id, removed_id, candidate_counts, candidate_rows)
                if best_swap is None or candidate[:3] < best_swap[:3]:
                    best_swap = candidate
        if best_swap is None:
            continue
        _, added_id, removed_id, test_counts, test_rows = best_swap
        test_ids.remove(removed_id)
        test_ids.add(added_id)
    return test_ids


def _improve_distribution(
    grouped: dict[str, list[LabeledRow]],
    initial_test_ids: set[str],
    test_ratio: float,
    max_passes: int = 20,
) -> set[str]:
    test_ids = set(initial_test_ids)
    group_counts = {
        item_id: Counter(row.joint_label for row in group)
        for item_id, group in grouped.items()
    }
    total_counts = sum(group_counts.values(), Counter())
    test_counts = sum((group_counts[item_id] for item_id in test_ids), Counter())
    test_rows = sum(len(grouped[item_id]) for item_id in test_ids)
    target_rows = sum(len(group) for group in grouped.values()) * test_ratio
    current_error = _distribution_error(
        test_counts, total_counts, test_rows, target_rows, test_ratio
    )

    for _ in range(max_passes):
        best: tuple[float, str, str, Counter[str], int] | None = None
        train_ids = grouped.keys() - test_ids
        for removed_id in sorted(test_ids):
            removed_counts = group_counts[removed_id]
            for added_id in sorted(train_ids):
                candidate_rows = test_rows - len(grouped[removed_id]) + len(grouped[added_id])
                candidate_error = current_error + _swap_error_delta(
                    test_counts,
                    total_counts,
                    removed_counts,
                    group_counts[added_id],
                    test_rows,
                    candidate_rows,
                    target_rows,
                    test_ratio,
                )
                if candidate_error >= current_error - 1e-12:
                    continue
                candidate_counts = test_counts - removed_counts + group_counts[added_id]
                candidate = (
                    candidate_error,
                    added_id,
                    removed_id,
                    candidate_counts,
                    candidate_rows,
                )
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        if best is None:
            break
        current_error, added_id, removed_id, test_counts, test_rows = best
        test_ids.remove(removed_id)
        test_ids.add(added_id)
    return test_ids


def _distribution_error(
    observed: Counter[str],
    totals: Counter[str],
    observed_rows: int,
    target_rows: float,
    test_ratio: float,
) -> float:
    label_error = sum(
        ((observed[label] - total * test_ratio) ** 2) / max(total * test_ratio, 1.0)
        for label, total in totals.items()
    )
    row_error = ((observed_rows - target_rows) ** 2) / max(target_rows, 1.0)
    return label_error + row_error


def _swap_error_delta(
    observed: Counter[str],
    totals: Counter[str],
    removed: Counter[str],
    added: Counter[str],
    old_rows: int,
    new_rows: int,
    target_rows: float,
    test_ratio: float,
) -> float:
    delta = (
        (new_rows - target_rows) ** 2 - (old_rows - target_rows) ** 2
    ) / max(target_rows, 1.0)
    for label in removed.keys() | added.keys():
        target = totals[label] * test_ratio
        old_count = observed[label]
        new_count = old_count - removed[label] + added[label]
        delta += (
            (new_count - target) ** 2 - (old_count - target) ** 2
        ) / max(target, 1.0)
    return delta
