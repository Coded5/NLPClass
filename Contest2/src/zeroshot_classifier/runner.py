from __future__ import annotations

import csv
import io
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .backends import LLMBackend, PermanentBackendError, TransientBackendError
from .checkpoint import RunStore, StoredItem
from .data import InputItem
from .io_utils import atomic_write_json, atomic_write_text
from .prompt import OUTPUT_SCHEMA, ResponseValidationError, messages_for, parse_response


@dataclass(frozen=True)
class RunnerSettings:
    max_attempts: int = 3
    initial_backoff: float = 2.0
    max_backoff: float = 30.0
    retry_failed: bool = False


def run_classification(
    items: list[InputItem],
    backend: LLMBackend,
    store: RunStore,
    run_dir: Path,
    settings: RunnerSettings,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    if settings.max_attempts < 1:
        raise ValueError('max_attempts must be at least 1')
    store.register_items(items)
    if settings.retry_failed:
        reset_count = store.reset_failed()
        if reset_count:
            print(f'Reset {reset_count} failed item(s) for retry.')

    runnable = store.runnable_items()
    print(f'{len(runnable)} item(s) to process; completed items will be skipped.')
    try:
        for index, item in enumerate(runnable, start=1):
            _classify_item(item, backend, store, settings, sleep)
            if index % 10 == 0 or index == len(runnable):
                summary = store.summary()
                print(
                    f'Progress {index}/{len(runnable)} | '
                    f'completed={summary["completed"]} failed={summary["failed"]}'
                )
    finally:
        export_predictions(store, run_dir / 'predictions.csv')
        atomic_write_json(run_dir / 'summary.json', store.summary())
    return store.summary()


def export_predictions(store: RunStore, output_path: Path) -> None:
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(('id', 'aspectCategory', 'polarity'))
    for item_id, predictions in store.completed_predictions():
        for prediction in predictions:
            writer.writerow((item_id, prediction['aspectCategory'], prediction['polarity']))
    atomic_write_text(output_path, output.getvalue())


def _classify_item(
    item: StoredItem,
    backend: LLMBackend,
    store: RunStore,
    settings: RunnerSettings,
    sleep: Callable[[float], None],
) -> None:
    final_error = 'Classification failed'
    remaining_attempts = settings.max_attempts - item.attempts
    if remaining_attempts <= 0:
        store.mark_failed(item.item_id, f'Attempt limit of {settings.max_attempts} reached')
        return
    for attempt in range(remaining_attempts):
        store.begin_attempt(item.item_id)
        raw_response: str | None = None
        try:
            raw_response = backend.complete(messages_for(item.text), OUTPUT_SCHEMA)
            predictions = parse_response(raw_response)
            store.mark_completed(item.item_id, raw_response, predictions)
            return
        except PermanentBackendError as error:
            final_error = str(error)
            store.record_error(item.item_id, final_error, raw_response)
            break
        except (TransientBackendError, ResponseValidationError) as error:
            final_error = str(error)
            store.record_error(item.item_id, final_error, raw_response)
        except Exception as error:
            final_error = f'Unexpected {type(error).__name__}: {error}'
            store.record_error(item.item_id, final_error, raw_response)

        if attempt + 1 < remaining_attempts:
            delay = min(settings.max_backoff, settings.initial_backoff * (2**attempt))
            sleep(delay * random.uniform(0.8, 1.2))
    store.mark_failed(item.item_id, final_error)
