from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .data import InputItem
from .io_utils import canonical_json, text_sha256
from .prompt import Prediction


class ResumeMismatchError(RuntimeError):
    pass


class RunLockedError(RuntimeError):
    pass


class RunLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self) -> RunLock:
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+', encoding='utf-8')
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            self.handle = None
            raise RunLockedError(f'Another process is using run directory {self.path.parent}') from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f'{os.getpid()}\n')
        self.handle.flush()
        return self

    def __exit__(self, *_: object) -> None:
        if self.handle is None:
            return
        import fcntl

        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


@dataclass(frozen=True)
class StoredItem:
    item_id: str
    text: str
    position: int
    status: str
    attempts: int


class RunStore:
    def __init__(self, path: Path, manifest: dict[str, object]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self._create_schema()
        self._check_manifest(manifest)
        self._recover_interrupted()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def register_items(self, items: Iterable[InputItem]) -> None:
        with self.connection:
            for item in items:
                text_hash = text_sha256(item.text)
                existing = self.connection.execute(
                    'SELECT text_hash, position FROM samples WHERE item_id = ?',
                    (item.item_id,),
                ).fetchone()
                if existing:
                    if existing['text_hash'] != text_hash or existing['position'] != item.position:
                        raise ResumeMismatchError(f'Input item {item.item_id!r} changed since this run started')
                    continue
                self.connection.execute(
                    '''
                    INSERT INTO samples (item_id, text, text_hash, position, status, attempts, updated_at)
                    VALUES (?, ?, ?, ?, 'pending', 0, ?)
                    ''',
                    (item.item_id, item.text, text_hash, item.position, _now()),
                )

    def reset_failed(self) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE samples SET status = 'pending', attempts = 0, updated_at = ? "
                "WHERE status = 'failed'",
                (_now(),),
            )
        return cursor.rowcount

    def runnable_items(self) -> list[StoredItem]:
        rows = self.connection.execute(
            '''
            SELECT item_id, text, position, status, attempts
            FROM samples
            WHERE status = 'pending'
            ORDER BY position
            ''',
        ).fetchall()
        return [StoredItem(**dict(row)) for row in rows]

    def begin_attempt(self, item_id: str) -> None:
        with self.connection:
            self.connection.execute(
                '''
                UPDATE samples
                SET status = 'in_progress', attempts = attempts + 1, updated_at = ?
                WHERE item_id = ?
                ''',
                (_now(), item_id),
            )

    def record_error(self, item_id: str, error: str, raw_response: str | None = None) -> None:
        with self.connection:
            self.connection.execute(
                '''
                UPDATE samples
                SET last_error = ?, raw_response = COALESCE(?, raw_response), updated_at = ?
                WHERE item_id = ?
                ''',
                (error, raw_response, _now(), item_id),
            )

    def mark_completed(
        self,
        item_id: str,
        raw_response: str,
        predictions: list[Prediction],
    ) -> None:
        predictions_json = json.dumps(
            [prediction.as_dict() for prediction in predictions],
            ensure_ascii=False,
            separators=(',', ':'),
        )
        with self.connection:
            self.connection.execute(
                '''
                UPDATE samples
                SET status = 'completed', raw_response = ?, predictions_json = ?,
                    last_error = NULL, updated_at = ?
                WHERE item_id = ?
                ''',
                (raw_response, predictions_json, _now(), item_id),
            )

    def mark_failed(self, item_id: str, error: str) -> None:
        with self.connection:
            self.connection.execute(
                '''
                UPDATE samples
                SET status = 'failed', last_error = ?, updated_at = ?
                WHERE item_id = ?
                ''',
                (error, _now(), item_id),
            )

    def completed_predictions(self) -> list[tuple[str, list[dict[str, str]]]]:
        rows = self.connection.execute(
            '''
            SELECT item_id, predictions_json
            FROM samples
            WHERE status = 'completed'
            ORDER BY position
            ''',
        ).fetchall()
        return [(row['item_id'], json.loads(row['predictions_json'])) for row in rows]

    def summary(self) -> dict[str, int]:
        status_counts = {
            row['status']: row['count']
            for row in self.connection.execute(
                'SELECT status, COUNT(*) AS count FROM samples GROUP BY status'
            )
        }
        total_attempts = self.connection.execute(
            'SELECT COALESCE(SUM(attempts), 0) AS attempts FROM samples'
        ).fetchone()['attempts']
        return {
            'total': sum(status_counts.values()),
            'completed': status_counts.get('completed', 0),
            'failed': status_counts.get('failed', 0),
            'pending': status_counts.get('pending', 0) + status_counts.get('in_progress', 0),
            'attempts': total_attempts,
        }

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                'CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)'
            )
            self.connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS samples (
                    item_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    position INTEGER NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    raw_response TEXT,
                    predictions_json TEXT,
                    last_error TEXT,
                    updated_at TEXT NOT NULL
                )
                '''
            )

    def _check_manifest(self, manifest: dict[str, object]) -> None:
        serialized = canonical_json(manifest)
        existing = self.connection.execute(
            "SELECT value FROM metadata WHERE key = 'manifest'"
        ).fetchone()
        if existing is None:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO metadata (key, value) VALUES ('manifest', ?)",
                    (serialized,),
                )
        elif existing['value'] != serialized:
            raise ResumeMismatchError(
                'Run configuration or input changed; choose a new --run-dir for this run'
            )

    def _recover_interrupted(self) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE samples SET status = 'pending', updated_at = ? WHERE status = 'in_progress'",
                (_now(),),
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
