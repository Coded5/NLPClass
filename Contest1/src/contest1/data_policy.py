from __future__ import annotations

import hashlib
from pathlib import Path

from .paths import DATA_DIR


FORBIDDEN_DEV_NAME = "devv_test.csv"
FORBIDDEN_DEV_SHA256 = (
    "233674b10484c23f9a53e5d59a1604c8b15d833d080c8fcd0400f465cd9b3023"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_forbidden_evaluation_path(path: str | Path) -> None:
    candidate = Path(path)
    resolved = candidate.resolve()
    forbidden = (DATA_DIR / FORBIDDEN_DEV_NAME).resolve()
    if (
        candidate.name.casefold() == FORBIDDEN_DEV_NAME
        or resolved.name.casefold() == FORBIDDEN_DEV_NAME
        or resolved == forbidden
    ):
        raise ValueError(f"Refusing to access forbidden evaluation file: {candidate}")
    if resolved.is_file() and _file_sha256(resolved) == FORBIDDEN_DEV_SHA256:
        raise ValueError(f"Refusing to access forbidden evaluation data: {candidate}")
