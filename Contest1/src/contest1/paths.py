import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("CONTEST1_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = PROJECT_ROOT / "train"
