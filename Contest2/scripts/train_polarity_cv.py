#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root / 'src') not in sys.path:
    sys.path.insert(0, str(root / 'src'))

from zeroshot_classifier.polarity_cv_experiment import main

if __name__ == '__main__':
    raise SystemExit(main())
