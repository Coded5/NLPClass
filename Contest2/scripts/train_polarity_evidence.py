#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

repository_root = Path(__file__).resolve().parents[1]
source_root = repository_root / 'src'
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from zeroshot_classifier.polarity_evidence_experiment import main

if __name__ == '__main__':
    raise SystemExit(main())
