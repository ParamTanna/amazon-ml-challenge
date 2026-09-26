"""Paths and run settings. Override any of them with environment variables (e.g. on Kaggle)."""

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # .../code/business_entity_resolution/src -> repo root

DATA_DIR = Path(os.environ.get("BER_DATA", _ROOT / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK", _ROOT / "work"))
OUT_DIR = Path(os.environ.get("BER_OUT", _ROOT / "output"))
N_JOBS = int(os.environ.get("BER_JOBS", max(1, (os.cpu_count() or 4) - 2)))

CACHE_DIR = WORK_DIR / "cache"
for _d in (WORK_DIR, CACHE_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SPLITS = ("train", "test")
SOURCES = ("source1", "source2", "source3")
