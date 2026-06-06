#!/usr/bin/env python3
"""Deprecated: use training/moss-realtime/scripts/legacy/prune_loli15s_teacher_dataset.py (500ms default)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parents[3] / "moss-realtime/scripts/legacy/prune_loli15s_teacher_dataset.py"
if not _SHARED.is_file():
    raise SystemExit(f"Missing shared prune script: {_SHARED}")
sys.argv[0] = str(_SHARED)
runpy.run_path(str(_SHARED), run_name="__main__")
