#!/usr/bin/env python3
"""Loli voice-clone ECAPA bench (delegates to major_03 script with loli defaults)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[3]))
LOLI = Path(os.environ.get("MOSS_RT_TRAIN_DIR", ROOT / "training/loli_15s"))
MAJOR_BENCH = ROOT / "training/major_03/scripts/bench_voice_similarity.py"

REF = Path(os.environ.get("REF_WAV", ROOT / "data/voices/loli/loli_15s.wav"))
GEN = Path(os.environ.get("GEN_DIR", LOLI / "eval/listen/v2_emotion"))
OUT = Path(os.environ.get("OUT_DIR", LOLI / "eval/bench/v2_emotion_ecapa"))


def main() -> int:
    if not MAJOR_BENCH.is_file():
        print(f"Missing: {MAJOR_BENCH}", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        str(MAJOR_BENCH),
        "--ref-wav", str(REF),
        "--gen-dir", str(GEN),
        "--out-dir", str(OUT),
        "--train-raw", str(LOLI / "train_raw.jsonl"),
        "--teacher-root", str(LOLI / "wavs/v15_pruned"),
        *sys.argv[1:],
    ]
    env = {**os.environ, "MOSS_RT_TRAIN_DIR": str(LOLI)}
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
