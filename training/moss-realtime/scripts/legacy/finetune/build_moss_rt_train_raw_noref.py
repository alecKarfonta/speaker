#!/usr/bin/env python3
"""Build train_raw.noref.jsonl — strip ref_wav for native-voice SFT."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import os

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[5]))
DEFAULT_TRAIN_DIR = Path(
    os.environ.get("MOSS_RT_TRAIN_DIR", ROOT / "training" / "moss-realtime")
)
DEFAULT_INPUT = DEFAULT_TRAIN_DIR / "train_raw.jsonl"
DEFAULT_OUTPUT = DEFAULT_TRAIN_DIR / "train_raw.noref.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Missing {args.input}", file=sys.stderr)
        return 1

    n = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.pop("ref_wav", None)
            row.pop("ref_audio_codes", None)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"rows={n} (ref_wav stripped) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
