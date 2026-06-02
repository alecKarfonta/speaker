#!/usr/bin/env python3
"""Add MOSS v1.5 style/instruction tags to loli_15s corpus for varied teacher synthesis."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import os
import sys

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[4]))
_LEGACY = Path(__file__).resolve().parent
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))
from v15_teacher_styles import assign_row_styles  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "training/loli_15s/corpus/texts.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backup", action="store_true", default=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    lines = [ln for ln in args.corpus.read_text().splitlines() if ln.strip()]
    rows = [json.loads(ln) for ln in lines]

    if args.backup:
        bak = args.corpus.with_suffix(".jsonl.bak")
        shutil.copy2(args.corpus, bak)
        print(f"Backup -> {bak}")

    styles: Counter[str] = Counter()
    for row in rows:
        assign_row_styles(row, rng)
        if row["type"] == "single":
            styles[row["style"]] += 1
        else:
            for turn in row["turns"]:
                if turn["role"] == "assistant":
                    styles[turn["style"]] += 1

    with args.corpus.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {"rows": len(rows), "style_counts": dict(styles)}
    stats_path = args.corpus.parent / "corpus_style_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
