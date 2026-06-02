#!/usr/bin/env python3
"""Train/val split for loli_15s finetune corpus."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import os
import sys

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[4]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / "training/loli_15s/corpus/texts.jsonl")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.corpus.read_text().splitlines() if line.strip()]
    rng = random.Random(args.seed)

    by_type: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_type[row.get("type", "single")].append(row["id"])

    train_ids: list[str] = []
    val_ids: list[str] = []
    for typ, ids in by_type.items():
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * args.val_ratio))
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    out_dir = args.corpus.parent
    (out_dir / "train_ids.txt").write_text("\n".join(sorted(train_ids)) + "\n")
    (out_dir / "val_ids.txt").write_text("\n".join(sorted(val_ids)) + "\n")
    stats = {
        "train": len(train_ids),
        "val": len(val_ids),
        "by_type_train": {t: sum(1 for i in train_ids if i.startswith("st_" if t == "single" else "mt_")) for t in by_type},
    }
    (out_dir / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
