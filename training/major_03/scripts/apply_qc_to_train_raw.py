#!/usr/bin/env python3
"""Point train_raw at pruned WAVs and drop quarantined clips."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MAJOR = ROOT / "training" / "major_03"
TRAIN = MAJOR / "train_raw.jsonl"
QUAR = MAJOR / "qc" / "quarantine_ids.txt"
OUT = MAJOR / "train_raw.jsonl"


def main() -> int:
    if not QUAR.is_file():
        print(f"Missing {QUAR} — run QC prune first", file=sys.stderr)
        return 1
    bad = {line.strip() for line in QUAR.read_text().splitlines() if line.strip()}
    rows_in = [json.loads(l) for l in TRAIN.read_text().splitlines() if l.strip()]
    bak = TRAIN.with_suffix(".jsonl.pre_pruned")
    if not bak.is_file():
        shutil.copy2(TRAIN, bak)

    kept = []
    dropped = 0
    for row in rows_in:
        conv = row.get("conversations") or []
        if not conv:
            dropped += 1
            continue
        wav = conv[0].get("wav", "")
        name = Path(wav).name
        if name in bad:
            dropped += 1
            continue
        if "/wavs/v15/" in wav and "/wavs/v15_pruned/" not in wav:
            conv[0]["wav"] = wav.replace("/wavs/v15/", "/wavs/v15_pruned/")
        kept.append(row)

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n")
    print(
        json.dumps(
            {
                "quarantined": len(bad),
                "input_rows": len(rows_in),
                "kept": len(kept),
                "dropped": dropped,
                "out": str(OUT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
