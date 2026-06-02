#!/usr/bin/env python3
"""Summarize current loli corpus vs gap targets (for planning batch 3)."""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOLI = ROOT / "training/loli_15s"
OUT = ROOT / "training/loli_15s_gap3/corpus/gap_report.json"


def load_texts() -> list[str]:
    texts: list[str] = []
    for path in (
        LOLI / "train_raw.noref.jsonl",
        ROOT / "training/loli_15s_batch2/corpus/texts.jsonl",
    ):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            t = row.get("text")
            if not t and row.get("conversations"):
                t = row["conversations"][0].get("text")
            if t:
                texts.append(t)
    return texts


def sample_durations(n: int = 300) -> dict:
    wav_dir = LOLI / "wavs/v15_pruned"
    if not wav_dir.is_dir():
        return {}
    import random

    wavs = list(wav_dir.glob("*.wav"))
    if not wavs:
        return {}
    sample = random.sample(wavs, min(n, len(wavs)))
    durs = []
    for p in sample:
        try:
            d = float(
                subprocess.check_output(
                    [
                        "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(p),
                    ],
                    text=True,
                ).strip()
            )
            durs.append(d)
        except (subprocess.CalledProcessError, ValueError):
            pass
    if not durs:
        return {}
    durs.sort()
    return {
        "sample_n": len(durs),
        "mean_s": round(sum(durs) / len(durs), 2),
        "p50_s": round(durs[len(durs) // 2], 2),
        "p90_s": round(durs[int(len(durs) * 0.9)], 2),
        "pct_ge_10s": round(100 * sum(1 for d in durs if d >= 10) / len(durs), 1),
        "pct_ge_15s": round(100 * sum(1 for d in durs if d >= 15) / len(durs), 1),
    }


def main() -> None:
    texts = load_texts()
    lens = [len(t) for t in texts]
    lens.sort()
    has_digit = sum(1 for t in texts if re.search(r"\d", t))
    has_q = sum(1 for t in texts if "?" in t)
    long_chars = sum(1 for l in lens if l >= 200)

    report = {
        "rows_analyzed": len(texts),
        "unique_texts": len(set(t.strip().lower() for t in texts)),
        "text_chars": {
            "p50": lens[len(lens) // 2] if lens else 0,
            "p90": lens[int(len(lens) * 0.9)] if lens else 0,
            "max": max(lens) if lens else 0,
        },
        "coverage_gaps": {
            "long_text_ge_200_chars_pct": round(100 * long_chars / max(len(texts), 1), 1),
            "contains_digit_pct": round(100 * has_digit / max(len(texts), 1), 1),
            "contains_question_pct": round(100 * has_q / max(len(texts), 1), 1),
        },
        "audio_sample": sample_durations(),
        "gap3_targets": {
            "planned_rows": 2500,
            "mix": "40% long, 15% numbers, 15% names, 15% questions, 15% emotion",
            "est_added_hours_at_8s_mean": round(2500 * 8 / 3600, 1),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
