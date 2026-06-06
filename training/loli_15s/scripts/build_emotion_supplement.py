#!/usr/bin/env python3
"""Append pure emotion lines to batch3 (or write standalone supplement)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT / "training/loli_15s/scripts/legacy"
sys.path.insert(0, str(LEGACY))
from v15_teacher_styles import assign_row_styles  # noqa: E402

BATCH3 = ROOT / "training/loli_15s_batch3"
EMOTION_START = 63000

PREFIXES = [
    "Oh wow!", "Wait wait wait!", "I'm so excited!", "Guess what?", "No way!",
    "Listen carefully!", "Shh,", "Curious question:", "Softly,", "Surprised whisper:",
    "Bright and bubbly:", "Gentle sigh:", "Playful tease:", "I can't believe it!",
    "Okay okay okay!", "Hold on —", "This is huge:", "My heart is full:",
]
CORE = [
    "I keep replaying that moment and it still sparkles.",
    "We actually did it together and I'm still buzzing.",
    "Tell me everything about the lantern boat at dusk.",
    "That sound in the attic might be magic, or maybe just wind.",
    "The festival drums are starting and I can barely stand still.",
    "Every time I think about it I grin from ear to ear.",
    "I need to tell you before I burst with happiness.",
    "My voice won't stop shaking because I'm so happy.",
    "I still get goosebumps when I remember that hug.",
    "Can we do that again tomorrow? Please?",
]
TAILS = [
    "I mean it!", "Seriously!", "Right?", "Don't you think?", "Tell me you feel it too.",
]


def gen_line(rng: random.Random, idx: int) -> dict:
    text = f"{rng.choice(PREFIXES)} {rng.choice(CORE)} {rng.choice(TAILS)}"
    if rng.random() < 0.2:
        text = f"{text} (beat {idx})"
    return {
        "id": f"loli3_st_{idx:05d}",
        "type": "single",
        "length": "short",
        "gap_category": "emotion",
        "target_dur_s": "5-15",
        "text": text,
        "char_len": len(text),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--start-id", type=int, default=EMOTION_START)
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--append-to", type=Path, default=BATCH3 / "corpus/texts.jsonl")
    p.add_argument("--out-only", type=Path, default=None, help="Write supplement only, no append")
    args = p.parse_args()

    rng = random.Random(args.seed)
    seen: set[str] = set()
    if args.append_to.is_file() and not args.out_only:
        for line in args.append_to.read_text().splitlines():
            if line.strip():
                seen.add(json.loads(line)["text"].strip().lower())

    rows: list[dict] = []
    idx = args.start_id
    while len(rows) < args.count:
        row = gen_line(rng, idx)
        idx += 1
        key = row["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        assign_row_styles(row, rng)
        rows.append(row)

    if args.out_only:
        out = args.out_only
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        print(f"Wrote {len(rows)} → {out}")
        return 0

    existing: list[dict] = []
    if args.append_to.is_file():
        existing = [json.loads(l) for l in args.append_to.read_text().splitlines() if l.strip()]
        bak = args.append_to.with_suffix(".jsonl.pre_emotion_append")
        if not bak.is_file():
            bak.write_text(args.append_to.read_text())
    merged = existing + rows
    args.append_to.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n")
    stats = {
        "appended": len(rows),
        "total": len(merged),
        "emotion_new_ids": f"loli3_st_{args.start_id:05d}..",
    }
    (args.append_to.parent / "emotion_append_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
