#!/usr/bin/env python3
"""Build ~2k emotion-heavy loli_15s_batch3 corpus (gap_category=emotion supplement)."""

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
OUT = BATCH3 / "corpus/texts.jsonl"

EMOTION_PREFIXES = [
    "Oh wow!", "Wait wait wait!", "I'm so excited!", "Guess what?", "No way!",
    "Listen carefully!", "Shh,", "Curious question:", "Softly,", "Surprised whisper:",
    "Bright and bubbly:", "Gentle sigh:", "Playful tease:", "Story time!",
]
EMOTION_MIDDLES = [
    "I keep replaying that moment in my head, and it still sparkles.",
    "My heart is racing — we actually did it, together!",
    "Tell me everything about the lantern boat at dusk.",
    "That sound in the attic might be magic, or maybe just wind.",
    "The festival drums are starting, and I can barely stand still!",
    "Every time I think about it, I grin from ear to ear.",
    "I need to tell you before I burst with happiness!",
]
STORY_PLACES = ["meadow", "village square", "hidden garden", "creek bank", "oak burrow", "festival green"]
STORY_DETAILS = ["fireflies danced", "a shy hedgehog waved", "golden light on the grass", "cherry blossoms fell"]
QUESTION_SLOTS = {
    "topic": ["stars", "clouds", "frogs", "cookies", "the map", "rainbows", "lanterns"],
    "place": ["meadow", "attic", "creek", "library", "garden gate"],
}
CALM_DETAILS = [
    "birds singing and a soft breeze", "a lantern boat on a quiet lake",
    "dew on the clover and distant wind chimes", "warm tea and a favorite book",
]

START_ID = 60000


def _unique_text(rng: random.Random, idx: int, base: str) -> str:
    if rng.random() < 0.25:
        return f"{base} (moment {idx})"
    return base


def gen_emotion(rng: random.Random, idx: int) -> dict:
    text = _unique_text(
        rng,
        idx,
        f"{rng.choice(EMOTION_PREFIXES)} {rng.choice(EMOTION_MIDDLES)} I mean it!",
    )
    return {
        "id": f"loli3_st_{idx:05d}",
        "type": "single",
        "length": "short",
        "gap_category": "emotion",
        "target_dur_s": "5-15",
        "text": text,
        "char_len": len(text),
    }


def gen_story(rng: random.Random, idx: int) -> dict:
    place = rng.choice(STORY_PLACES)
    detail = rng.choice(STORY_DETAILS)
    text = _unique_text(
        rng,
        idx,
        f"Once upon a time, near the {place}, a curious girl watched while {detail}. "
        f"She waved at friends and dreamed of sailing before the sun turned the water to gold.",
    )
    return {
        "id": f"loli3_st_{idx:05d}",
        "type": "single",
        "length": "long",
        "gap_category": "story",
        "target_dur_s": "15-30",
        "text": text,
        "char_len": len(text),
    }


def gen_question(rng: random.Random, idx: int) -> dict:
    topic = rng.choice(QUESTION_SLOTS["topic"])
    place = rng.choice(QUESTION_SLOTS["place"])
    templates = [
        f"Do you think the {topic} look brighter after it rains?",
        f"Can you hear me clearly from the {place}?",
        f"What should we pack for the surprise — cookies, lanterns, or both?",
        f"Why do the {topic} change so fast, like secrets only the wind understands?",
    ]
    text = _unique_text(rng, idx, rng.choice(["Hey!", "Oh!", "Wow!", ""]) + rng.choice(templates))
    return {
        "id": f"loli3_st_{idx:05d}",
        "type": "single",
        "length": "medium",
        "gap_category": "question",
        "target_dur_s": "8-20",
        "text": text.strip(),
        "char_len": len(text),
    }


def gen_calm(rng: random.Random, idx: int) -> dict:
    detail = rng.choice(CALM_DETAILS)
    text = _unique_text(
        rng,
        idx,
        f"The meadow feels peaceful this morning, with {detail}. "
        "Thank you for spending time with me today.",
    )
    return {
        "id": f"loli3_st_{idx:05d}",
        "type": "single",
        "length": "medium",
        "gap_category": "calm",
        "target_dur_s": "8-18",
        "text": text,
        "char_len": len(text),
    }


def fill_shortfall(rows: list[dict], seen: set[str], rng: random.Random, idx: int, gen_fn, n: int) -> int:
    added = 0
    attempts = 0
    while added < n and attempts < n * 50:
        row = gen_fn(rng, idx)
        idx += 1
        attempts += 1
        key = row["text"].strip().lower()
        if key in seen:
            row["text"] = f"{row['text']} #{idx}"
            key = row["text"].strip().lower()
        seen.add(key)
        assign_row_styles(row, rng)
        rows.append(row)
        added += 1
    return idx


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=3000)
    p.add_argument("--seed", type=int, default=20260529)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    rng = random.Random(args.seed)
    n_em = int(args.count * 0.40)
    n_st = int(args.count * 0.30)
    n_q = int(args.count * 0.20)
    n_c = args.count - n_em - n_st - n_q

    rows: list[dict] = []
    seen: set[str] = set()
    idx = START_ID
    for gen_fn, n in [(gen_emotion, n_em), (gen_story, n_st), (gen_question, n_q), (gen_calm, n_c)]:
        idx = fill_shortfall(rows, seen, rng, idx, gen_fn, n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    stats = {
        "total": len(rows),
        "emotion": sum(1 for r in rows if r.get("gap_category") == "emotion"),
        "story": sum(1 for r in rows if r.get("gap_category") == "story"),
        "question": sum(1 for r in rows if r.get("gap_category") == "question"),
        "calm": sum(1 for r in rows if r.get("gap_category") == "calm"),
        "out": str(args.out),
    }
    (args.out.parent / "corpus_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
