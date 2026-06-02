#!/usr/bin/env python3
"""Build text corpus for major_2_03 Realtime SFT (~10–30s target clips)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[3]))
_LEGACY = ROOT / "training/moss-realtime/scripts/legacy"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))
from major_teacher_styles import assign_row_styles  # noqa: E402

OUT = ROOT / "training/major_03/corpus"
SEED = 42

# Target ~10–30s at ~12–13 chars/sec → ~120–390 characters.
SHORT_MIN, SHORT_MAX = 120, 175
MED_MIN, MED_MAX = 175, 280
LONG_MIN, LONG_MAX = 280, 390

OPENERS = [
    "Listen.", "Consider this.", "Here's the situation.", "To be clear,", "In practice,",
    "From where I stand,", "The short version is:", "Let me walk you through it.",
    "Before we proceed,", "The data suggests", "What matters is", "Keep in mind that",
]

MIDDLES = [
    "the network latency spikes whenever the backup node takes over",
    "our last simulation assumed perfect conditions that never appear in the field",
    "the interface between hardware and instinct is thinner than most people admit",
    "every decision leaves a trace, even when you think nobody is watching",
    "the city breathes differently after midnight, as if it is waiting for instructions",
    "trust is a resource you spend carefully, not a currency you print on demand",
    "the algorithm predicted failure, but the team chose hope anyway",
    "silence can be a weapon, or a shield, depending on who breaks it first",
]

CLOSERS = [
    "That is the part people forget.", "We proceed from there.", "Make of that what you will.",
    "I will leave the conclusion to you.", "The rest is execution.", "Nothing more complicated than that.",
    "And that changes the entire timeline.", "You already know what comes next.",
]

PARAGRAPH_A = [
    "The corridor lights flickered once, then steadied. She studied the readout without blinking, "
    "counting heartbeats the way others count coins. Somewhere above them, turbines sang a low hymn. "
    "When the alert finally arrived, it was almost polite — a soft chime, a line of text, a map "
    "highlighted in amber. She tapped the console twice and the world rearranged itself around a "
    "single decision. There was no applause, only the hum of machines agreeing to try again.",
    "Rain had scrubbed the streets clean, leaving reflections that looked like second skylines. "
    "He walked without hurry, hands in pockets, mind already three meetings ahead. A drone passed "
    "overhead, whispering regulations into the wind. At the corner cafe, steam rose from cups held "
    "by strangers who would never know how close the day had come to fracture. He ordered black coffee "
    "and wrote one sentence in a notebook: adapt, then commit.",
    "They met at the edge of the old district, where brick gave way to glass and memory gave way to "
    "policy. She brought questions; he brought diagrams. For an hour they traded hypotheses like "
    "cards, each reveal tightening the game. When the sun slipped behind the towers, neither had won, "
    "but both understood the board. She closed her tablet and said they would reconvene at dawn.",
]

PARAGRAPH_B = [
    "In the lab, numbers climbed the wall like ivy. Every chart told a story, and every story demanded "
    "a verdict. She circled the anomaly twice, then a third time for luck. Colleagues drifted past with "
    "coffee and theories, but she stayed anchored to the screen. The pattern was not chaos; it was a "
    "language she had almost learned as a child. When the translation clicked, she laughed once, "
    "quietly, and forwarded the result to people who would either celebrate or panic.",
    "The train moved through farmland that looked painted rather than grown. Fields stitched themselves "
    "to the horizon until the sky took over. He read the briefing again, not because he had forgotten, "
    "but because repetition made fear manageable. At the next stop, a child waved at the window as if "
    "trains were friendly animals. He waved back, then returned to the document that would define the week.",
]

LONG_FORM = [
    " ".join(
        [
            PARAGRAPH_A[0],
            "Hours later, the same corridor smelled of ozone and cooled metal.",
            PARAGRAPH_B[0][:200],
            "By morning, the team would call it progress; she would call it survival.",
        ]
    ),
    " ".join(
        [
            PARAGRAPH_B[1],
            "The briefing had warned about collateral doubt — the kind that spreads quietly.",
            PARAGRAPH_A[1][:220],
            "He signed the form anyway, because waiting had never improved the odds.",
        ]
    ),
]

SENTENCE_BANK = [
    "Systems fail in predictable ways when humans pretend they are unpredictable.",
    "Clarity is expensive, but confusion invoices you daily.",
    "You can optimize a machine; you can only negotiate with a person.",
    "The map is never the territory, especially when the territory is moving.",
    "Courage is not the absence of fear, but the decision to act while fear watches.",
    "Every protocol exists because someone learned the hard way.",
    "Speed without direction is just noise with ambition.",
    "The quietest room in the building held the loudest truth.",
]


def _clip_chars(text: str, lo: int, hi: int) -> str:
    text = " ".join(text.split())
    if len(text) > hi:
        text = text[: hi - 3].rsplit(" ", 1)[0] + "..."
    while len(text) < lo and len(SENTENCE_BANK) > 0:
        extra = random.choice(SENTENCE_BANK)
        if extra not in text:
            text = f"{text} {extra}"
    return text[:hi]


def gen_short(rng: random.Random) -> str:
    text = f"{rng.choice(OPENERS)} {rng.choice(MIDDLES)}, {rng.choice(CLOSERS)}"
    return _clip_chars(text, SHORT_MIN, SHORT_MAX)


def gen_medium(rng: random.Random) -> str:
    parts = [rng.choice(PARAGRAPH_A), rng.choice(SENTENCE_BANK)]
    return _clip_chars(" ".join(parts), MED_MIN, MED_MAX)


def gen_long(rng: random.Random) -> str:
    base = rng.choice(LONG_FORM)
    if rng.random() < 0.5:
        base = f"{base} {rng.choice(SENTENCE_BANK)}"
    return _clip_chars(base, LONG_MIN, LONG_MAX)


def gen_single_turn(rng: random.Random, idx: int) -> dict:
    bucket = rng.random()
    if bucket < 0.20:
        text, length = gen_short(rng), "short"
    elif bucket < 0.55:
        text, length = gen_medium(rng), "medium"
    elif bucket < 0.85:
        text, length = gen_long(rng), "long"
    else:
        text, length = gen_long(rng), "medium_long"
    return {
        "id": f"st_{idx:05d}",
        "type": "single",
        "length": length,
        "target_dur_s": "10-30",
        "text": text,
        "char_len": len(text),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT / "texts.jsonl")
    parser.add_argument("--single", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen: set[str] = set()
    i = 0
    attempts = 0
    max_attempts = args.single * 30
    while len(rows) < args.single and attempts < max_attempts:
        row = gen_single_turn(rng, i)
        i += 1
        attempts += 1
        key = row["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

    while len(rows) < args.single:
        n = len(rows)
        base = gen_long(rng)
        text = _clip_chars(f"{base} Variant {n}.", LONG_MIN, LONG_MAX)
        rows.append(
            {
                "id": f"st_{n:05d}",
                "type": "single",
                "length": "pad",
                "target_dur_s": "10-30",
                "text": text,
                "char_len": len(text),
            }
        )

    rng.shuffle(rows)
    for row in rows:
        assign_row_styles(row, rng)

    lengths = [r["char_len"] for r in rows]
    stats = {
        "total": len(rows),
        "single": len(rows),
        "out": str(args.out),
        "target_dur_s": "10-30",
        "char_len_min": min(lengths),
        "char_len_max": max(lengths),
        "char_len_avg": round(sum(lengths) / len(lengths), 1),
    }
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out.parent / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
