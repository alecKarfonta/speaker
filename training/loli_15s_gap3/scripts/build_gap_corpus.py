#!/usr/bin/env python3
"""Build gap-targeted loli_15s corpus (batch 3): long, numbers, names, questions, emotion."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEGACY = ROOT / "training/moss-realtime/scripts/legacy"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

from v15_teacher_styles import assign_row_styles  # noqa: E402

DEFAULT_OUT = ROOT / "training/loli_15s_gap3/corpus/texts.jsonl"

# Existing corpora to dedupe against (normalized lower-case text).
DEDUPE_SOURCES = [
    ROOT / "training/loli_15s/corpus/texts.jsonl",
    ROOT / "training/loli_15s/train_raw.noref.jsonl",
    ROOT / "training/loli_15s_batch2/corpus/texts.jsonl",
    ROOT / "training/loli_15s_batch2/corpus/texts_supplement.jsonl",
]

LONG_INTROS = [
    "Let me tell you a longer story, okay?",
    "So this is what happened yesterday, and it still makes me smile.",
    "I want to walk you through everything, step by step, because it matters.",
    "Picture this with me for a moment — close your eyes if you want.",
]

LONG_MIDDLES = [
    "We started at the little bridge where the stream sounds like bells. The air smelled like rain and honey, and someone had tied ribbons on the willow branches. I kept thinking how lucky we are to notice small things before they slip away.",
    "First we met the hedgehog who always wears a leaf like a hat. Then the bluebird taught us a song with only three notes, but somehow it felt complete. By afternoon our pockets were full of pebbles that glittered, and nobody wanted to go home yet.",
    "The market was loud in the nicest way — laughter, kettle whistles, and boots on cobblestones. I bought a paper star for two coins and a promise to help clean up after the puppet show. When the lanterns rose, the whole square looked like a bowl of warm light.",
    "She opened the journal and read aloud about a map that only appears when you are kind to strangers. We followed chalk arrows past bakeries and bike bells until we found a gate covered in ivy. Behind it, fireflies spelled hello in the dark.",
]

LONG_OUTROS = [
    "That is why I still believe gentle days can change everything, if we pay attention.",
    "So yeah — that is the whole adventure. I hope it felt cozy in your imagination too.",
    "And when the last firefly blinked goodbye, we promised to meet there again next week.",
    "Thanks for listening all the way through. I am really glad you stayed with me.",
]

NAMES = [
    "Mira", "Lumi", "Hana", "Noa", "Sora", "Kai", "Ren", "Yuki", "Aya", "Niko",
    "Elio", "Piper", "June", "Cleo", "Rin", "Tess", "Arlo", "Fern", "Ivy", "Leo",
]
PLACES = [
    "Willowfen", "Starfall Lane", "Mossgleam", "Cherryhollow", "Briarwick",
    "Moonpond", "Driftmill", "Hazelcross", "Cloudmere", "Pebbleford",
]

NUMBER_TEMPLATES = [
    "It is exactly {t} right now, and I have {n} tiny tasks left before tea.",
    "We counted {n} fireflies, then {m} more appeared — can you believe that?",
    "The recipe needs {n} grams of sugar and {m} milliliters of milk, please.",
    "On {month} {day}, at {hour}:{minute}, the clock tower chimed {n} times.",
    "If we walk {n} steps north and {m} steps east, we reach the garden gate.",
    "I saved {n} coins, spent {m}, and still have enough for two strawberry tarts.",
    "The temperature is {n} degrees, but with sunshine it feels like {m}.",
    "Chapter {n}, page {m} — that is where the bunny finds the hidden key.",
]

QUESTION_TEMPLATES = [
    "Do you think {thing} will still be there tomorrow morning?",
    "Why does the {thing} glow only after the rain stops?",
    "How many {thing} can we fit in this little basket, honestly?",
    "What would you do if a {thing} knocked on your door at midnight?",
    "Could we maybe visit the {thing} before the festival starts?",
    "Have you ever seen a {thing} that winks back at you?",
    "Where did the {thing} go — did it hop away or fly?",
    "Would it be okay if we named the {thing} something silly?",
    "Is it just me, or does the {thing} sound happier when it rains?",
    "Can you help me decide which {thing} to bring to the picnic?",
    "Should we tell {name} about the {thing} we found by the gate?",
    "What if the {thing} is actually a secret map — would you follow it?",
    "Do you remember when the {thing} hid under the porch in {place}?",
    "Are you free to look at the {thing} with me after sunset?",
    "Will the {thing} still work if we whisper instead of shout?",
]

EMOTION_OPENERS = {
    "excited": ["Oh my gosh!", "Wait wait wait!", "This is huge!"],
    "gentle": ["Softly now,", "Hey, it is okay,", "Breathe with me —"],
    "curious": ["Hmm, interesting,", "I wonder,", "Tell me honestly —"],
    "proud": ["We did it!", "Look at us!", "I am so proud of us —"],
    "shy": ["Um, so,", "I hope this is not silly, but", "Maybe quietly —"],
}

THING_WORDS = [
    "lantern", "kitten", "music box", "rainbow", "storybook", "cookie jar",
    "wind chime", "secret path", "paper boat", "star map",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def load_existing_texts() -> set[str]:
    seen: set[str] = set()
    for path in DEDUPE_SOURCES:
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not text and row.get("conversations"):
                text = row["conversations"][0].get("text")
            if text:
                seen.add(text.strip().lower())
    return seen


def gen_long(rng: random.Random) -> str:
    n_mids = rng.randint(2, 3)
    parts = [rng.choice(LONG_INTROS)]
    parts.extend(rng.choice(LONG_MIDDLES) for _ in range(n_mids))
    parts.append(rng.choice(LONG_OUTROS))
    return " ".join(parts)


def gen_numbers(rng: random.Random) -> str:
    tpl = rng.choice(NUMBER_TEMPLATES)
    return tpl.format(
        t=f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d}",
        n=rng.randint(2, 99),
        m=rng.randint(1, 50),
        month=rng.choice(MONTHS),
        day=rng.randint(1, 28),
        hour=rng.randint(1, 12),
        minute=rng.randint(0, 59),
    )


def gen_names(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    place = rng.choice(PLACES)
    templates = [
        f"{name} from {place} left me a note on the garden gate — should we follow the map?",
        f"Everyone in {place} knows {name} and their little red scarf. Today they brought hot cocoa for the whole square.",
        f"I met {name} near {place} Station, and we talked about constellations until the streetlights hummed on.",
        f"If you visit {place}, ask for {name} at the bookshop. They always know which story fits your mood.",
    ]
    return rng.choice(templates)


def gen_question(rng: random.Random) -> str:
    return rng.choice(QUESTION_TEMPLATES).format(
        thing=rng.choice(THING_WORDS),
        name=rng.choice(NAMES),
        place=rng.choice(PLACES),
    )


EMOTION_BODIES = [
    "the meadow feels like a hug today.",
    "I keep replaying that moment in my head, and it still sparkles.",
    "I was nervous at first, but your message made everything lighter.",
    "my hands were shaking, but I read the note anyway and smiled.",
    "I want to laugh and cry at the same time, in the nicest way.",
    "everything is quiet except my heartbeat, and that is okay.",
    "I practiced what to say three times, then just waved instead.",
    "the world felt too big yesterday, but today it fits in my palms.",
]


def gen_emotion(rng: random.Random) -> str:
    mood = rng.choice(list(EMOTION_OPENERS.keys()))
    opener = rng.choice(EMOTION_OPENERS[mood])
    body = rng.choice(LONG_MIDDLES + EMOTION_BODIES)
    tail = rng.choice(["", " Does that make sense?", " I mean it.", " Thank you for listening."])
    return f"{opener} {body}{tail}"


GENERATORS = {
    "long": gen_long,
    "numbers": gen_numbers,
    "names": gen_names,
    "questions": gen_question,
    "emotion": gen_emotion,
}


def category_quotas(total: int, mix: dict[str, float]) -> dict[str, int]:
    """Fixed per-category counts (largest remainder) so dedupe cannot skew the mix."""
    names = list(mix.keys())
    raw = {c: total * mix[c] for c in names}
    quotas = {c: int(raw[c]) for c in names}
    remainder = total - sum(quotas.values())
    for c in sorted(names, key=lambda k: raw[k] - quotas[k], reverse=True)[:remainder]:
        quotas[c] += 1
    return quotas


def fill_category(
    rng: random.Random,
    cat: str,
    need: int,
    seen: set[str],
    start_idx: int,
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    idx = start_idx
    attempts = 0
    max_attempts = max(need * 200, 2000)
    while len(rows) < need and attempts < max_attempts:
        attempts += 1
        text = GENERATORS[cat](rng).strip()
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        row = {
            "id": f"g3_st_{idx:05d}",
            "type": "single",
            "gap_category": cat,
            "length": "long" if cat == "long" else cat,
            "text": text,
        }
        assign_row_styles(row, rng)
        rows.append(row)
        idx += 1
    return rows, idx


def build_rows(
    rng: random.Random,
    total: int,
    mix: dict[str, float],
    seen: set[str],
    start_id: int,
) -> list[dict]:
    quotas = category_quotas(total, mix)
    rows: list[dict] = []
    idx = start_id
    for cat in mix:
        need = quotas[cat]
        batch, idx = fill_category(rng, cat, need, seen, idx)
        rows.extend(batch)
        if len(batch) < need:
            print(f"WARN: {cat} only {len(batch)}/{need} unique lines", file=sys.stderr)
    rng.shuffle(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--total", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument(
        "--mix",
        default="long:0.40,numbers:0.15,names:0.15,questions:0.15,emotion:0.15",
        help="category:weight,...",
    )
    args = parser.parse_args()

    mix: dict[str, float] = {}
    for part in args.mix.split(","):
        name, w = part.split(":")
        mix[name.strip()] = float(w)
    if abs(sum(mix.values()) - 1.0) > 0.01:
        raise SystemExit(f"mix weights must sum to 1.0, got {sum(mix.values())}")

    rng = random.Random(args.seed)
    seen = load_existing_texts()
    print(f"Deduping against {len(seen)} existing lines", file=sys.stderr)

    rows = build_rows(rng, args.total, mix, seen, args.start_id)
    if len(rows) < args.total:
        print(f"WARN: only generated {len(rows)}/{args.total} unique lines", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_cat: dict[str, int] = {}
    lens: list[int] = []
    for row in rows:
        by_cat[row["gap_category"]] = by_cat.get(row["gap_category"], 0) + 1
        lens.append(len(row["text"]))

    stats = {
        "total": len(rows),
        "by_category": by_cat,
        "text_chars": {
            "min": min(lens) if lens else 0,
            "median": sorted(lens)[len(lens) // 2] if lens else 0,
            "max": max(lens) if lens else 0,
            "mean": round(sum(lens) / len(lens), 1) if lens else 0,
        },
        "id_prefix": "g3_st_",
        "out": str(args.out),
        "seed": args.seed,
        "mix": mix,
    }
    stats_path = args.out.parent / "corpus_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
