#!/usr/bin/env python3
"""Build large text corpus for loli_15s Realtime SFT (single + multi-turn)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import os
import sys

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[4]))
_LEGACY = Path(__file__).resolve().parent
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))
from v15_teacher_styles import assign_row_styles  # noqa: E402

OUT = ROOT / "training/loli_15s/corpus"

SEED = 42

# --- Single-turn seeds ---
BASE_TEXTS = [
    "Hello! This is the first clone sample. MOSS-TTS is copying this reference voice.",
    "The rain fell softly on the old city streets. Nobody spoke for a long moment, and then the voice returned, calm and clear as before.",
    "Once upon a time, there was a little bunny who lived in a cozy burrow. Every day, the bunny would hop through the meadow looking for yummy carrots and making new friends along the way.",
    "Hi there! Welcome to the stream, everyone!",
    "Once upon a time, in a village tucked between rolling hills, a small girl discovered a hidden garden where fireflies danced every evening.",
    "The morning sun painted the sky in shades of gold and rose. She walked along the cobblestone path, humming a tune her grandmother had taught her.",
    "Did you see that? I think something magical just happened behind the cherry trees!",
    "Oh wow, oh wow! We did it! I knew we could make it if we just kept trying, even when things got really hard!",
    "Hi everyone! Thanks for listening — hope you enjoy the story!",
    "Can you hear me clearly? The meadow is so peaceful this morning.",
    "Good morning, sunshine! The birds are singing and the dew still sparkles on the grass.",
    "Thank you so much for spending time with me today. Sweet dreams, and I'll see you again soon!",
]

SHORT_OPENERS = [
    "Hey!", "Hi there!", "Oh!", "Wow!", "Guess what?", "Listen!", "Okay so,", "You know what?",
    "Good news!", "Quick question:", "Hmm,", "Well,", "Actually,", "So today,", "Right now,",
]
SHORT_MIDDLES = [
    "the meadow looks absolutely beautiful", "I found a tiny ladybug on my sleeve",
    "the clouds are shaped like little bunnies", "my tea is finally the perfect temperature",
    "the wind chimes sound like a lullaby", "someone left a note on the garden gate",
    "the stars are starting to peek out", "there is a rainbow after the rain",
    "the library cat waved at me", "the cookies smell amazing from here",
]
SHORT_CLOSERS = [
    "isn't that sweet?", "what do you think?", "I had to tell you!",
    "it made my whole day.", "can you believe it?", "so magical!",
    "let's go see!", "I'm so happy!", "tell me everything!", "okay bye for now!",
]

MEDIUM_TEMPLATES = [
    "I was walking through the {place} when I noticed {detail}. It reminded me of {memory}, and I couldn't help but smile.",
    "Yesterday at {time}, something funny happened. {event} Everyone laughed, and we stayed until the sky turned pink.",
    "If you ever visit the {place}, make sure to {action}. The air smells like {scent} and the world feels wonderfully calm.",
    "My favorite part of {season} is when {detail}. I like to sit by the window and imagine all the stories waiting outside.",
    "Let me tell you about {topic}. It started quietly, but soon {event} and we all felt like we were part of something special.",
]

MEDIUM_SLOTS = {
    "place": ["meadow", "little bridge", "flower market", "cozy burrow", "willow tree", "village square", "stream bank"],
    "detail": ["fireflies dancing", "a hidden path", "soft moss underfoot", "a friendly frog", "golden light on the grass"],
    "memory": ["childhood summers", "grandma's stories", "our first picnic", "a rainy afternoon", "a birthday wish"],
    "time": ["sunrise", "late afternoon", "just after lunch", "early evening", "the quiet hour before bed"],
    "event": ["a kitten joined our circle.", "someone started humming.", "we shared the last cookie.", "a breeze carried cherry blossoms."],
    "action": ["listen to the birds", "feed the ducks", "pick wildflowers", "watch the clouds drift", "write in your journal"],
    "scent": ["fresh rain", "warm bread", "pine and honey", "lavender and grass", "apple cinnamon"],
    "season": ["spring", "summer", "autumn", "winter", "the rainy season"],
    "topic": ["the little bunny", "our garden", "the old clock tower", "the star festival", "the secret map"],
}

LONG_PARAGRAPHS = [
    "Once upon a time, there was a little bunny who lived in a cozy burrow beneath an ancient oak. Every morning she would poke her nose outside, sniff the dew on the clover, and hop through the meadow looking for the crunchiest carrots. Along the way she met a shy hedgehog, a chatty bluebird, and a wise old tortoise who taught her that kindness travels faster than fear. By sunset the burrow glowed with candlelight, and she whispered thank you to the stars.",
    "The village festival was finally here, and ribbons fluttered from every window. Children ran with paper lanterns while musicians played on the green. Our narrator stood on tiptoe to see the puppet show, laughing when the dragon sneezed glitter. Later, when fireworks painted the sky, everyone held hands and promised to meet again next year. It was the sort of night you tuck into memory like a warm blanket.",
    "Rain tapped gently on the roof while tea steamed in a chipped blue cup. Outside, puddles mirrored the streetlamps, and earth smelled rich and alive. She opened her favorite book and read aloud to an audience of stuffed animals, giving each character a different voice. When the last page turned, thunder rolled far away and she felt safe, curious, and ready for tomorrow's adventure.",
]

# --- Multi-turn templates ---
USER_LINES = [
    "Hey, can you tell me a short story?",
    "What's the weather like in the meadow today?",
    "I'm feeling a little nervous about tomorrow.",
    "Do you remember the bunny from yesterday?",
    "Can we talk about something happy?",
    "I found a strange map in the attic.",
    "What should I pack for a picnic?",
    "Tell me something cozy before bed.",
    "Did you hear that sound outside?",
    "I need help planning a surprise party.",
]

ASSISTANT_REPLIES = [
    "Of course! Let me tell you about a brave little bunny who loved exploring.",
    "The meadow is sunny and warm, with a soft breeze and butterflies everywhere.",
    "That's okay. Take a deep breath — you're stronger than you think.",
    "Yes! The bunny with the pink ribbon who shares carrots with everyone.",
    "Happy things are everywhere if you look closely. Want me to list three?",
    "Ooh, a map! Maybe it leads to the hidden garden behind the hill.",
    "Bring a blanket, lemonade, and your favorite book. Don't forget sunscreen!",
    "Close your eyes and imagine a lantern boat drifting on a quiet lake.",
    "Probably just the wind in the willows, but let's peek together carefully.",
    "Surprises are fun! We could bake cookies and hide them in flower boxes.",
]

FOLLOWUP_USER = [
    "That sounds lovely. What happened next?",
    "Can you describe it more?",
    "I'm smiling already. Go on!",
    "Wait, really? Tell me more!",
    "How did it end?",
]

FOLLOWUP_ASSISTANT = [
    "Well, the bunny hopped over a tiny bridge and found a field of golden flowers.",
    "Picture soft grass, blue sky, and the smell of honey on the breeze.",
    "Then everyone joined in, and the whole meadow felt like one big hug.",
    "It ended with laughter, shared snacks, and a promise to meet again soon.",
    "And they all slept peacefully, dreaming of the next adventure.",
]


def fill_template(template: str, rng: random.Random) -> str:
    text = template
    for key, options in MEDIUM_SLOTS.items():
        text = text.replace("{" + key + "}", rng.choice(options))
    return text


def gen_short(rng: random.Random) -> str:
    return f"{rng.choice(SHORT_OPENERS)} {rng.choice(SHORT_MIDDLES)}, {rng.choice(SHORT_CLOSERS)}"


def gen_medium(rng: random.Random) -> str:
    return fill_template(rng.choice(MEDIUM_TEMPLATES), rng)


def gen_single_turn(rng: random.Random, idx: int) -> dict:
    bucket = rng.random()
    if bucket < 0.15:
        text = rng.choice(BASE_TEXTS)
        length = "seed"
    elif bucket < 0.40:
        text = gen_short(rng)
        length = "short"
    elif bucket < 0.75:
        text = gen_medium(rng)
        length = "medium"
    else:
        text = rng.choice(LONG_PARAGRAPHS)
        length = "long"
    return {
        "id": f"st_{idx:05d}",
        "type": "single",
        "length": length,
        "text": text,
    }


def gen_multi_turn(rng: random.Random, idx: int) -> dict:
    n_assistant = rng.randint(2, 3)
    turns = []
    turns.append({"role": "user", "text": rng.choice(USER_LINES)})
    turns.append({"role": "assistant", "text": rng.choice(ASSISTANT_REPLIES)})
    for _ in range(n_assistant - 1):
        turns.append({"role": "user", "text": rng.choice(FOLLOWUP_USER)})
        turns.append({"role": "assistant", "text": rng.choice(FOLLOWUP_ASSISTANT)})
    return {
        "id": f"mt_{idx:05d}",
        "type": "multi",
        "turns": turns,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT / "texts.jsonl")
    parser.add_argument("--single", type=int, default=2500)
    parser.add_argument("--multi", type=int, default=400)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen_text: set[str] = set()

    i = 0
    attempts = 0
    max_attempts = args.single * 20
    while len([r for r in rows if r["type"] == "single"]) < args.single and attempts < max_attempts:
        row = gen_single_turn(rng, i)
        i += 1
        attempts += 1
        key = row["text"].strip().lower()
        if key in seen_text:
            continue
        seen_text.add(key)
        rows.append(row)

    if len([r for r in rows if r["type"] == "single"]) < args.single:
        # Pad with indexed variants if template space exhausted
        n = len([r for r in rows if r["type"] == "single"])
        while n < args.single:
            base = rng.choice(BASE_TEXTS + LONG_PARAGRAPHS)
            text = f"{base} ({n})"
            rows.append({"id": f"st_{n:05d}", "type": "single", "length": "pad", "text": text})
            n += 1

    j = 0
    while len([r for r in rows if r["type"] == "multi"]) < args.multi:
        rows.append(gen_multi_turn(rng, j))
        j += 1

    rng.shuffle(rows)
    for row in rows:
        assign_row_styles(row, rng)
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "total": len(rows),
        "single": sum(1 for r in rows if r["type"] == "single"),
        "multi": sum(1 for r in rows if r["type"] == "multi"),
        "out": str(args.out),
    }
    (args.out.parent / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
