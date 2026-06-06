#!/usr/bin/env python3
"""Generate varied native-voice samples from a running MOSS-RT finetune server."""

from __future__ import annotations

import io
import json
import os
import struct
import time
import wave
from pathlib import Path

import requests

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[3]))
TRAIN_DIR = Path(os.environ.get("MOSS_RT_TRAIN_DIR", ROOT / "training" / "loli_15s"))
DEFAULT_OUT = TRAIN_DIR / "eval" / "listen" / "epoch7_variety"
DEFAULT_API = os.environ.get("MOSS_RT_API", "http://127.0.0.1:8016")

# Varied prompts: length, tone, and structure (no ref wav — native-voice LoRA).
SAMPLES: list[tuple[str, str, str]] = [
    ("01_greeting_short.wav", "greeting", "Hi there! I'm so happy you're here today."),
    ("02_question.wav", "question", "Guess what? Do you think the stars look brighter after it rains?"),
    ("03_excited.wav", "excited", "Oh wow, you won't believe what I just found behind the garden shed!"),
    ("04_calm_medium.wav", "calm", "The meadow feels peaceful this morning, with birds singing and a soft breeze."),
    (
        "05_story_long.wav",
        "story",
        "Once upon a time, in a tiny village between rolling hills, a curious girl wandered past "
        "wildflowers every dawn. She collected shiny pebbles, waved at the baker, and dreamed of "
        "sailing across the lake before the sun turned the water to gold.",
    ),
    (
        "06_goodbye.wav",
        "sign-off",
        "Thank you so much for spending time with me today. Sweet dreams, and I'll see you again soon!",
    ),
    (
        "07_whisper_soft.wav",
        "soft",
        "Shh, listen carefully. I think I hear tiny footsteps in the attic, but maybe it's just the wind.",
    ),
    (
        "08_punctuation.wav",
        "punctuation",
        "Wait — really? Well... okay! First we eat, then we play; after that, who knows?",
    ),
    (
        "09_list.wav",
        "list",
        "We need three things: a flashlight, a warm scarf, and one very brave cookie.",
    ),
    (
        "10_surprise.wav",
        "surprise",
        "No way! A ladybug landed right on my nose, and it tickled so much I almost sneezed!",
    ),
    (
        "11_reflective.wav",
        "reflective",
        "Sometimes I wonder why clouds change shape so fast, like they're telling secrets only the wind understands.",
    ),
    (
        "12_invite.wav",
        "invite",
        "Want to come explore the creek with me? I heard the frogs are singing their loudest songs tonight.",
    ),
]

# Training-style emotion cues (text prefixes + prosody from gap_category=emotion corpus).
EMOTION_SAMPLES: list[tuple[str, str, str]] = [
    (
        "01_excited_wait.wav",
        "excited",
        "Wait wait wait! I keep replaying that moment in my head, and it still sparkles. I mean it.",
    ),
    (
        "02_curious_tell_me.wav",
        "curious",
        "Tell me honestly — the meadow feels like a hug today.",
    ),
    (
        "03_gentle_softly.wav",
        "gentle",
        "Softly now, everything is quiet except my heartbeat, and that is okay.",
    ),
    (
        "04_cheerful_wonder.wav",
        "cheerful",
        "I wonder, I want to laugh and cry at the same time, in the nicest way. Thank you for listening.",
    ),
    (
        "05_cozy_quietly.wav",
        "cozy",
        "Maybe quietly — The market was loud in the nicest way, with laughter and kettle whistles.",
    ),
    (
        "06_whisper_secret.wav",
        "whisper_soft",
        "Softly now, my hands were shaking, but I read the note anyway and smiled.",
    ),
    (
        "07_excited_huge.wav",
        "excited",
        "This is huge! I was nervous at first, but your message made everything lighter.",
    ),
    (
        "08_gentle_um.wav",
        "gentle",
        "Um, so, everything is quiet except my heartbeat, and that is okay.",
    ),
    (
        "09_curious_hmm.wav",
        "curious",
        "Hmm, interesting, we started at the little bridge where the stream sounds like bells.",
    ),
    (
        "10_proud_cozy.wav",
        "cozy",
        "I am so proud of us — I practiced what to say three times, then just waved instead.",
    ),
    (
        "11_oh_my_gosh.wav",
        "excited",
        "Oh my gosh! the world felt too big yesterday, but today it fits in my palms.",
    ),
    (
        "12_playful_look.wav",
        "playful",
        "Look at us! everything is quiet except my heartbeat, and that is okay.",
    ),
    (
        "13_story_market.wav",
        "storytelling",
        "Wait wait wait! The market was loud in the nicest way — laughter, kettle whistles, and boots on cobblestones. "
        "When the lanterns rose, the whole square looked like a bowl of warm light.",
    ),
    (
        "14_we_did_it.wav",
        "excited",
        "We did it! I keep replaying that moment in my head, and it still sparkles.",
    ),
]


def stream_to_wav(api: str, text: str) -> tuple[bytes, dict]:
    t0 = time.perf_counter()
    resp = requests.post(
        f"{api.rstrip('/')}/tts/stream",
        json={"text": text, "language": "en"},
        timeout=600,
    )
    resp.raise_for_status()
    raw = resp.content
    wall_s = time.perf_counter() - t0

    pcm = bytearray()
    sr = 24000
    offset = 0
    while offset + 8 <= len(raw):
        audio_len, meta_len = struct.unpack_from("<II", raw, offset)
        offset += 8
        if offset + audio_len + meta_len > len(raw):
            break
        if audio_len > 0:
            chunk = raw[offset : offset + audio_len]
            try:
                with wave.open(io.BytesIO(chunk), "rb") as wf:
                    sr = wf.getframerate()
                    pcm.extend(wf.readframes(wf.getnframes()))
            except Exception:
                pcm.extend(chunk)
        offset += audio_len + meta_len

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(pcm))

    audio_s = len(pcm) / (sr * 2) if pcm else 0.0
    return buf.getvalue(), {"wall_s": round(wall_s, 2), "audio_s": round(audio_s, 2), "sample_rate": sr}


def write_index(out: Path, rows: list[dict], *, html_title: str = "loli15s checkpoint-epoch-7 — variety eval") -> None:
    html = [
        "<!DOCTYPE html><html><head><meta charset=utf-8>",
        "<title>loli15s epoch-7 variety samples</title>",
        "<style>body{font-family:system-ui;max-width:52rem;margin:2rem auto;padding:0 1rem}",
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.5rem;vertical-align:top}",
        "audio{width:100%}</style></head><body>",
        f"<h1>{html_title}</h1>",
        "<table><tr><th>#</th><th>Tag</th><th>Text</th><th>Audio</th><th>dur</th></tr>",
    ]
    for i, row in enumerate(rows, 1):
        html.append(
            f"<tr><td>{i}</td><td>{row['tag']}</td><td>{row['text']}</td>"
            f"<td><audio controls src='{row['file']}'></audio></td>"
            f"<td>{row.get('audio_s', '?')}s</td></tr>"
        )
    html.append("</table></body></html>")
    (out / "index.html").write_text("\n".join(html), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-url", default=DEFAULT_API)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--preset",
        choices=("variety", "emotion"),
        default="variety",
        help="variety=length/tone mix; emotion=training-style cue prefixes",
    )
    p.add_argument("--wait-health", type=int, default=0, help="Seconds to poll /health before giving up")
    p.add_argument(
        "--quick-ab",
        action="store_true",
        help="Emotion preset only: 8 clips for fast v1/v2 comparison",
    )
    args = p.parse_args()
    samples = EMOTION_SAMPLES if args.preset == "emotion" else SAMPLES
    if args.quick_ab:
        if args.preset != "emotion":
            raise SystemExit("--quick-ab requires --preset emotion")
        keep = {
            "01_excited_wait.wav",
            "02_curious_tell_me.wav",
            "03_gentle_softly.wav",
            "05_cozy_quietly.wav",
            "07_excited_huge.wav",
            "12_playful_look.wav",
            "13_story_market.wav",
            "14_we_did_it.wav",
        }
        samples = [s for s in EMOTION_SAMPLES if s[0] in keep]
    if args.preset == "emotion" and args.out == DEFAULT_OUT:
        args.out = TRAIN_DIR / "eval" / "listen" / "epoch7_emotion_cues"
    api = args.api_url.rstrip("/")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.wait_health > 0:
        deadline = time.time() + args.wait_health
        while time.time() < deadline:
            try:
                r = requests.get(f"{api}/health", timeout=5)
                if r.ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(5)
        else:
            print(f"Server not healthy at {api}/health after {args.wait_health}s")
            return 1

    rows: list[dict] = []
    print(f"API: {api}\nOut: {out}\n")
    title = (
        "loli15s epoch-7 — emotion cue eval"
        if args.preset == "emotion"
        else "loli15s checkpoint-epoch-7 — variety eval"
    )
    for fname, tag, text in samples:
        wav, meta = stream_to_wav(api, text)
        path = out / fname
        path.write_bytes(wav)
        row = {"file": fname, "tag": tag, "text": text, **meta}
        rows.append(row)
        print(f"  {fname}  {meta['audio_s']}s  ({tag})")
    write_index(out, rows, html_title=title)
    print(f"\nWrote {len(rows)} WAVs + index.html + manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
