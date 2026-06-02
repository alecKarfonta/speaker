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


def write_index(out: Path, rows: list[dict]) -> None:
    html = [
        "<!DOCTYPE html><html><head><meta charset=utf-8>",
        "<title>loli15s epoch-7 variety samples</title>",
        "<style>body{font-family:system-ui;max-width:52rem;margin:2rem auto;padding:0 1rem}",
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.5rem;vertical-align:top}",
        "audio{width:100%}</style></head><body>",
        "<h1>loli15s checkpoint-epoch-7 — variety eval</h1>",
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
    p.add_argument("--wait-health", type=int, default=0, help="Seconds to poll /health before giving up")
    args = p.parse_args()
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
    for fname, tag, text in SAMPLES:
        wav, meta = stream_to_wav(api, text)
        path = out / fname
        path.write_bytes(wav)
        row = {"file": fname, "tag": tag, "text": text, **meta}
        rows.append(row)
        print(f"  {fname}  {meta['audio_s']}s  ({tag})")
    write_index(out, rows)
    print(f"\nWrote {len(rows)} WAVs + index.html + manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
