#!/usr/bin/env python3
"""Compare OpenMOSS C++ vs Python MOSS STT round-trip on the same phrases."""
from __future__ import annotations

import hashlib
import os
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests

OPENMOSS = os.environ.get("OPENMOSS_TTS_URL", "http://localhost:8014")
PYTHON_MOSS = os.environ.get("PYTHON_MOSS_TTS_URL", "http://localhost:8013")
STT = os.environ.get("STT_API_URL", "http://192.168.1.196:8603/v1/audio/transcriptions")
STT_KEY = os.environ.get("STT_API_KEY", "your-api-key")

CASES = [
    ("short", "Hello world test one two three."),
    ("fox", "The quick brown fox jumps over the lazy dog."),
    ("clone", "Hi there! I'm so happy to see you today.", "loli"),
]


def transcribe(wav: bytes) -> str:
    r = requests.post(
        STT,
        headers={"Authorization": f"Bearer {STT_KEY}"},
        files={"file": ("audio.wav", wav, "audio/wav")},
        data={"model": "base", "language": "en"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("text", "").strip()


def tts(url: str, text: str, voice: str | None = None) -> tuple[bytes, float, dict]:
    payload = {"text": text, "language": "English", "max_new_tokens": 300}
    if voice:
        payload["voice_name"] = voice
    t0 = time.perf_counter()
    r = requests.post(f"{url.rstrip('/')}/tts", json=payload, timeout=300)
    wall = time.perf_counter() - t0
    r.raise_for_status()
    hdrs = {k.lower(): v for k, v in r.headers.items()}
    return r.content, wall, hdrs


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main() -> int:
    out = Path("tests/output/stt_compare")
    out.mkdir(parents=True, exist_ok=True)

    print(f"OpenMOSS: {OPENMOSS}")
    print(f"Python:   {PYTHON_MOSS}")
    print(f"STT:      {STT}\n")

    fails = 0
    for case in CASES:
        name, text = case[0], case[1]
        voice = case[2] if len(case) > 2 else None
        print(f"=== {name}: {text[:50]}... ===")

        for label, url in [("openmoss", OPENMOSS), ("python", PYTHON_MOSS)]:
            try:
                wav, wall, hdrs = tts(url, text, voice)
            except Exception as e:
                print(f"  [{label}] TTS ERROR: {e}")
                fails += 1
                continue
            md5 = hashlib.md5(wav).hexdigest()[:12]
            path = out / f"{name}_{label}.wav"
            path.write_bytes(wav)
            try:
                stt = transcribe(wav)
            except Exception as e:
                stt = f"STT_ERROR: {e}"
            score = sim(text, stt) if not stt.startswith("STT_ERROR") else 0.0
            rtf = hdrs.get("x-rtf", "?")
            print(f"  [{label}] sim={score:.0%} rtf={rtf} wall={wall:.1f}s md5={md5}")
            print(f"           STT: {stt!r}")
            if score < 0.55:
                fails += 1
        print()

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
