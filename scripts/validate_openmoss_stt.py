#!/usr/bin/env python3
"""Generate openmoss TTS samples and verify with STT round-trip."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests

DEFAULT_TTS = "http://localhost:8014"
DEFAULT_STT = os.environ.get("STT_API_URL", "http://192.168.1.196:8603/v1/audio/transcriptions")
DEFAULT_STT_KEY = os.environ.get("STT_API_KEY", "your-api-key")

TESTS = [
    ("plain_short", {"text": "Hello world test one two three.", "language": "English", "max_new_tokens": 300}),
    ("plain_fox", {"text": "The quick brown fox jumps over the lazy dog.", "language": "English", "max_new_tokens": 400}),
    ("clone_greeting", {"text": "Hi there! I'm so happy to see you today.", "language": "English", "max_new_tokens": 350, "voice_name": "loli"}),
    ("clone_welcome", {"text": "Good morning everyone, welcome to the show.", "language": "English", "max_new_tokens": 400, "voice_name": "loli"}),
]


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def transcribe(stt_url: str, stt_key: str, wav_bytes: bytes) -> str:
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {"model": "base", "language": "en"}
    headers = {"Authorization": f"Bearer {stt_key}"}
    r = requests.post(stt_url, files=files, data=data, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json().get("text", "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenMOSS TTS + STT validation")
    parser.add_argument("--tts-url", default=DEFAULT_TTS)
    parser.add_argument("--stt-url", default=DEFAULT_STT)
    parser.add_argument("--stt-key", default=DEFAULT_STT_KEY)
    parser.add_argument("--out-dir", default="tests/output/openmoss_stt_proof")
    parser.add_argument("--min-similarity", type=float, default=0.55)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"TTS: {args.tts_url}")
    print(f"STT: {args.stt_url}")
    print(f"Output: {out_dir}\n")

    results = []
    for name, payload in TESTS:
        text = payload["text"]
        t0 = time.perf_counter()
        r = requests.post(f"{args.tts_url.rstrip('/')}/tts", json=payload, timeout=300)
        wall = time.perf_counter() - t0
        r.raise_for_status()
        md5 = hashlib.md5(r.content).hexdigest()
        wav_path = out_dir / f"{name}.wav"
        wav_path.write_bytes(r.content)

        stt = transcribe(args.stt_url, args.stt_key, r.content)
        sim = similarity(text, stt)
        dur = r.headers.get("X-Audio-Duration", "?")
        rtf = r.headers.get("X-RTF", "?")
        ok = sim >= args.min_similarity

        results.append((name, md5, sim, ok, stt))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        print(f"  input:  {text}")
        print(f"  STT:    {stt!r}")
        print(f"  sim:    {sim:.0%}  md5={md5[:12]}  dur={dur}s rtf={rtf} wall={wall:.1f}s")
        print()

    unique_md5 = len({m for _, m, _, _, _ in results})
    passed = sum(1 for *_, ok, _ in results if ok)
    print(f"Unique MD5: {unique_md5}/{len(results)}")
    print(f"STT pass (>={args.min_similarity:.0%}): {passed}/{len(results)}")

    if unique_md5 < len(results):
        print("WARNING: some outputs share identical audio bytes")
    return 0 if passed == len(results) and unique_md5 == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
