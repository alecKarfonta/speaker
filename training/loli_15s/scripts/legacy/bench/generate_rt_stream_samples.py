#!/usr/bin/env python3
"""Generate /tts/stream samples with TTFA + throughput metrics."""

from __future__ import annotations

import io
import json
import os
import struct
import time
import wave
from pathlib import Path

import requests

API = "http://127.0.0.1:8016"
OUT = Path(__file__).resolve().parents[1] / "training/loli_15s/eval/listen/rt_merged_demo"

SAMPLES = [
    ("stream_01_short.wav", "Hello! The streaming voice is working now."),
    (
        "stream_02_medium.wav",
        "I've been working on real-time text to speech for a while now, "
        "and it's finally starting to sound natural and consistent.",
    ),
    (
        "stream_03_long.wav",
        "This is a longer paragraph designed to test sustained streaming performance. "
        "It includes multiple sentences with various punctuation marks. "
        "The quality should remain consistent throughout the entire generation.",
    ),
    (
        "stream_04_story.wav",
        "Once upon a time, in a small village nestled between rolling hills, "
        "there lived a curious girl who loved to explore. Every morning she would "
        "wander through the meadows, listening to the birds and watching the clouds drift by.",
    ),
    (
        "stream_05_question.wav",
        "What do you think about the weather today? I heard it might rain later, "
        "but the sun feels nice right now. Should we go for a walk?",
    ),
]


def stream_to_wav(text: str) -> tuple[bytes, dict]:
    t0 = time.perf_counter()
    response = requests.post(
        f"{API}/tts/stream",
        json={"text": text, "language": "en"},
        stream=True,
        timeout=300,
    )
    response.raise_for_status()
    raw = response.content
    wall_s = time.perf_counter() - t0

    ttfa_s: float | None = None
    all_pcm = bytearray()
    sr = 24000
    n_chunks = 0
    total_audio_s = 0.0
    gen_time_s = 0.0

    offset = 0
    while offset + 8 <= len(raw):
        audio_len, meta_len = struct.unpack_from("<II", raw, offset)
        offset += 8
        if offset + audio_len + meta_len > len(raw):
            break
        if audio_len > 0:
            if ttfa_s is None:
                ttfa_s = wall_s  # conservative; use benchmark_rt_clean for precise TTFA
            wav_data = raw[offset : offset + audio_len]
            try:
                with wave.open(io.BytesIO(wav_data), "rb") as wf:
                    sr = wf.getframerate()
                    all_pcm.extend(wf.readframes(wf.getnframes()))
            except Exception:
                all_pcm.extend(wav_data)
            n_chunks += 1
        offset += audio_len
        if meta_len > 0:
            meta = json.loads(raw[offset : offset + meta_len])
            offset += meta_len
            total_audio_s = float(meta.get("total_audio_duration", total_audio_s))
            gen_time_s = float(meta.get("total_generation_time", gen_time_s))
    if total_audio_s <= 0 and all_pcm:
        total_audio_s = len(all_pcm) / (sr * 2)

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(all_pcm))

    ttfa_ms = (ttfa_s or wall_s) * 1000
    sustained_s = wall_s - (ttfa_s or 0)
    sustained_rtf = total_audio_s / sustained_s if sustained_s > 0 else 0

    return wav_buf.getvalue(), {
        "ttfa_ms": round(ttfa_ms, 0),
        "wall_s": round(wall_s, 2),
        "audio_s": round(total_audio_s, 2),
        "gen_s": round(gen_time_s or wall_s, 2),
        "chunks": n_chunks,
        "throughput": round(total_audio_s / (gen_time_s or wall_s), 2),
        "sustained_rtf": round(sustained_rtf, 2),
    }


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api-url", default=os.environ.get("MOSS_RT_API", "http://127.0.0.1:8016"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    api = args.api_url.rstrip("/")
    root = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[4]))
    train = Path(os.environ.get("MOSS_RT_TRAIN_DIR", root / "training" / "loli_15s"))
    out = args.out or (train / "eval/listen/rt_merged_demo")
    out.mkdir(parents=True, exist_ok=True)
    print(f"API: {api}\nOutput: {out}\n")
    print(f"{'file':<24} {'TTFA':>7} {'audio':>7} {'wall':>7} {'sustRTF':>8} {'chunks':>7}")
    print("-" * 70)
    global API
    API = api
    for name, text in SAMPLES:
        path = out / name
        wav, m = stream_to_wav(text)
        path.write_bytes(wav)
        print(
            f"{name:<24} {m['ttfa_ms']:>6.0f}ms {m['audio_s']:>6.1f}s "
            f"{m['wall_s']:>6.1f}s {m['sustained_rtf']:>7.2f}x {m['chunks']:>7}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
