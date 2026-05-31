#!/usr/bin/env python3
"""Evaluate baseline vs finetuned Realtime on held-out loli_15s prompts."""

from __future__ import annotations

import argparse
import io
import json
import struct
import time
import wave
from pathlib import Path

import requests
import soundfile as sf

import os

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[4]))

HELD_OUT = [
    "Once upon a time, there was a little bunny who lived in a cozy burrow. Every day, the bunny would hop through the meadow looking for yummy carrots and making new friends along the way.",
    "Hi everyone! Thanks for listening — hope you enjoy the story!",
    "Can you hear me clearly? The meadow is so peaceful this morning.",
    "Good morning, sunshine! The birds are singing and the dew still sparkles on the grass.",
]

BEST = {
    "audio_temperature": 0.75,
    "audio_top_p": 0.55,
    "audio_top_k": 30,
    "audio_repetition_penalty": 1.1,
}


def stream_to_wav(base_url: str, text: str, voice: str, cfg: dict) -> tuple[bytes, float]:
    url = base_url.rstrip("/") + "/tts/stream"
    payload = {"text": text, "language": "en", "voice_name": voice, **cfg}
    t0 = time.time()
    resp = requests.post(url, json=payload, stream=True, timeout=600)
    resp.raise_for_status()
    raw = resp.content
    ttfa = time.time() - t0
    offset, pcm, sr = 0, bytearray(), 24000
    while offset + 8 <= len(raw):
        al, ml = struct.unpack_from("<II", raw, offset)
        offset += 8
        audio = raw[offset : offset + al]
        offset += al + ml
        if al > 0:
            with wave.open(io.BytesIO(audio), "rb") as wf:
                sr = wf.getframerate()
                pcm.extend(wf.readframes(wf.getnframes()))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(bytes(pcm))
    return buf.getvalue(), ttfa


def transcribe(wav_path: Path, stt_url: str) -> str:
    with wav_path.open("rb") as f:
        r = requests.post(
            stt_url,
            files={"file": (wav_path.name, f, "audio/wav")},
            data={"model": "base", "language": "en", "response_format": "json"},
            timeout=120,
        )
    r.raise_for_status()
    return r.json().get("text", "").strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-url", default="http://127.0.0.1:8016")
    parser.add_argument("--finetuned-url", default=None)
    parser.add_argument("--voice", default="loli_15s")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training/loli_15s/eval")
    parser.add_argument("--stt", default="http://localhost:8603/v1/audio/transcriptions")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for label, base in [("baseline", args.baseline_url), ("finetuned", args.finetuned_url)]:
        if not base:
            continue
        sub = args.out_dir / label
        sub.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(HELD_OUT):
            wav_bytes, ttfa = stream_to_wav(base, text, args.voice, BEST)
            path = sub / f"{i:02d}.wav"
            path.write_bytes(wav_bytes)
            x, sr = sf.read(str(path))
            dur = len(x) / sr
            try:
                hyp = transcribe(path, args.stt)
            except requests.RequestException:
                hyp = ""
            results.append({
                "model": label,
                "idx": i,
                "text": text,
                "wav": str(path.relative_to(ROOT)),
                "duration_s": round(dur, 2),
                "ttfa_s": round(ttfa, 2),
                "transcript": hyp,
            })
            print(f"{label} [{i}] dur={dur:.1f}s ttfa={ttfa:.1f}s", flush=True)

    (args.out_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out_dir / 'eval_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
