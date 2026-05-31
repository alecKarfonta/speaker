#!/usr/bin/env python3
"""MOSS-TTS-Realtime /tts/stream → STT round-trip (quality + latency)."""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import time
import wave
from pathlib import Path

import requests

MOSS_URL = os.environ.get("MOSS_RT_URL", "http://127.0.0.1:8016")
STT_URL = os.environ.get(
    "STT_API",
    os.environ.get("STT_API_URL", "http://localhost:8603/v1/audio/transcriptions"),
)
STT_KEY = os.environ.get("STT_API_KEY", os.environ.get("STT_KEY", "stt-api-key"))
STT_MODEL = os.environ.get("STT_MODEL", "medium")

TEST_SENTENCES = [
    "Hello, this is a streaming test of the loli voice.",
    "The quick brown fox jumps over the lazy dog.",
    "Once upon a time there was a little bunny in a cozy burrow.",
]


def assemble_chunks(raw_stream: bytes) -> tuple[bytes, int, list]:
    offset = 0
    pcm_chunks: list[bytes] = []
    meta_list: list = []
    sample_rate = 24000
    while offset + 8 <= len(raw_stream):
        audio_len, meta_len = struct.unpack_from("<II", raw_stream, offset)
        offset += 8
        if offset + audio_len + meta_len > len(raw_stream):
            break
        audio_bytes = raw_stream[offset : offset + audio_len]
        offset += audio_len
        meta_bytes = raw_stream[offset : offset + meta_len]
        offset += meta_len
        try:
            meta = json.loads(meta_bytes)
            meta_list.append(meta)
            sample_rate = int(meta.get("sr", meta.get("sample_rate", sample_rate)))
        except Exception:
            meta = {}
            meta_list.append(meta)
        if audio_len > 0:
            if audio_bytes[:4] == b"RIFF":
                pcm_chunks.append(audio_bytes[44:])
            else:
                pcm_chunks.append(audio_bytes)
    return b"".join(pcm_chunks), sample_rate, meta_list


def pcm_to_wav(raw_pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_pcm)
    return buf.getvalue()


def transcribe_wav(wav_bytes: bytes, stt_url: str = STT_URL) -> str:
    for url, key in (
        (stt_url, STT_KEY),
        ("http://192.168.1.196:8603/v1/audio/transcriptions", "your-api-key"),
    ):
        try:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"model": STT_MODEL, "language": "en"},
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json().get("text", "").strip()
            print(f"  STT {url} -> {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        except Exception as e:
            print(f"  STT {url} unreachable: {e}", file=sys.stderr)
    return ""


def word_similarity(original: str, transcribed: str) -> float:
    a = set(original.lower().split())
    b = set(transcribed.lower().split())
    return len(a & b) / len(a) if a else 0.0


def run_one(
    text: str,
    base_url: str,
    voice: str | None,
    save_dir: Path | None,
    stt_url: str,
) -> dict:
    payload: dict = {"text": text, "language": "en"}
    if voice:
        payload["voice_name"] = voice

    t0 = time.perf_counter()
    resp = requests.post(
        f"{base_url.rstrip('/')}/tts/stream",
        json=payload,
        stream=True,
        timeout=300,
    )
    if resp.status_code != 200:
        return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}

    parts: list[bytes] = []
    ttfb_ms = None
    for chunk in resp.iter_content(16384):
        if chunk:
            if ttfb_ms is None:
                ttfb_ms = (time.perf_counter() - t0) * 1000
            parts.append(chunk)
    wall_ms = (time.perf_counter() - t0) * 1000

    raw_pcm, sr, meta = assemble_chunks(b"".join(parts))
    if not raw_pcm:
        return {"ok": False, "error": "empty audio"}

    dur_s = len(raw_pcm) / 2 / sr
    wav = pcm_to_wav(raw_pcm, sr)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        slug = "".join(c if c.isalnum() else "_" for c in text[:24]).strip("_")
        out = save_dir / f"moss_stt_{slug}.wav"
        out.write_bytes(wav)
        print(f"  saved {out}")

    print("  transcribing…")
    hyp = transcribe_wav(wav, stt_url)
    sim = word_similarity(text, hyp)

    return {
        "ok": True,
        "text": text,
        "ttfb_ms": round(ttfb_ms or 0, 1),
        "wall_ms": round(wall_ms, 1),
        "audio_s": round(dur_s, 2),
        "chunks": len(meta),
        "transcribed": hyp,
        "similarity": round(sim, 3),
        "pass": sim >= 0.75,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="MOSS realtime stream → STT round-trip")
    p.add_argument("--base-url", default=MOSS_URL)
    p.add_argument("--voice", default=None, help="Omit for native LoRA voice")
    p.add_argument("--text", default=None)
    p.add_argument("--save-audio", type=Path, default=None)
    p.add_argument("--stt-url", default=STT_URL)
    args = p.parse_args()
    stt_url = args.stt_url

    base = args.base_url.rstrip("/")
    h = requests.get(f"{base}/health", timeout=10).json()
    print(f"MOSS: {h.get('status')} model={h.get('model_id')} rt={h.get('realtime_enabled')}")
  # rt_codec_backend may be in extended health dict
    if isinstance(h, dict):
        extra = requests.get(f"{base}/health").json()
        if "rt_codec_backend" in extra:
            print(f"codec_backend={extra['rt_codec_backend']}")
    print(f"STT: {stt_url}")

    texts = [args.text] if args.text else TEST_SENTENCES
    results = []
    for text in texts:
        print(f"\n--- {text[:60]}…" if len(text) > 60 else f"\n--- {text}")
        r = run_one(text, base, args.voice, args.save_audio, stt_url)
        results.append(r)
        if not r.get("ok"):
            print(f"  ERROR: {r.get('error')}")
            continue
        print(f"  TTFB {r['ttfb_ms']:.0f}ms  wall {r['wall_ms']:.0f}ms  audio {r['audio_s']:.1f}s")
        print(f"  STT: {r['transcribed']!r}")
        print(f"  similarity {r['similarity']*100:.0f}%  {'PASS' if r['pass'] else 'FAIL'}")

    ok = [r for r in results if r.get("ok")]
    passed = sum(1 for r in ok if r.get("pass"))
    print(f"\n=== {passed}/{len(texts)} passed  avg TTFB {sum(r['ttfb_ms'] for r in ok)/len(ok):.0f}ms ===" if ok else "\n=== all failed ===")
    return 0 if passed == len(texts) else 1


if __name__ == "__main__":
    sys.exit(main())
