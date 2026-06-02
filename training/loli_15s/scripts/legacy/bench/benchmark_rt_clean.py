#!/usr/bin/env python3
"""Clean benchmark: server defaults only (steady_large), proper TTFB timing."""

from __future__ import annotations

import json
import struct
import sys
import time

import requests

BASE = "http://127.0.0.1:8016"
LONG_TEXT = (
    "This is a longer paragraph designed to test sustained streaming performance. "
    "It includes multiple sentences with various punctuation marks. "
    "The quality should remain consistent throughout the entire generation."
)
SHORT_TEXT = "Hello, this is a quick streaming test."


def stream_metrics(text: str) -> dict:
    t0 = time.perf_counter()
    resp = requests.post(
        f"{BASE}/tts/stream",
        json={"text": text, "language": "en"},
        stream=True,
        timeout=300,
    )
    if resp.status_code != 200:
        return {"ok": False, "error": f"HTTP {resp.status_code}"}
    parts: list[bytes] = []
    first_byte_t: float | None = None
    for chunk in resp.iter_content(8192):
        if chunk:
            if first_byte_t is None:
                first_byte_t = time.perf_counter()
            parts.append(chunk)
    wall = time.perf_counter() - t0
    ttfb_ms = (first_byte_t - t0) * 1000 if first_byte_t else 0.0

    raw = b"".join(parts)
    offset = 0
    sr = 24000
    audio_s = 0.0
    chunks = 0
    while offset + 8 <= len(raw):
        audio_len, meta_len = struct.unpack_from("<II", raw, offset)
        offset += 8
        if offset + audio_len + meta_len > len(raw):
            break
        audio_bytes = raw[offset : offset + audio_len]
        offset += audio_len
        meta_bytes = raw[offset : offset + meta_len]
        offset += meta_len
        if audio_len > 0:
            try:
                meta = json.loads(meta_bytes)
                sr = int(meta.get("sr", meta.get("sample_rate", sr)))
            except Exception:
                pass
            pcm = (audio_len - 44) if audio_bytes[:4] == b"RIFF" else audio_len
            audio_s += (pcm // 2) / sr
            chunks += 1

    sustained_s = wall - (ttfb_ms / 1000.0)
    return {
        "ok": True,
        "ttfb_ms": ttfb_ms,
        "wall_s": wall,
        "audio_s": audio_s,
        "chunks": chunks,
        "overall_rtf": audio_s / wall if wall > 0 else 0,
        "sustained_rtf": audio_s / sustained_s if sustained_s > 0 else 0,
    }


def main() -> int:
    print("Waiting for health...", flush=True)
    for _ in range(90):
        try:
            h = requests.get(f"{BASE}/health", timeout=3).json()
            if h.get("status") == "ready":
                print(f"  ready: {json.dumps({k: h[k] for k in h if k.startswith('rt_') or k == 'realtime_enabled'}, default=str(h))}")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("Server not ready")
        return 1

    print("\n=== Clean benchmark (server defaults, no rt_* overrides) ===\n")
    for label, text in [("short", SHORT_TEXT), ("long-1", LONG_TEXT), ("long-2", LONG_TEXT)]:
        m = stream_metrics(text)
        if not m.get("ok"):
            print(f"{label}: FAIL {m.get('error')}")
            continue
        print(
            f"{label:8}  TTFB={m['ttfb_ms']:6.0f}ms  "
            f"RTF={m['overall_rtf']:.2f}x  sustained={m['sustained_rtf']:.2f}x  "
            f"audio={m['audio_s']:.1f}s  wall={m['wall_s']:.1f}s  chunks={m['chunks']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
