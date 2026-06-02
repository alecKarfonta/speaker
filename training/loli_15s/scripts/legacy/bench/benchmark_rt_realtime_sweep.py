#!/usr/bin/env python3
"""Find fastest real-time profile: TTFB + sustained RTF >= 1.0."""

from __future__ import annotations

import json
import struct
import sys
import time
from typing import Any

import requests

BASE = "http://127.0.0.1:8016"
LONG_TEXT = (
    "This is a longer paragraph designed to test sustained streaming performance. "
    "It includes multiple sentences with various punctuation marks. "
    "The quality should remain consistent throughout the entire generation."
)

# Keep text/dec at steady_large compile shape; sweep buffers first.
CONFIGS: list[tuple[str, dict[str, Any]]] = [
    ("steady_480", {"rt_min_samples_first_ms": 80, "rt_min_samples_steady_ms": 480}),
    ("steady_240", {"rt_min_samples_first_ms": 80, "rt_min_samples_steady_ms": 240}),
    ("steady_120", {"rt_min_samples_first_ms": 80, "rt_min_samples_steady_ms": 120}),
    ("fast_120", {"rt_min_samples_first_ms": 40, "rt_min_samples_steady_ms": 120}),
    ("fast_80", {"rt_min_samples_first_ms": 40, "rt_min_samples_steady_ms": 80}),
    ("text_12_buf240", {
        "rt_initial_text_chunk": 1,
        "rt_steady_text_chunk": 12,
        "rt_min_samples_first_ms": 80,
        "rt_min_samples_steady_ms": 240,
        "rt_decoder_chunk_frames": 12,
    }),
]


def measure(text: str, tune: dict[str, Any]) -> dict[str, Any]:
    payload = {"text": text, "language": "en", **tune}
    t0 = time.perf_counter()
    resp = requests.post(f"{BASE}/tts/stream", json=payload, stream=True, timeout=300)
    if resp.status_code != 200:
        return {"ok": False, "error": resp.status_code}
    parts: list[bytes] = []
    t_first = None
    for chunk in resp.iter_content(8192):
        if chunk:
            if t_first is None:
                t_first = time.perf_counter()
            parts.append(chunk)
    wall = time.perf_counter() - t0
    ttfb_ms = (t_first - t0) * 1000 if t_first else 0.0
    raw = b"".join(parts)
    offset, sr, audio_s, n = 0, 24000, 0.0, 0
    while offset + 8 <= len(raw):
        al, ml = struct.unpack_from("<II", raw, offset)
        offset += 8
        if offset + al + ml > len(raw):
            break
        ab = raw[offset : offset + al]
        offset += al + ml
        if al > 0:
            pcm = (al - 44) if ab[:4] == b"RIFF" else al
            audio_s += (pcm // 2) / sr
            n += 1
    sus = wall - ttfb_ms / 1000.0
    sustained = audio_s / sus if sus > 0 else 0.0
    return {
        "ok": True,
        "ttfb_ms": ttfb_ms,
        "wall_s": wall,
        "audio_s": audio_s,
        "chunks": n,
        "overall_rtf": audio_s / wall if wall else 0,
        "sustained_rtf": sustained,
        "realtime_ok": sustained >= 0.98 and ttfb_ms < 2000,
    }


def main() -> int:
    print("Warmup...", flush=True)
    measure("Hello.", {})
    time.sleep(1)

    rows = []
    print(f"\n{'name':<14} {'TTFB':>7} {'sust':>6} {'RTF':>6} {'wall':>6} {'rt?':>4}")
    print("-" * 50)
    for name, tune in CONFIGS:
        m = measure(LONG_TEXT, tune)
        if not m.get("ok"):
            print(f"{name}: FAIL")
            continue
        rows.append((name, tune, m))
        score = m["sustained_rtf"] - m["ttfb_ms"] / 5000.0
        m["score"] = score
        ok = "yes" if m["realtime_ok"] else "no"
        print(
            f"{name:<14} {m['ttfb_ms']:6.0f}ms {m['sustained_rtf']:5.2f}x "
            f"{m['overall_rtf']:5.2f}x {m['wall_s']:5.1f}s {ok:>4}"
        )

    good = [r for r in rows if r[2].get("realtime_ok")]
    if good:
        best = max(good, key=lambda r: r[2]["score"])
        print(f"\nBest real-time profile: {best[0]} -> {json.dumps(best[1])}")
    elif rows:
        best = max(rows, key=lambda r: r[2]["score"])
        print(f"\nBest compromise: {best[0]} -> {json.dumps(best[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
