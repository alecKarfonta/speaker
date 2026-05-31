#!/usr/bin/env python3
"""Sweep MOSS-Realtime streaming chunk settings via /tts/stream request fields."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

DEFAULT_URL = "http://127.0.0.1:8016"
LONG_TEXT = (
    "This is a longer paragraph designed to test sustained streaming performance. "
    "It includes multiple sentences with various punctuation marks. "
    "The quality should remain consistent throughout the entire generation."
)

# name, initial_text, steady_text, first_buf_ms, steady_buf_ms, decoder_frames
CONFIGS: list[tuple[str, dict[str, Any]]] = [
    (
        "default",
        {
            "rt_initial_text_chunk": 1,
            "rt_steady_text_chunk": 12,
            "rt_min_samples_first_ms": 80,
            "rt_min_samples_steady_ms": 240,
            "rt_decoder_chunk_frames": 12,
        },
    ),
    (
        "ttfa_fast",
        {
            "rt_initial_text_chunk": 1,
            "rt_steady_text_chunk": 4,
            "rt_min_samples_first_ms": 40,
            "rt_min_samples_steady_ms": 120,
            "rt_decoder_chunk_frames": 6,
        },
    ),
    (
        "steady_large",
        {
            "rt_initial_text_chunk": 1,
            "rt_steady_text_chunk": 24,
            "rt_min_samples_first_ms": 80,
            "rt_min_samples_steady_ms": 480,
            "rt_decoder_chunk_frames": 24,
        },
    ),
    (
        "uniform_4",
        {
            "rt_initial_text_chunk": 4,
            "rt_steady_text_chunk": 4,
            "rt_min_samples_first_ms": 80,
            "rt_min_samples_steady_ms": 80,
            "rt_decoder_chunk_frames": 12,
        },
    ),
    (
        "uniform_12",
        {
            "rt_initial_text_chunk": 12,
            "rt_steady_text_chunk": 12,
            "rt_min_samples_first_ms": 240,
            "rt_min_samples_steady_ms": 240,
            "rt_decoder_chunk_frames": 12,
        },
    ),
    (
        "throughput",
        {
            "rt_initial_text_chunk": 1,
            "rt_steady_text_chunk": 12,
            "rt_min_samples_first_ms": 160,
            "rt_min_samples_steady_ms": 480,
            "rt_decoder_chunk_frames": 24,
        },
    ),
]


@dataclass
class Result:
    name: str
    ok: bool = True
    error: str = ""
    ttfb_ms: float = 0.0
    wall_s: float = 0.0
    audio_s: float = 0.0
    overall_rtf: float = 0.0
    sustained_rtf: float = 0.0
    stalls: int = 0
    chunks: int = 0
    tune: dict[str, Any] = field(default_factory=dict)


def _parse_audio_duration(raw: bytes) -> tuple[float, int, float]:
    offset = 0
    sr = 24000
    audio_s = 0.0
    chunks = 0
    ttfb_s = 0.0
    first = True
    t0 = time.perf_counter()
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
            if first:
                ttfb_s = time.perf_counter() - t0
                first = False
            try:
                meta = json.loads(meta_bytes)
                sr = int(meta.get("sr", meta.get("sample_rate", sr)))
            except Exception:
                pass
            pcm = (audio_len - 44) if audio_bytes[:4] == b"RIFF" else audio_len
            audio_s += (pcm // 2) / sr
            chunks += 1
    return audio_s, chunks, ttfb_s


def run_config(base_url: str, name: str, tune: dict[str, Any], text: str) -> Result:
    payload = {"text": text, "language": "en", **tune}
    r = Result(name=name, tune=tune)
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{base_url}/tts/stream", json=payload, stream=True, timeout=300)
    except Exception as e:
        r.ok = False
        r.error = str(e)
        return r
    if resp.status_code != 200:
        r.ok = False
        r.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        return r
    parts = []
    for chunk in resp.iter_content(16384):
        if chunk:
            parts.append(chunk)
    r.wall_s = time.perf_counter() - t0
    audio_s, chunks, ttfb_s = _parse_audio_duration(b"".join(parts))
    r.audio_s = audio_s
    r.chunks = chunks
    r.ttfb_ms = ttfb_s * 1000
    if r.wall_s > 0:
        r.overall_rtf = audio_s / r.wall_s
    sustained = r.wall_s - ttfb_s
    if sustained > 0:
        r.sustained_rtf = audio_s / sustained
    return r


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    print("=" * 88)
    print("MOSS-Realtime chunk size sweep")
    print(f"Server: {base}")
    print("=" * 88)

    if not args.no_warmup:
        print("\nWarmup (default tuning)...", flush=True)
        w = run_config(base, "warmup", CONFIGS[0][1], "Hello warmup.")
        if not w.ok:
            print(f"Warmup failed: {w.error}")
            return 1
        print(f"  {w.audio_s:.1f}s audio, TTFB={w.ttfb_ms:.0f}ms\n")

    rows: list[Result] = []
    for name, tune in CONFIGS:
        print(f"Config: {name} ...", flush=True)
        res = run_config(base, name, tune, LONG_TEXT)
        rows.append(res)
        if not res.ok:
            print(f"  FAIL: {res.error}")
            continue
        print(
            f"  TTFB={res.ttfb_ms:.0f}ms  RTF={res.overall_rtf:.2f}x  "
            f"sustained={res.sustained_rtf:.2f}x  audio={res.audio_s:.1f}s  "
            f"wall={res.wall_s:.1f}s  chunks={res.chunks}"
        )

    ok = [r for r in rows if r.ok]
    if not ok:
        return 1

    print("\n" + "=" * 88)
    print(f"{'config':<14} {'TTFB':>7} {'RTF':>6} {'sust':>6} {'audio':>6} {'wall':>6} {'#chk':>5}  tuning")
    print("-" * 88)
    for r in sorted(ok, key=lambda x: (-x.sustained_rtf, x.ttfb_ms)):
        t = r.tune
        print(
            f"{r.name:<14} {r.ttfb_ms:>6.0f}ms {r.overall_rtf:>5.2f}x {r.sustained_rtf:>5.2f}x "
            f"{r.audio_s:>5.1f}s {r.wall_s:>5.1f}s {r.chunks:>5d}  "
            f"txt {t['rt_initial_text_chunk']}/{t['rt_steady_text_chunk']} "
            f"buf {t['rt_min_samples_first_ms']:.0f}/{t['rt_min_samples_steady_ms']:.0f}ms "
            f"dec {t['rt_decoder_chunk_frames']}"
        )
    best = max(ok, key=lambda x: x.sustained_rtf)
    lowest_ttfb = min(ok, key=lambda x: x.ttfb_ms)
    print("=" * 88)
    print(f"Best sustained RTF: {best.name} ({best.sustained_rtf:.2f}x)")
    print(f"Lowest TTFB:         {lowest_ttfb.name} ({lowest_ttfb.ttfb_ms:.0f}ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
