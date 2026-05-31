#!/usr/bin/env python3
"""Benchmark POST /tts (non-streaming) RTF via response headers."""

from __future__ import annotations

import argparse
import time

import requests

DEFAULT_URL = "http://127.0.0.1:8016"
LONG_TEXT = (
    "This is a longer paragraph designed to test sustained streaming performance. "
    "It includes multiple sentences with various punctuation marks. "
    "The quality should remain consistent throughout the entire generation."
)
SHORT_TEXT = "Hello, this is a quick non-streaming synthesis test."


def bench_tts(base: str, text: str) -> dict:
    t0 = time.perf_counter()
    resp = requests.post(
        f"{base.rstrip('/')}/tts",
        json={"text": text, "language": "en"},
        timeout=300,
    )
    wall = time.perf_counter() - t0
    if resp.status_code != 200:
        return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    audio_s = float(resp.headers.get("X-Audio-Duration", 0))
    gen_s = float(resp.headers.get("X-Generation-Time", 0))
    # MOSS convention: gen/audio < 1.0 = faster than realtime
    moss_rtf = gen_s / audio_s if audio_s > 0 else 0.0
    throughput = audio_s / gen_s if gen_s > 0 else 0.0
    return {
        "ok": True,
        "wall_s": wall,
        "gen_s": gen_s,
        "audio_s": audio_s,
        "moss_rtf": moss_rtf,
        "throughput": throughput,
        "kb": len(resp.content) / 1024,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark MOSS POST /tts RTF")
    p.add_argument("--url", default=DEFAULT_URL)
    args = p.parse_args()
    base = args.url.rstrip("/")

    print("Waiting for health...", flush=True)
    for _ in range(90):
        try:
            h = requests.get(f"{base}/health", timeout=3).json()
            if h.get("realtime_enabled"):
                print(f"  ready: {h.get('model_id')}", flush=True)
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("Server not ready")
        return 1

    print("\n=== POST /tts benchmark (warm repeat) ===\n")
    for label, text in [("short", SHORT_TEXT), ("long-1", LONG_TEXT), ("long-2", LONG_TEXT)]:
        bench_tts(base, text)  # warm
        r = bench_tts(base, text)
        if not r["ok"]:
            print(f"{label:8} FAIL: {r['error']}")
            continue
        print(
            f"{label:8} gen={r['gen_s']:.2f}s  audio={r['audio_s']:.2f}s  "
            f"moss_rtf={r['moss_rtf']:.2f}x  throughput={r['throughput']:.2f}x  "
            f"wall={r['wall_s']:.2f}s  {r['kb']:.0f}KB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
