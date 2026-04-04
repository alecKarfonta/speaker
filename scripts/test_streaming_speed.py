#!/usr/bin/env python3
"""
Streaming TTS Speed Benchmark — Real-Time Factor (RTF) Evaluation

Measures whether the /tts/stream endpoint can sustain real-time playback.

Key metrics:
  - RTF (Real-Time Factor): audio_duration / generation_time. Must be ≥1.0 for real-time.
  - TTFB (Time to First Byte): Latency until the first audio chunk arrives.
  - Sustained RTF: RTF excluding the TTFB warm-up period.
  - Per-chunk timing: Whether individual chunks arrive fast enough to keep a buffer filled.

Usage:
    python3 scripts/test_streaming_speed.py
    python3 scripts/test_streaming_speed.py --voice biden --base-url http://localhost:8012
    python3 scripts/test_streaming_speed.py --target-rtf 1.2  # require 20% margin
"""

import argparse
import json
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_URL = "http://localhost:8012"
DEFAULT_VOICE = "biden"
TARGET_RTF = 1.0  # minimum RTF for real-time playback

TEST_SENTENCES = [
    # Short
    ("short", "The quick brown fox jumps over the lazy dog."),
    ("short", "Hello, this is a streaming text to speech test."),
    # Medium
    ("medium", "In a world without walls and fences, who needs windows and gates?"),
    ("medium", "The rain in Spain falls mainly on the plain, at least according to the musical."),
    # Long
    ("long", "This is a longer paragraph designed to test sustained streaming performance. "
             "It includes multiple sentences with various punctuation marks. "
             "The quality should remain consistent throughout the entire generation."),
    ("long", "In the heart of the bustling city, where skyscrapers reached toward the heavens "
             "and streets hummed with the rhythm of countless footsteps, there existed a small "
             "coffee shop that time seemed to have forgotten."),
]


@dataclass
class ChunkTiming:
    """Timing data for a single streamed chunk."""
    chunk_index: int
    arrival_time: float  # absolute perf_counter
    audio_bytes: int
    pcm_samples: int  # int16 samples
    sample_rate: int


@dataclass
class StreamResult:
    """Result of a single streaming test."""
    text: str
    category: str
    voice: str
    ok: bool = True
    error: Optional[str] = None
    # Timing
    ttfb_ms: float = 0.0
    total_time_s: float = 0.0
    # Audio
    total_audio_s: float = 0.0
    total_bytes: int = 0
    chunk_count: int = 0
    sample_rate: int = 24000
    # Computed
    overall_rtf: float = 0.0
    sustained_rtf: float = 0.0  # RTF excluding TTFB
    # Per-chunk
    chunk_timings: List[ChunkTiming] = field(default_factory=list)


def run_stream_test(text: str, category: str, voice: str, base_url: str) -> StreamResult:
    """Run one streaming speed test and return timing results."""
    payload = {
        "text": text,
        "voice_name": voice,
        "language": "en",
        "output_format": "wav",  # WAV for compatibility with existing frame parser
    }

    result = StreamResult(text=text, category=category, voice=voice)
    t_start = time.perf_counter()

    try:
        resp = requests.post(
            f"{base_url}/tts/stream",
            json=payload,
            stream=True,
            timeout=120,
        )
    except Exception as e:
        result.ok = False
        result.error = str(e)
        return result

    if resp.status_code != 200:
        result.ok = False
        result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        return result

    # Stream and collect per-chunk timing
    raw_chunks = []
    chunk_arrivals = []

    for chunk in resp.iter_content(chunk_size=16384):
        if chunk:
            now = time.perf_counter()
            raw_chunks.append(chunk)
            chunk_arrivals.append(now)

    t_done = time.perf_counter()
    result.total_time_s = t_done - t_start
    result.total_bytes = sum(len(c) for c in raw_chunks)

    if not raw_chunks:
        result.ok = False
        result.error = "No data received"
        return result

    result.ttfb_ms = (chunk_arrivals[0] - t_start) * 1000

    # Parse framed chunks to extract per-chunk audio sizes
    raw_stream = b"".join(raw_chunks)
    offset = 0
    chunk_idx = 0

    while offset < len(raw_stream):
        if offset + 8 > len(raw_stream):
            break
        audio_len, meta_len = struct.unpack_from("<II", raw_stream, offset)
        offset += 8

        if offset + audio_len + meta_len > len(raw_stream):
            break

        audio_bytes = raw_stream[offset:offset + audio_len]
        offset += audio_len
        meta_bytes = raw_stream[offset:offset + meta_len]
        offset += meta_len

        # Parse metadata for sample rate
        sr = result.sample_rate
        try:
            meta = json.loads(meta_bytes)
            sr = meta.get("sr", meta.get("sample_rate", sr))
            result.sample_rate = sr
        except Exception:
            pass

        if audio_len > 0:
            # Determine PCM sample count
            # WAV: strip 44-byte header, int16 = 2 bytes/sample
            if audio_bytes[:4] == b"RIFF":
                pcm_bytes = audio_len - 44
            else:
                pcm_bytes = audio_len
            pcm_samples = pcm_bytes // 2  # int16

            result.chunk_timings.append(ChunkTiming(
                chunk_index=chunk_idx,
                arrival_time=chunk_arrivals[min(chunk_idx, len(chunk_arrivals) - 1)],
                audio_bytes=audio_len,
                pcm_samples=pcm_samples,
                sample_rate=sr,
            ))
            result.total_audio_s += pcm_samples / sr

        chunk_idx += 1

    result.chunk_count = len(result.chunk_timings)

    # Compute RTF
    if result.total_time_s > 0:
        result.overall_rtf = result.total_audio_s / result.total_time_s

    # Sustained RTF: exclude TTFB
    sustained_time = result.total_time_s - (result.ttfb_ms / 1000)
    if sustained_time > 0 and result.total_audio_s > 0:
        result.sustained_rtf = result.total_audio_s / sustained_time

    return result


def print_result(r: StreamResult):
    """Print a single test result."""
    if not r.ok:
        print(f"  ❌ FAIL: {r.error}")
        return

    rtf_icon = "✅" if r.overall_rtf >= TARGET_RTF else "❌"
    sustained_icon = "✅" if r.sustained_rtf >= TARGET_RTF else "⚠️"

    text_preview = r.text[:50] + "..." if len(r.text) > 50 else r.text
    print(
        f"  {rtf_icon} RTF={r.overall_rtf:.2f}x  "
        f"sustained={r.sustained_rtf:.2f}x  "
        f"TTFB={r.ttfb_ms:.0f}ms  "
        f"gen={r.total_time_s:.2f}s  "
        f"audio={r.total_audio_s:.2f}s  "
        f"chunks={r.chunk_count}  "
        f"| {text_preview}"
    )


def main():
    parser = argparse.ArgumentParser(description="Streaming TTS Speed Benchmark")
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--target-rtf", type=float, default=TARGET_RTF,
                        help="Minimum RTF to pass (default: 1.0)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Skip server readiness poll")
    args = parser.parse_args()

    target_rtf = args.target_rtf
    base_url = args.base_url.rstrip("/")

    print("=" * 72)
    print("  Streaming TTS Speed Benchmark")
    print(f"  Target: RTF ≥ {target_rtf:.1f}x  |  Voice: {args.voice}  |  Server: {base_url}")
    print("=" * 72)

    # Wait for server
    if not args.no_wait:
        print("Waiting for server...", end="", flush=True)
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                r = requests.get(f"{base_url}/health", timeout=3)
                if r.status_code == 200:
                    print(" ready ✓")
                    break
            except Exception:
                pass
            print(".", end="", flush=True)
            time.sleep(2)
        else:
            print(" TIMEOUT ✗")
            sys.exit(1)

    # Run tests
    results: List[StreamResult] = []
    for category, text in TEST_SENTENCES:
        r = run_stream_test(text, category, args.voice, base_url)
        results.append(r)
        print_result(r)

    # ── Summary ──────────────────────────────────────────────────────────────
    ok_results = [r for r in results if r.ok]
    if not ok_results:
        print("\n❌ All tests failed!")
        return 1

    avg_rtf = sum(r.overall_rtf for r in ok_results) / len(ok_results)
    avg_sustained = sum(r.sustained_rtf for r in ok_results) / len(ok_results)
    avg_ttfb = sum(r.ttfb_ms for r in ok_results) / len(ok_results)
    total_audio = sum(r.total_audio_s for r in ok_results)
    total_compute = sum(r.total_time_s for r in ok_results)
    passed = sum(1 for r in ok_results if r.overall_rtf >= target_rtf)
    errors = sum(1 for r in results if not r.ok)

    # Per-category breakdown
    categories = {}
    for r in ok_results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    print(f"\n{'=' * 72}")
    print("  SPEED BENCHMARK RESULTS")
    print(f"{'=' * 72}")
    print(f"  Overall RTF:     {avg_rtf:.2f}x  {'✅ PASS' if avg_rtf >= target_rtf else '❌ FAIL'}")
    print(f"  Sustained RTF:   {avg_sustained:.2f}x  (excludes TTFB)")
    print(f"  Avg TTFB:        {avg_ttfb:.0f}ms")
    print(f"  Total audio:     {total_audio:.2f}s generated in {total_compute:.2f}s")
    print(f"  Tests passed:    {passed}/{len(ok_results)}  (errors: {errors})")

    print(f"\n  By category:")
    for cat, cat_results in sorted(categories.items()):
        cat_rtf = sum(r.overall_rtf for r in cat_results) / len(cat_results)
        cat_icon = "✅" if cat_rtf >= target_rtf else "❌"
        print(f"    {cat_icon} {cat:>8}: RTF={cat_rtf:.2f}x  (n={len(cat_results)})")

    # Playback buffer simulation
    print(f"\n  Playback buffer analysis:")
    for buffer_s in [1.0, 2.0, 3.0]:
        can_sustain = avg_sustained >= 1.0 and avg_ttfb / 1000 < buffer_s
        icon = "✅" if can_sustain else "❌"
        print(f"    {icon} {buffer_s:.0f}s buffer: {'sustainable' if can_sustain else 'will stall'}")

    print(f"\n  Verdict: {'✅ REAL-TIME CAPABLE' if avg_rtf >= target_rtf else '❌ TOO SLOW FOR REAL-TIME'}")
    print("=" * 72)

    return 0 if avg_rtf >= target_rtf else 1


if __name__ == "__main__":
    sys.exit(main())
