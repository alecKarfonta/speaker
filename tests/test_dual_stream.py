#!/usr/bin/env python3
"""Test dual concurrent TTS streaming.

Sends two requests: stream A starts first, stream B starts 2s later.
Reports timing, GPU assignment, and overlap for each stream.

Usage:
    python tests/test_dual_stream.py [--host HOST] [--port PORT]
"""
import argparse
import struct
import json
import time
import threading
import requests


def stream_request(label: str, text: str, voice: str, host: str, results: dict):
    """Send a streaming TTS request and collect timing/audio metrics."""
    url = f"{host}/tts/stream"
    t0 = time.perf_counter()
    results[label] = {
        "start": t0,
        "ttfa": None,
        "gpu": None,
        "audio_dur": 0.0,
        "gen_time": 0.0,
        "chunks": 0,
        "error": None,
    }

    try:
        resp = requests.post(
            url,
            json={"text": text, "voice_name": voice},
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()

        buf = b""
        for chunk in resp.iter_content(chunk_size=4096):
            buf += chunk
            # Parse framed binary: 4-byte audio_len + 4-byte meta_len + audio + meta
            while len(buf) >= 8:
                audio_len, meta_len = struct.unpack_from("<II", buf, 0)
                frame_size = 8 + audio_len + meta_len
                if len(buf) < frame_size:
                    break  # incomplete frame
                meta_bytes = buf[8 + audio_len : 8 + audio_len + meta_len]
                buf = buf[frame_size:]

                if meta_len > 0:
                    meta = json.loads(meta_bytes)
                    if meta.get("error"):
                        results[label]["error"] = meta["error"]
                        return

                    ad = meta.get("audio_duration", 0)
                    if ad > 0:
                        results[label]["chunks"] += 1
                        results[label]["audio_dur"] += ad
                        results[label]["gen_time"] = meta.get("generation_time", 0)
                        if results[label]["ttfa"] is None:
                            results[label]["ttfa"] = time.perf_counter() - t0
                        if results[label]["gpu"] is None:
                            results[label]["gpu"] = meta.get("gpu", "?")

    except Exception as e:
        results[label]["error"] = str(e)
    finally:
        results[label]["end"] = time.perf_counter()
        results[label]["wall_time"] = results[label]["end"] - t0


def main():
    parser = argparse.ArgumentParser(description="Test dual concurrent TTS streaming")
    parser.add_argument("--host", default="http://localhost:8013")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--voice", default="loli")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to wait before starting stream B")
    args = parser.parse_args()

    host = args.host
    if args.port:
        host = f"http://localhost:{args.port}"

    print(f"Target: {host}")
    print(f"Voice: {args.voice}")
    print(f"Delay between streams: {args.delay}s")
    print()

    # Check health first
    try:
        health = requests.get(f"{host}/health", timeout=5).json()
        print(f"Status: {health.get('status')}")
        workers = health.get("rt_workers", [])
        if workers:
            print(f"Workers: {len(workers)} — {workers}")
        print()
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    results = {}

    text_a = "Stream Alpha is the first request. It should start generating audio immediately after voice encoding and session setup."
    text_b = "Stream Bravo is the second request. It starts while Alpha is still running. On a single GPU it queues behind Alpha."

    t_global = time.perf_counter()

    # Start stream A
    print(f"[{0:.1f}s] Starting Stream A...")
    thread_a = threading.Thread(
        target=stream_request,
        args=("A", text_a, args.voice, host, results),
    )
    thread_a.start()

    # Wait, then start stream B
    time.sleep(args.delay)
    elapsed = time.perf_counter() - t_global
    print(f"[{elapsed:.1f}s] Starting Stream B (A has been running for {args.delay}s)...")
    thread_b = threading.Thread(
        target=stream_request,
        args=("B", text_b, args.voice, host, results),
    )
    thread_b.start()

    # Wait for both
    thread_a.join()
    elapsed = time.perf_counter() - t_global
    print(f"[{elapsed:.1f}s] Stream A finished")
    thread_b.join()
    elapsed = time.perf_counter() - t_global
    print(f"[{elapsed:.1f}s] Stream B finished")

    print()
    print("=" * 65)
    print(f"{'':4} {'GPU':10} {'TTFA':>8} {'Audio':>8} {'Gen':>8} {'RTF':>6} {'Chunks':>7} {'Wall':>8}")
    print("-" * 65)

    for label in ["A", "B"]:
        r = results.get(label, {})
        if r.get("error"):
            print(f"  {label}:  ERROR: {r['error'][:60]}")
            continue
        ttfa = r.get("ttfa", 0)
        ad = r.get("audio_dur", 0)
        gt = r.get("gen_time", 0)
        rtf = gt / ad if ad > 0 else 0
        gpu = r.get("gpu", "?")
        chunks = r.get("chunks", 0)
        wall = r.get("wall_time", 0)
        print(f"  {label}:  {gpu:10} {ttfa:7.2f}s {ad:7.2f}s {gt:7.2f}s {rtf:5.2f}x {chunks:6d}  {wall:7.2f}s")

    print("-" * 65)

    # Overlap analysis
    a = results.get("A", {})
    b = results.get("B", {})
    if a.get("start") and b.get("start") and a.get("end") and b.get("end"):
        overlap_start = max(a["start"], b["start"])
        overlap_end = min(a.get("end", 0), b.get("end", 0))
        overlap = max(0, overlap_end - overlap_start)
        total_wall = max(a.get("end", 0), b.get("end", 0)) - min(a["start"], b["start"])

        a_gpu = a.get("gpu", "?")
        b_gpu = b.get("gpu", "?")
        same_gpu = a_gpu == b_gpu

        if same_gpu:
            print(f"\n  Both streams ran on {a_gpu} (serialized — single worker)")
            print(f"  Total wall time: {total_wall:.2f}s")
        else:
            print(f"\n  Streams ran on DIFFERENT GPUs: A={a_gpu}, B={b_gpu} (concurrent!)")
            print(f"  Overlap duration: {overlap:.2f}s")
            print(f"  Total wall time: {total_wall:.2f}s (vs sequential: {a.get('wall_time',0) + b.get('wall_time',0):.2f}s)")

    print()


if __name__ == "__main__":
    main()
