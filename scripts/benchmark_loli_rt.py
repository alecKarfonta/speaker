#!/usr/bin/env python3
"""Generate loli native-voice samples and benchmark /tts vs /tts/stream (TTFA, RTF)."""

from __future__ import annotations

import argparse
import io
import json
import re
import struct
import subprocess
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

API_URL = "http://localhost:8013"
VOICE = "loli"
CONTAINER = "speaker-moss-tts-1"
OUT_DIR = Path(__file__).resolve().parents[1] / "tests/output/loli_rt_benchmark"
DEFAULT_OUT_DIR = OUT_DIR

# Listening samples (varied tone/length)
LISTEN_SAMPLES = [
    ("01_greeting", "Hi everyone! Thanks for listening — hope you enjoy the story!"),
    ("02_fun", "Oh wow, that is so cool! I can't believe it actually worked!"),
    ("03_wonder", "Once upon a time, there was a little bunny who lived in a cozy burrow."),
    ("04_soft", "Good night, sleep tight. I'll see you in the morning, okay?"),
    ("05_excited", "We did it! We really did it! This is the best day ever!"),
]

# Benchmark texts by approximate word count
BENCHMARK_TEXTS = [
    ("short_5w", "Hello everyone, welcome back!"),
    ("medium_25w",
     "Good morning, sunshine! The birds are singing and the dew still sparkles on the grass. "
     "Can you hear me clearly? The meadow is so peaceful this morning."),
    ("long_60w",
     "Once upon a time, there was a little bunny who lived in a cozy burrow under an old oak tree. "
     "Every day, the bunny would hop through the meadow looking for yummy carrots and making friends "
     "with the butterflies. One morning, the bunny discovered a hidden path leading to a sparkling pond "
     "where fireflies danced at dusk. It was the most magical place the bunny had ever seen."),
    ("xlong_120w",
     "Let me tell you a story about a brave little explorer who set out across the whispering woods "
     "with nothing but a lantern and a heart full of curiosity. The trees grew taller with every step, "
     "their branches weaving a canopy that filtered the sunlight into golden ribbons on the forest floor. "
     "Somewhere ahead, a brook sang its endless song, and the explorer followed the sound until the trees "
     "parted to reveal a clearing bathed in warm afternoon light. In the center stood an ancient stone "
     "well, covered in moss and mystery, and carved around its rim were words in a language nobody had "
     "spoken for a thousand years. The explorer leaned closer, held the lantern high, and read aloud the "
     "first line — and the whole forest seemed to hold its breath, waiting to hear what happened next."),
]


@dataclass
class TtsResult:
    label: str
    text_words: int
    text_chars: int
    audio_duration_s: float
    generation_time_s: float
    rtf: float
    wav_path: str
    http_status: int


@dataclass
class StreamResult:
    label: str
    text_words: int
    text_chars: int
    ttfa_s: float
    total_time_s: float
    audio_duration_s: float
    rtf: float
    sustained_rtf: float
    num_chunks: int
    wav_path: str
    http_status: int
    infer_s: float | None = None
    decode_s: float | None = None
    iteration: int = 1


def _word_count(text: str) -> int:
    return len(text.split())


def _parse_stream_timing_logs(container: str, since_seconds: int = 120) -> dict[str, float]:
    """Parse infer/decode split from moss-tts container logs."""
    try:
        proc = subprocess.run(
            ["docker", "logs", "--since", f"{since_seconds}s", container],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    text = proc.stdout + proc.stderr
    matches = re.findall(
        r"\[Stream-RT\]\[Timing\] infer=([\d.]+)s \(\d+ calls\), decode=([\d.]+)s",
        text,
    )
    if not matches:
        return {}
    infer_s, decode_s = matches[-1]
    return {"infer_s": float(infer_s), "decode_s": float(decode_s)}


def _sustained_rtf(total_s: float, ttfa_s: float, audio_s: float) -> float:
    if audio_s <= 0:
        return 0.0
    gen_after_ttfa = max(total_s - ttfa_s, 0.0)
    return gen_after_ttfa / audio_s


def _save_wav_from_stream(raw: bytes) -> tuple[bytes, float, int]:
    """Parse stream framing; return combined wav bytes, duration, chunk count."""
    offset = 0
    all_pcm = bytearray()
    sample_rate = 24000
    chunks = 0

    while offset + 8 <= len(raw):
        audio_len, meta_len = struct.unpack_from("<II", raw, offset)
        offset += 8
        if audio_len > 0:
            wav_data = raw[offset : offset + audio_len]
            offset += audio_len
            chunks += 1
            try:
                with wave.open(io.BytesIO(wav_data), "rb") as wf:
                    sample_rate = wf.getframerate()
                    all_pcm.extend(wf.readframes(wf.getnframes()))
            except Exception:
                all_pcm.extend(wav_data)
        else:
            offset += audio_len
        offset += meta_len

    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(all_pcm))

    duration = len(all_pcm) / (sample_rate * 2) if all_pcm else 0.0
    return wav_buf.getvalue(), duration, chunks


def _stream_with_ttfa(text: str) -> tuple[bytes, float, float, float, int]:
    """Stream TTS; return raw bytes, ttfa, total time, audio duration, chunk count."""
    t0 = time.perf_counter()
    ttfa: float | None = None
    buf = b""

    with requests.post(
        f"{API_URL}/tts/stream",
        json={"text": text, "voice_name": VOICE, "language": "en"},
        stream=True,
        timeout=600,
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            buf += chunk
            # Detect first audio frame as it arrives
            parse_off = 0
            while parse_off + 8 <= len(buf):
                audio_len, meta_len = struct.unpack_from("<II", buf, parse_off)
                frame_size = 8 + audio_len + meta_len
                if len(buf) < parse_off + frame_size:
                    break
                if ttfa is None and audio_len > 0:
                    ttfa = time.perf_counter() - t0
                parse_off += frame_size
            # Keep only unparsed tail for next iteration (full re-parse is fine for benchmark)
            # We only need ttfa on first audio frame; full body collected in buf

    total = time.perf_counter() - t0
    wav_bytes, duration, num_chunks = _save_wav_from_stream(buf)
    return wav_bytes, ttfa or total, total, duration, num_chunks


def benchmark_tts(label: str, text: str, out_path: Path) -> TtsResult:
    t0 = time.perf_counter()
    resp = requests.post(
        f"{API_URL}/tts",
        json={"text": text, "voice_name": VOICE, "language": "en"},
        timeout=600,
    )
    gen_time = time.perf_counter() - t0
    if resp.status_code == 200:
        out_path.write_bytes(resp.content)
        audio_dur = float(resp.headers.get("X-Audio-Duration", 0))
        gen_hdr = float(resp.headers.get("X-Generation-Time", gen_time))
        gen_time = gen_hdr
    else:
        audio_dur = 0.0

    words = _word_count(text)
    rtf = gen_time / audio_dur if audio_dur > 0 else 0.0
    return TtsResult(
        label=label,
        text_words=words,
        text_chars=len(text),
        audio_duration_s=audio_dur,
        generation_time_s=gen_time,
        rtf=rtf,
        wav_path=str(out_path),
        http_status=resp.status_code,
    )


def benchmark_stream(
    label: str,
    text: str,
    out_path: Path,
    *,
    iteration: int = 1,
    save_wav: bool = True,
    parse_logs: bool = False,
) -> StreamResult:
    wav_bytes, ttfa, total, duration, chunks = _stream_with_ttfa(text)
    if save_wav and wav_bytes:
        out_path.write_bytes(wav_bytes)
    words = _word_count(text)
    rtf = total / duration if duration > 0 else 0.0
    timing = _parse_stream_timing_logs(CONTAINER) if parse_logs else {}
    return StreamResult(
        label=label,
        text_words=words,
        text_chars=len(text),
        ttfa_s=ttfa,
        total_time_s=total,
        audio_duration_s=duration,
        rtf=rtf,
        sustained_rtf=_sustained_rtf(total, ttfa, duration),
        num_chunks=chunks,
        wav_path=str(out_path) if save_wav else "",
        http_status=200 if wav_bytes else 500,
        infer_s=timing.get("infer_s"),
        decode_s=timing.get("decode_s"),
        iteration=iteration,
    )


def run_profile(api_url: str, iterations: int) -> int:
    global API_URL
    API_URL = api_url
    print(f"=== Profile mode ({iterations} iterations) ===\n")
    print("Warmup stream...")
    _stream_with_ttfa("Hello.")

    results: list[StreamResult] = []
    for it in range(1, iterations + 1):
        print(f"\n--- Iteration {it}/{iterations} ---")
        for label, text in BENCHMARK_TEXTS:
            r = benchmark_stream(
                label, text, OUT_DIR / "bench_stream" / f"{label}.wav",
                iteration=it,
                save_wav=(it == iterations),
                parse_logs=(it == iterations),
            )
            results.append(r)
            infer = f" infer={r.infer_s:.2f}s" if r.infer_s else ""
            decode = f" decode={r.decode_s:.2f}s" if r.decode_s else ""
            print(
                f"  {label:12} TTFA {r.ttfa_s*1000:6.0f}ms | "
                f"RTF {r.rtf:.2f}x | sustained {r.sustained_rtf:.2f}x{infer}{decode}"
            )

    profile_path = OUT_DIR / "profile_report.json"
    profile_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"\nProfile report: {profile_path}")
    return 0


def main() -> int:
    global API_URL, OUT_DIR
    parser = argparse.ArgumentParser(description="Benchmark loli MOSS-Realtime TTS")
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--out-dir", default=None, help="Output directory for samples and report")
    parser.add_argument("--profile", action="store_true", help="Run profile iterations only")
    parser.add_argument("--iterations", type=int, default=3, help="Profile iterations")
    args = parser.parse_args()

    API_URL = args.api_url
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    if args.profile:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "bench_stream").mkdir(exist_ok=True)
        return run_profile(args.api_url, args.iterations)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    listen_dir = OUT_DIR / "samples"
    listen_dir.mkdir(exist_ok=True)
    tts_dir = OUT_DIR / "bench_tts"
    stream_dir = OUT_DIR / "bench_stream"
    tts_dir.mkdir(exist_ok=True)
    stream_dir.mkdir(exist_ok=True)

    print(f"API: {API_URL}  voice: {VOICE}")
    print(f"Output: {OUT_DIR}\n")

    # Warmup (JIT / compile on first request)
    print("Warmup request...")
    try:
        requests.post(
            f"{API_URL}/tts",
            json={"text": "Hello.", "voice_name": VOICE},
            timeout=120,
        )
    except Exception as e:
        print(f"Warmup failed: {e}")
        return 1

    # Listening samples via /tts
    print("\n=== Listening samples (/tts) ===")
    for name, text in LISTEN_SAMPLES:
        path = listen_dir / f"{name}.wav"
        r = benchmark_tts(name, text, path)
        print(
            f"  {name}: {r.audio_duration_s:.1f}s audio in {r.generation_time_s:.1f}s "
            f"(RTF {r.rtf:.2f}x) → {path.name}"
        )

    # Speed benchmark at different lengths
    print("\n=== /tts speed by text length ===")
    tts_results: list[TtsResult] = []
    for label, text in BENCHMARK_TEXTS:
        path = tts_dir / f"{label}.wav"
        r = benchmark_tts(label, text, path)
        tts_results.append(r)
        print(
            f"  {label:12} {r.text_words:4}w {r.text_chars:4}c | "
            f"audio {r.audio_duration_s:5.1f}s | gen {r.generation_time_s:5.1f}s | "
            f"RTF {r.rtf:.2f}x"
        )

    print("\n=== /tts/stream TTFA + speed by text length ===")
    stream_results: list[StreamResult] = []
    for label, text in BENCHMARK_TEXTS:
        path = stream_dir / f"{label}.wav"
        r = benchmark_stream(label, text, path)
        stream_results.append(r)
        print(
            f"  {label:12} {r.text_words:4}w | TTFA {r.ttfa_s*1000:6.0f}ms | "
            f"total {r.total_time_s:5.1f}s | audio {r.audio_duration_s:5.1f}s | "
            f"RTF {r.rtf:.2f}x | sustained {r.sustained_rtf:.2f}x | {r.num_chunks} chunks"
        )

    report = {
        "api_url": API_URL,
        "voice": VOICE,
        "listen_samples": [str(listen_dir / f"{n}.wav") for n, _ in LISTEN_SAMPLES],
        "tts_benchmark": [asdict(r) for r in tts_results],
        "stream_benchmark": [asdict(r) for r in stream_results],
    }
    report_path = OUT_DIR / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown summary for easy reading
    md_lines = [
        "# Loli Realtime Benchmark",
        "",
        f"API: `{API_URL}` | Voice: `{VOICE}` (native LoRA, no ref)",
        "",
        "## Listening samples",
        "",
        "| File | Text |",
        "|------|------|",
    ]
    for name, text in LISTEN_SAMPLES:
        md_lines.append(f"| `{listen_dir / (name + '.wav')}` | {text[:60]}... |")

    md_lines += [
        "",
        "## /tts batch generation",
        "",
        "| Label | Words | Audio (s) | Gen (s) | RTF |",
        "|-------|-------|-----------|---------|-----|",
    ]
    for r in tts_results:
        md_lines.append(
            f"| {r.label} | {r.text_words} | {r.audio_duration_s:.1f} | "
            f"{r.generation_time_s:.1f} | {r.rtf:.2f}x |"
        )

    md_lines += [
        "",
        "## /tts/stream (TTFA)",
        "",
        "| Label | Words | TTFA (ms) | Total (s) | Audio (s) | RTF | Sustained RTF | Chunks |",
        "|-------|-------|-----------|-----------|-----------|-----|---------------|--------|",
    ]
    for r in stream_results:
        md_lines.append(
            f"| {r.label} | {r.text_words} | {r.ttfa_s*1000:.0f} | {r.total_time_s:.1f} | "
            f"{r.audio_duration_s:.1f} | {r.rtf:.2f}x | {r.sustained_rtf:.2f}x | {r.num_chunks} |"
        )

    md_path = OUT_DIR / "benchmark_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\nReport: {report_path}")
    print(f"        {md_path}")
    print(f"\nListen: {listen_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
