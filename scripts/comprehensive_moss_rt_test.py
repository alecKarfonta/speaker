#!/usr/bin/env python3
"""Comprehensive MOSS-Realtime (loli LoRA) test suite."""

from __future__ import annotations

import io
import json
import struct
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

API = "http://localhost:8016"
OUT = Path(__file__).resolve().parents[1] / "tests/output/loli_rt_comprehensive"
STT_URL = "http://localhost:8603/v1/audio/transcriptions"

EDGE_CASES = [
    ("punctuation", "Wait — really?! Oh... I see. (Okay.)"),
    ("numbers", "There are 3 apples, 42 stars, and 1,024 fireflies tonight."),
    ("short", "Hi."),
    ("question", "What do you think happened next?"),
    ("exclaim", "We did it! This is amazing!"),
]

BENCH_TEXTS = [
    ("short", "Hello everyone, welcome back!"),
    ("medium",
     "Good morning! The birds are singing and the meadow is peaceful this morning."),
    ("long",
     "Once upon a time, there was a little bunny who lived under an old oak tree. "
     "Every day it hopped through the meadow looking for carrots and making friends."),
]


@dataclass
class CaseResult:
    name: str
    ok: bool
    detail: str
    extra: dict | None = None


def wav_info(data: bytes) -> dict:
    with wave.open(io.BytesIO(data), "rb") as wf:
        frames = wf.getnframes()
        sr = wf.getframerate()
        return {
            "sample_rate": sr,
            "channels": wf.getnchannels(),
            "duration_s": round(frames / sr, 3),
            "bytes": len(data),
        }


def post_tts(text: str, timeout: int = 300) -> tuple[int, bytes, dict]:
    t0 = time.perf_counter()
    r = requests.post(f"{API}/tts", json={"text": text, "language": "en"}, timeout=timeout)
    wall = time.perf_counter() - t0
    hdr = {
        "gen_s": float(r.headers.get("X-Generation-Time", 0) or 0),
        "audio_s": float(r.headers.get("X-Audio-Duration", 0) or 0),
        "wall_s": round(wall, 3),
    }
    if hdr["audio_s"] > 0 and hdr["gen_s"] > 0:
        hdr["rtf"] = round(hdr["gen_s"] / hdr["audio_s"], 3)
    return r.status_code, r.content, hdr


def post_stream(text: str, timeout: int = 300) -> tuple[int, bytes, dict]:
    t0 = time.perf_counter()
    r = requests.post(f"{API}/tts/stream", json={"text": text, "language": "en"}, timeout=timeout)
    raw = r.content
    wall = time.perf_counter() - t0
    if r.status_code != 200:
        return r.status_code, raw, {"wall_s": round(wall, 3), "error": raw[:200].decode(errors="replace")}

    offset = 0
    audio_s = 0.0
    chunks = 0
    ttfa_ms = None
    sr = 24000
    first_audio = False
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
            if not first_audio:
                ttfa_ms = round(wall * 1000, 1)  # approximate; wall is total
                first_audio = True
            try:
                meta = json.loads(meta_bytes)
                sr = int(meta.get("sr", meta.get("sample_rate", sr)))
            except Exception:
                pass
            pcm = (audio_len - 44) if audio_bytes[:4] == b"RIFF" else audio_len
            audio_s += (pcm // 2) / sr
            chunks += 1

    meta = {
        "wall_s": round(wall, 3),
        "audio_s": round(audio_s, 3),
        "chunks": chunks,
        "ttfa_ms": ttfa_ms,
    }
    if audio_s > 0:
        meta["rtf"] = round(wall / audio_s, 3)
    return r.status_code, raw, meta


def stt_similarity(text: str, wav: bytes) -> tuple[float | None, str]:
    try:
        r = requests.post(
            STT_URL,
            files={"file": ("test.wav", wav, "audio/wav")},
            data={"model": "base", "language": "en"},
            timeout=120,
        )
        r.raise_for_status()
        trans = r.json().get("text", "").strip()
        orig = set(text.lower().split())
        trans_w = set(trans.lower().split())
        sim = len(orig & trans_w) / len(orig) if orig else 0.0
        return sim, trans
    except Exception as exc:
        return None, str(exc)


def run_subprocess(cmd: list[str], label: str) -> CaseResult:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        ok = proc.returncode == 0
        tail = (proc.stdout + proc.stderr)[-2000:]
        return CaseResult(label, ok, tail)
    except Exception as exc:
        return CaseResult(label, False, str(exc))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "edge").mkdir(exist_ok=True)
    (OUT / "bench").mkdir(exist_ok=True)
    results: list[CaseResult] = []

    # Health
    try:
        h = requests.get(f"{API}/health", timeout=10).json()
        ok = h.get("status") == "ready" and h.get("realtime_enabled")
        results.append(CaseResult("health", ok, json.dumps(h, indent=2)))
    except Exception as exc:
        results.append(CaseResult("health", False, str(exc)))
        _write_report(results)
        return 1

    # Voices
    try:
        v = requests.get(f"{API}/voices", timeout=10)
        results.append(CaseResult("voices", v.status_code == 200, f"status={v.status_code} count={len(v.json())}"))
    except Exception as exc:
        results.append(CaseResult("voices", False, str(exc)))

    # Edge cases — batch /tts
    for name, text in EDGE_CASES:
        code, body, hdr = post_tts(text)
        path = OUT / "edge" / f"{name}.wav"
        ok = code == 200 and len(body) > 1000
        extra = hdr
        if ok:
            path.write_bytes(body)
            extra = {**hdr, **wav_info(body)}
            if extra.get("duration_s", 0) < 0.3:
                ok = False
                extra["fail"] = "audio too short"
        results.append(CaseResult(f"edge_tts/{name}", ok, text[:60], extra))

    # Benchmark texts — batch + stream
    for label, text in BENCH_TEXTS:
        code, body, hdr = post_tts(text)
        ok = code == 200 and hdr.get("audio_s", 0) > 1.0
        if ok:
            (OUT / "bench" / f"{label}_tts.wav").write_bytes(body)
            hdr = {**hdr, **wav_info(body)}
        results.append(CaseResult(f"bench_tts/{label}", ok, f"rtf={hdr.get('rtf')}", hdr))

        sc, sraw, shdr = post_stream(text)
        ok_s = sc == 200 and shdr.get("audio_s", 0) > 1.0 and shdr.get("chunks", 0) >= 1
        if ok_s:
            (OUT / "bench" / f"{label}_stream.bin").write_bytes(sraw)
        results.append(CaseResult(f"bench_stream/{label}", ok_s, f"chunks={shdr.get('chunks')}", shdr))

    # Batch vs stream duration consistency (medium)
    _, batch_wav, batch_hdr = post_tts(BENCH_TEXTS[1][1])
    _, stream_raw, stream_hdr = post_stream(BENCH_TEXTS[1][1])
    if batch_hdr.get("audio_s") and stream_hdr.get("audio_s"):
        ratio = batch_hdr["audio_s"] / stream_hdr["audio_s"]
        ok = 0.7 <= ratio <= 1.3
        results.append(
            CaseResult(
                "consistency_batch_vs_stream",
                ok,
                f"batch={batch_hdr['audio_s']}s stream={stream_hdr['audio_s']}s ratio={ratio:.2f}",
            )
        )

    # Sequential stability — 5 rapid batch requests
    rtts = []
    for i in range(5):
        _, _, hdr = post_tts("Stability check number five.")
        rtts.append(hdr.get("rtf", 99))
    ok = all(r < 2.0 for r in rtts if r) and len(rtts) == 5
    results.append(CaseResult("stability_5x_batch", ok, f"rtfs={rtts}"))

    # STT round-trip (medium text)
    code, wav, _ = post_tts(BENCH_TEXTS[1][1])
    if code == 200:
        sim, trans = stt_similarity(BENCH_TEXTS[1][1], wav)
        if sim is None:
            results.append(CaseResult("stt_roundtrip", True, f"skipped: {trans}"))
        else:
            ok = sim >= 0.5
            results.append(CaseResult("stt_roundtrip", ok, trans[:120], {"similarity": round(sim, 3)}))

    # External scripts
    results.append(
        run_subprocess(
            ["python3", "scripts/test_moss_stream.py", "--api-url", API, "--skip-stt"],
            "script_test_moss_stream",
        )
    )
    results.append(
        run_subprocess(
            ["python3", "training/moss-realtime/scripts/legacy/bench/benchmark_rt_tts.py", "--url", API],
            "script_benchmark_rt_tts",
        )
    )

    passed = sum(1 for r in results if r.ok)
    failed = [r for r in results if not r.ok]
    _write_report(results, passed, failed)
    print(f"\n{'='*60}\nComprehensive test: {passed}/{len(results)} passed")
    print(f"Report: {OUT / 'comprehensive_report.md'}")
    for r in failed:
        print(f"  FAIL {r.name}: {r.detail[:100]}")
    return 0 if not failed else 1


def _write_report(results: list[CaseResult], passed: int = 0, failed: list[CaseResult] | None = None) -> None:
    failed = failed or [r for r in results if not r.ok]
    lines = [
        "# MOSS-Realtime Comprehensive Test Report",
        "",
        f"API: `{API}`",
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Result: {passed or sum(1 for r in results if r.ok)}/{len(results)} passed**",
        "",
        "## Summary",
        "",
        "| Test | Pass | Detail |",
        "|------|------|--------|",
    ]
    for r in results:
        mark = "yes" if r.ok else "**NO**"
        extra = ""
        if r.extra:
            extra = " " + json.dumps(r.extra)
        lines.append(f"| {r.name} | {mark} | {r.detail[:80]}{extra[:60]} |")

    if failed:
        lines.extend(["", "## Failures", ""])
        for r in failed:
            lines.append(f"### {r.name}")
            lines.append(f"- {r.detail}")
            if r.extra:
                lines.append(f"- `{json.dumps(r.extra)}`")
            lines.append("")

    lines.extend(["", f"Artifacts: `{OUT}/`", ""])
    (OUT / "comprehensive_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "comprehensive_report.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
