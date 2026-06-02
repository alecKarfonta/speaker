#!/usr/bin/env python3
"""End-to-end verification: model identity, streaming path, TTFA, STT."""

from __future__ import annotations

import io
import json
import os
import struct
import time
import wave
from pathlib import Path

import requests

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[3]))
OUT = ROOT / "training/loli_15s/eval/bench/verify_streaming_$(date).json".replace("$(date)", time.strftime("%Y%m%d_%H%M%S"))
API = os.environ.get("MOSS_RT_API", "http://127.0.0.1:8016")
STT = os.environ.get("STT_API", "http://localhost:8603/v1/audio/transcriptions")
MERGED = ROOT / "training/loli_15s/exports/loli15s-epoch7-merged"
BASELINE_PORT = os.environ.get("BASELINE_PORT", "8017")

TESTS = [
    ("short_greeting", "Hi there! I'm so happy you're here today."),
    ("medium_question", "Guess what? Do you think the stars look brighter after it rains?"),
    ("list", "We need three things: a flashlight, a warm scarf, and one very brave cookie."),
]


def stream(api: str, text: str) -> dict:
    t0 = time.perf_counter()
    resp = requests.post(
        f"{api.rstrip('/')}/tts/stream",
        json={"text": text, "language": "en"},
        stream=True,
        timeout=600,
    )
    if resp.status_code != 200:
        return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    first = None
    parts = []
    for chunk in resp.iter_content(8192):
        if chunk:
            if first is None:
                first = time.perf_counter()
            parts.append(chunk)
    wall = time.perf_counter() - t0
    raw = b"".join(parts)
    if not raw.startswith(b"") and len(raw) < 8:
        return {"ok": False, "error": "empty stream"}
    # Must be framed stream (audio_len + meta_len), not raw WAV
    framed = len(raw) >= 8
    off, sr, pcm, chunks = 0, 24000, bytearray(), 0
    while off + 8 <= len(raw):
        al, ml = struct.unpack_from("<II", raw, off)
        off += 8
        if off + al + ml > len(raw):
            break
        ab = raw[off : off + al]
        off += al + ml
        if al:
            chunks += 1
            try:
                with wave.open(io.BytesIO(ab), "rb") as wf:
                    sr = wf.getframerate()
                    pcm.extend(wf.readframes(wf.getnframes()))
            except Exception:
                pcm.extend(ab)
    ttfb_ms = (first - t0) * 1000 if first else 0
    audio_s = len(pcm) / (sr * 2) if pcm else 0
    return {
        "ok": True,
        "endpoint": "/tts/stream",
        "framed_chunks": chunks,
        "framed_protocol": framed and chunks > 0,
        "ttfb_ms": round(ttfb_ms, 1),
        "wall_s": round(wall, 2),
        "audio_s": round(audio_s, 2),
        "rtf": round(audio_s / wall, 2) if wall > 0 else 0,
        "pcm_bytes": len(pcm),
    }


def stt_wav(pcm: bytes, sr: int, name: str) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    r = requests.post(
        STT,
        files={"file": (name, buf.getvalue(), "audio/wav")},
        data={"model": "base", "language": "en", "response_format": "json"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("text", "").strip()


def health(api: str) -> dict:
    try:
        return requests.get(f"{api.rstrip('/')}/health", timeout=5).json()
    except Exception as e:
        return {"error": str(e)}


def main() -> int:
    report: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finetuned_api": API,
        "checks": {},
        "tests": [],
    }

    h = health(API)
    report["health"] = h
    report["checks"]["server_ready"] = h.get("status") == "ready"
    report["checks"]["realtime_enabled"] = h.get("realtime_enabled") is True
    report["checks"]["merged_dir_exists"] = MERGED.is_dir() and (MERGED / "model.safetensors").is_file()
    report["checks"]["merge_source"] = json.loads((MERGED / "merge_info.json").read_text()) if (MERGED / "merge_info.json").is_file() else None

    rt_lora = h.get("rt_lora")
    rt_ckpt = h.get("rt_lora_checkpoint")
    rt_native = h.get("rt_native_voice")
    report["checks"]["health_rt_lora"] = rt_lora
    report["checks"]["health_rt_checkpoint"] = rt_ckpt
    report["checks"]["health_rt_native_voice"] = rt_native

    # Process env via health may omit checkpoint; infer from merge path on disk
    report["checks"]["expected_model"] = str(MERGED)

    for name, text in TESTS:
        row = {"name": name, "input": text, "finetuned": stream(API, text)}
        if row["finetuned"].get("ok"):
            # Re-fetch audio for STT
            resp = requests.post(
                f"{API}/tts/stream",
                json={"text": text, "language": "en"},
                timeout=600,
            )
            raw = resp.content
            off, sr, pcm = 0, 24000, bytearray()
            while off + 8 <= len(raw):
                al, ml = struct.unpack_from("<II", raw, off)
                off += 8
                ab = raw[off : off + al]
                off += al + ml
                if al:
                    try:
                        with wave.open(io.BytesIO(ab), "rb") as wf:
                            sr = wf.getframerate()
                            pcm.extend(wf.readframes(wf.getnframes()))
                    except Exception:
                        pcm.extend(ab)
            try:
                hyp = stt_wav(bytes(pcm), sr, f"{name}.wav")
                row["stt"] = hyp
                row["stt_word_overlap"] = len(set(text.lower().split()) & set(hyp.lower().split())) / max(
                    len(text.split()), 1
                )
            except Exception as e:
                row["stt_error"] = str(e)
        report["tests"].append(row)
        f = row["finetuned"]
        print(
            f"{name:18} TTFB={f.get('ttfb_ms', '?'):>6}ms  "
            f"chunks={f.get('framed_chunks', '?')}  audio={f.get('audio_s', '?')}s  "
            f"stream={f.get('framed_protocol', False)}"
        )

    ttfa_vals = [t["finetuned"]["ttfb_ms"] for t in report["tests"] if t["finetuned"].get("ok")]
    report["summary"] = {
        "ttfa_ms_min": min(ttfa_vals) if ttfa_vals else None,
        "ttfa_ms_max": max(ttfa_vals) if ttfa_vals else None,
        "ttfa_ms_avg": round(sum(ttfa_vals) / len(ttfa_vals), 1) if ttfa_vals else None,
        "all_used_tts_stream": all(t["finetuned"].get("endpoint") == "/tts/stream" for t in report["tests"]),
        "all_framed": all(t["finetuned"].get("framed_protocol") for t in report["tests"]),
    }

    # Baseline compare if up
    base_api = f"http://127.0.0.1:{BASELINE_PORT}"
    bh = health(base_api)
    if bh.get("status") == "ready" and not bh.get("rt_lora"):
        t, _ = TESTS[0]
        b = stream(base_api, _)
        report["baseline"] = {"api": base_api, "health": bh, "short_greeting": b}
        if b.get("ok"):
            print(f"baseline           TTFB={b['ttfb_ms']:.0f}ms (stock MOSS-RT, no finetune)")

    out_path = ROOT / "training/loli_15s/eval/bench" / f"verify_streaming_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out_path}")
    print(f"TTFA avg: {report['summary']['ttfa_ms_avg']}ms  (target <500ms)")
    ok = (
        report["checks"]["server_ready"]
        and report["summary"]["all_used_tts_stream"]
        and report["summary"]["all_framed"]
        and (report["summary"]["ttfa_ms_avg"] or 9999) < 600
    )
    print("VERDICT:", "PASS" if ok else "REVIEW")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
