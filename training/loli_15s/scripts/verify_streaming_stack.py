#!/usr/bin/env python3
"""Verify MOSS-RT streaming: TTFA, framed protocol, STT completeness (last-word cutoff)."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import struct
import time
import wave
from pathlib import Path

import requests

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[3]))
LOLI = ROOT / "training/loli_15s"
DEFAULT_OUT = LOLI / "eval/bench"
API = os.environ.get("MOSS_RT_API", "http://127.0.0.1:8016")
STT = os.environ.get("STT_API", "http://localhost:8603/v1/audio/transcriptions")
MERGED_E7 = LOLI / "exports/loli15s-epoch7-merged"
MERGED_V2 = LOLI / "exports/loli15s-v2-merged"

TESTS = [
    ("short_greeting", "Hi there! I'm so happy you're here today."),
    ("medium_question", "Guess what? Do you think the stars look brighter after it rains?"),
    ("list", "We need three things: a flashlight, a warm scarf, and one very brave cookie."),
    ("excited", "Wait wait wait! I keep replaying that moment — it still sparkles. I mean it!"),
    ("story", (
        "Once upon a time, in a tiny village between rolling hills, a curious girl wandered past "
        "wildflowers every dawn. She collected shiny pebbles and dreamed of sailing before the sun "
        "turned the water to gold."
    )),
]

TAIL_GAP_FAIL_S = 0.30  # hyp ends this many seconds before audio end → likely cutoff


def _normalize_words(text: str) -> list[str]:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return [w for w in text.split() if w]


def stream_collect(api: str, text: str) -> dict:
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
    if len(raw) < 8:
        return {"ok": False, "error": "empty stream"}
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
    audio_s = len(pcm) / (sr * 2) if pcm and sr else 0
    return {
        "ok": True,
        "endpoint": "/tts/stream",
        "framed_chunks": chunks,
        "framed_protocol": chunks > 0,
        "ttfb_ms": round(ttfb_ms, 1),
        "wall_s": round(wall, 2),
        "audio_s": round(audio_s, 2),
        "sample_rate": sr,
        "pcm": bytes(pcm),
    }


def stt_verbose(pcm: bytes, sr: int, name: str) -> tuple[str, float | None]:
    """Return (text, last_word_end_s) from verbose_json if available."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)
    data_variants = [
        {"model": "base", "language": "en", "response_format": "verbose_json", "timestamp_granularities[]": "word"},
        {"model": "base", "language": "en", "response_format": "verbose_json"},
        {"model": "base", "language": "en", "response_format": "json"},
    ]
    last_err = ""
    for data in data_variants:
        try:
            r = requests.post(
                STT,
                files={"file": (name, buf.getvalue(), "audio/wav")},
                data=data,
                timeout=120,
            )
            if r.status_code != 200:
                last_err = r.text[:200]
                continue
            payload = r.json()
            text = str(payload.get("text", "")).strip()
            last_end = None
            for seg in payload.get("segments") or []:
                for w in seg.get("words") or []:
                    if w.get("end") is not None:
                        last_end = float(w["end"])
            return text, last_end
        except requests.RequestException as exc:
            last_err = str(exc)
    raise RuntimeError(last_err or "STT failed")


def analyze_completeness(ref_text: str, hyp: str, audio_s: float, last_word_end_s: float | None) -> dict:
    ref_w = _normalize_words(ref_text)
    hyp_w = _normalize_words(hyp)
    ref_last = ref_w[-1] if ref_w else ""
    hyp_last = hyp_w[-1] if hyp_w else ""
    last_word_match = bool(ref_last and hyp_last and ref_last == hyp_last)
    last_word_in_hyp = ref_last in hyp_w if ref_last else None
    tail_gap_s = None
    likely_cutoff = False
    if last_word_end_s is not None and audio_s > 0:
        tail_gap_s = round(max(0.0, audio_s - last_word_end_s), 3)
        likely_cutoff = tail_gap_s < TAIL_GAP_FAIL_S and not last_word_match
    missing_tail_words = max(0, len(ref_w) - len(hyp_w))
    return {
        "ref_last_word": ref_last,
        "hyp_last_word": hyp_last,
        "last_word_match": last_word_match,
        "last_word_in_hyp": last_word_in_hyp,
        "last_word_end_s": last_word_end_s,
        "tail_gap_s": tail_gap_s,
        "likely_cutoff": likely_cutoff,
        "missing_tail_words": missing_tail_words,
        "wer_proxy_overlap": round(len(set(ref_w) & set(hyp_w)) / max(len(ref_w), 1), 3),
    }


def health(api: str) -> dict:
    try:
        return requests.get(f"{api.rstrip('/')}/health", timeout=5).json()
    except Exception as e:
        return {"error": str(e)}


def run_suite(api: str, label: str) -> dict:
    rows = []
    for name, text in TESTS:
        row: dict = {"name": name, "input": text, "label": label}
        fin = stream_collect(api, text)
        row["stream"] = {k: v for k, v in fin.items() if k != "pcm"}
        if not fin.get("ok"):
            rows.append(row)
            continue
        pcm = fin["pcm"]
        sr = fin["sample_rate"]
        try:
            hyp, last_end = stt_verbose(pcm, sr, f"{name}.wav")
            row["stt_hyp"] = hyp[:300]
            row["completeness"] = analyze_completeness(text, hyp, fin["audio_s"], last_end)
        except Exception as exc:
            row["stt_error"] = str(exc)
        rows.append(row)
        c = row.get("completeness", {})
        print(
            f"  {name:18} TTFB={fin.get('ttfb_ms', '?'):>6}ms  "
            f"last_ok={c.get('last_word_match', '?')}  "
            f"tail_gap={c.get('tail_gap_s', '?')}s  "
            f"cutoff={c.get('likely_cutoff', '?')}"
        )
    complete = [r for r in rows if r.get("completeness")]
    n_ok = sum(1 for r in complete if not r["completeness"].get("likely_cutoff"))
    n_last = sum(1 for r in complete if r["completeness"].get("last_word_match"))
    return {
        "label": label,
        "api": api,
        "tests": rows,
        "summary": {
            "n": len(rows),
            "last_word_match_rate": round(n_last / max(len(complete), 1), 3),
            "not_likely_cutoff_rate": round(n_ok / max(len(complete), 1), 3),
            "likely_cutoff_count": sum(1 for r in complete if r["completeness"].get("likely_cutoff")),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=API)
    parser.add_argument("--compare-api", default="", help="Second API for A/B (e.g. v2 merged)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tail_gap_fail_s": TAIL_GAP_FAIL_S,
        "health": health(args.api_url),
        "suites": [],
    }

    print(f"=== Streaming verify: {args.api_url} ===")
    report["suites"].append(run_suite(args.api_url, "primary"))

    if args.compare_api:
        print(f"\n=== Compare: {args.compare_api} ===")
        report["suites"].append(run_suite(args.compare_api, "compare"))

    s0 = report["suites"][0]["summary"]
    report["verdict"] = {
        "pass_last_word_rate": s0["last_word_match_rate"] >= 0.9,
        "pass_not_cutoff_rate": s0["not_likely_cutoff_rate"] >= 0.9,
    }
    out_path = out_dir / f"verify_streaming_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {out_path}")
    print(
        f"Summary: last_word_match={s0['last_word_match_rate']:.0%}  "
        f"not_cutoff={s0['not_likely_cutoff_rate']:.0%}"
    )
    ok = report["verdict"]["pass_last_word_rate"] and report["verdict"]["pass_not_cutoff_rate"]
    print("VERDICT:", "PASS" if ok else "REVIEW")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
