#!/usr/bin/env python3
"""Smoke-test MOSS v1.5 teacher clips for last-word cutoff (STT verify)."""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import wave
from difflib import SequenceMatcher
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/moss-realtime/scripts/legacy"))

from generate_voice_clone_batch import (  # noqa: E402
    openmoss_token_budget,
    preload_reference,
    OPENMOSS_SAMPLING,
)

CORPUS = ROOT / "training/loli_15s_smoke/corpus/texts.jsonl"
REF = ROOT / "data/voices/loli/loli_15s.wav"
OUT_DIR = ROOT / "training/loli_15s_smoke/wavs/v15"
REPORT = ROOT / "training/loli_15s_smoke/eval/smoke_report.json"
STT_URL = "http://localhost:8603/v1/audio/transcriptions"


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def last_word(text: str) -> str:
    words = normalize(text).split()
    return words[-1] if words else ""


def transcribe(wav: Path) -> str:
    with wav.open("rb") as f:
        r = requests.post(
            STT_URL,
            files={"file": (wav.name, f, "audio/wav")},
            data={"model": "base", "language": "en"},
            timeout=120,
        )
    r.raise_for_status()
    return r.json().get("text", "").strip()


def wav_duration(wav: Path) -> float:
    with wave.open(str(wav)) as w:
        return w.getnframes() / w.getframerate()


def generate_raw(container: str, text: str, out: Path, *, use_tokens: bool) -> dict:
    preload_reference(REF)
    payload: dict = {
        "text": text,
        "language": "en",
        "sampling": dict(OPENMOSS_SAMPLING),
        "reference_wav_b64": base64.b64encode(REF.read_bytes()).decode(),
    }
    if use_tokens:
        tokens, max_new = openmoss_token_budget(text, REF)
        payload["tokens"] = tokens
        payload["max_new_tokens"] = max(256, max_new)
    else:
        payload["max_new_tokens"] = 512

    script = f"""
import json, urllib.request, sys
payload = json.loads(sys.stdin.read())
req = urllib.request.Request(
    "http://127.0.0.1:8081/tts",
    data=json.dumps(payload).encode(),
    headers={{"Content-Type": "application/json"}},
)
resp = urllib.request.urlopen(req, timeout=600)
sys.stdout.buffer.write(resp.read())
"""
    proc = subprocess.run(
        ["docker", "exec", "-i", container, "python3", "-c", script],
        input=json.dumps(payload).encode(),
        capture_output=True,
        timeout=600,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.decode()[:300]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(proc.stdout)
    if out.stat().st_size < 1024:
        return {"ok": False, "error": "output too small"}
    return {"ok": True, "bytes": out.stat().st_size, "duration_s": wav_duration(out)}


def check_clip(cid: str, text: str, hyp: str, duration_s: float) -> dict:
    ref_norm = normalize(text)
    hyp_norm = normalize(hyp)
    ref_last = last_word(text)
    hyp_words = hyp_norm.split()
    hyp_last = hyp_words[-1] if hyp_words else ""
    sim = SequenceMatcher(None, ref_norm, hyp_norm).ratio()
    last_ok = (
        ref_last in hyp_words[-3:]
        or hyp_last == ref_last
        or ref_last.rstrip("s") == hyp_last.rstrip("s")
        or SequenceMatcher(None, ref_last, hyp_last).ratio() >= 0.8
    )
    return {
        "id": cid,
        "text": text,
        "transcript": hyp,
        "duration_s": round(duration_s, 2),
        "similarity": round(sim, 3),
        "ref_last_word": ref_last,
        "hyp_last_word": hyp_last,
        "last_word_ok": last_ok,
        "pass": sim >= 0.55 and last_ok,
    }


def main() -> int:
    container = sys.argv[1] if len(sys.argv) > 1 else "speaker-openmoss-tts-0-1"
    rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]

    results = {"container": container, "with_tokens": [], "without_tokens": []}

    print(f"=== Teacher smoke test ({container}, raw :8081) ===\n")

    for mode, use_tokens in [("with_tokens", True), ("without_tokens", False)]:
        print(f"--- {mode} ---")
        subdir = OUT_DIR / mode
        subdir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            cid = row["id"]
            text = row["text"]
            wav = subdir / f"{cid}.wav"
            gen = generate_raw(container, text, wav, use_tokens=use_tokens)
            if not gen.get("ok"):
                print(f"  {cid}: GEN FAIL {gen.get('error')}")
                results[mode].append({"id": cid, "pass": False, "error": gen.get("error")})
                continue
            try:
                hyp = transcribe(wav)
            except Exception as e:
                print(f"  {cid}: STT FAIL {e}")
                results[mode].append({"id": cid, "pass": False, "error": str(e)})
                continue
            chk = check_clip(cid, text, hyp, gen["duration_s"])
            results[mode].append(chk)
            status = "PASS" if chk["pass"] else "FAIL"
            print(
                f"  {cid}: {status} | {gen['duration_s']:.1f}s | "
                f"sim={chk['similarity']:.2f} | last={chk['ref_last_word']!r}→{chk['hyp_last_word']!r}"
            )
        print()

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(results, indent=2))

    wt_pass = sum(1 for r in results["with_tokens"] if r.get("pass"))
    nt_pass = sum(1 for r in results["without_tokens"] if r.get("pass"))
    print(f"Summary: with_tokens {wt_pass}/{len(rows)} | without_tokens {nt_pass}/{len(rows)}")
    print(f"Report: {REPORT}")
    return 0 if wt_pass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
