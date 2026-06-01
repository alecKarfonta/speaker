#!/usr/bin/env python3
"""A/B test MOSS-RT env configs: restart container, warmup, quick benchmark."""

from __future__ import annotations

import json
import struct
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"
OVERRIDE = ROOT / "docker-compose.moss-ab.override.yml"
OUT = ROOT / "tests/output/loli_rt_ab"
API = "http://localhost:8013"

QUICK_TEXTS = [
    ("short", "Hello everyone, welcome back!"),
    ("medium",
     "Good morning, sunshine! The birds are singing and the dew still sparkles on the grass. "
     "Can you hear me clearly? The meadow is so peaceful this morning."),
    ("long",
     "Once upon a time, there was a little bunny who lived in a cozy burrow under an old oak tree. "
     "Every day, the bunny would hop through the meadow looking for yummy carrots and making friends "
     "with the butterflies. One morning, the bunny discovered a hidden path leading to a sparkling pond "
     "where fireflies danced at dusk. It was the most magical place the bunny had ever seen."),
]

CONFIGS = [
    {"name": "bf16_baseline", "MOSS_RT_DTYPE": "bf16", "MOSS_RT_QUANTIZE": "none",
     "MOSS_RT_FAST_MODE": "false", "MOSS_RT_GREEDY": "false"},
    {"name": "fp16", "MOSS_RT_DTYPE": "fp16", "MOSS_RT_QUANTIZE": "none",
     "MOSS_RT_FAST_MODE": "false", "MOSS_RT_GREEDY": "false"},
    {"name": "fp16_fast", "MOSS_RT_DTYPE": "fp16", "MOSS_RT_QUANTIZE": "none",
     "MOSS_RT_FAST_MODE": "true", "MOSS_RT_GREEDY": "false"},
    {"name": "fp16_greedy", "MOSS_RT_DTYPE": "fp16", "MOSS_RT_QUANTIZE": "none",
     "MOSS_RT_FAST_MODE": "false", "MOSS_RT_GREEDY": "true"},
    {"name": "8bit", "MOSS_RT_DTYPE": "bf16", "MOSS_RT_QUANTIZE": "8bit",
     "MOSS_RT_FAST_MODE": "false", "MOSS_RT_GREEDY": "false"},
    {"name": "4bit", "MOSS_RT_DTYPE": "bf16", "MOSS_RT_QUANTIZE": "4bit",
     "MOSS_RT_FAST_MODE": "false", "MOSS_RT_GREEDY": "false"},
    {"name": "fp16_8bit_fast", "MOSS_RT_DTYPE": "fp16", "MOSS_RT_QUANTIZE": "8bit",
     "MOSS_RT_FAST_MODE": "true", "MOSS_RT_GREEDY": "false"},
]


@dataclass
class Row:
    config: str
    label: str
    tts_rtf: float
    tts_gen_s: float
    tts_audio_s: float
    stream_ttfa_ms: float
    stream_rtf: float
    stream_sustained: float
    ok: bool
    error: str = ""


def _write_override(overrides: dict[str, str]) -> None:
    """Write override with full merged env (compose replaces env lists on merge)."""
    data = yaml.safe_load(COMPOSE.read_text())
    env_list = data["services"]["moss-tts"]["environment"]
    env_map: dict[str, str] = {}
    for item in env_list:
        if isinstance(item, str) and "=" in item:
            k, v = item.split("=", 1)
            env_map[k] = v
    env_map.update(overrides)
    lines = ["services:", "  moss-tts:", "    environment:"]
    for k, v in env_map.items():
        lines.append(f"      - {k}={v}")
    OVERRIDE.write_text("\n".join(lines) + "\n")


def _restart_and_wait(timeout_s: int = 600) -> tuple[bool, str]:
    subprocess.run(
        [
            "docker", "compose",
            "-f", str(COMPOSE),
            "-f", str(OVERRIDE),
            "up", "-d", "moss-tts", "--force-recreate",
        ],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{API}/health", timeout=5)
            if r.status_code == 200:
                logs = subprocess.run(
                    ["docker", "logs", "speaker-moss-tts-1"],
                    capture_output=True, text=True,
                )
                blob = logs.stdout + logs.stderr
                if "Failed to load MOSS-TTS-Realtime" in blob or "Failed to load worker" in blob:
                    return False, "worker load failed"
                if "[Warmup]" in blob and "Done in" in blob:
                    return True, "warmup done"
                if time.time() - t0 > 120 and r.json().get("realtime_enabled"):
                    return True, "healthy (warmup may be partial)"
        except Exception:
            pass
        time.sleep(10)
    return False, "timeout"


def _tts(text: str) -> tuple[float, float, bool, str]:
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{API}/tts",
            json={"text": text, "voice_name": "loli", "language": "en"},
            timeout=300,
        )
        gen = time.perf_counter() - t0
        if r.status_code != 200:
            return 0.0, gen, False, f"HTTP {r.status_code}: {r.text[:200]}"
        dur = float(r.headers.get("X-Audio-Duration", 0))
        if dur <= 0:
            return 0.0, gen, False, "zero audio duration"
        return dur, gen, True, ""
    except Exception as e:
        return 0.0, time.perf_counter() - t0, False, str(e)


def _stream(text: str) -> tuple[float, float, float, float, bool, str]:
    t0 = time.perf_counter()
    ttfa = None
    buf = b""
    try:
        with requests.post(
            f"{API}/tts/stream",
            json={"text": text, "voice_name": "loli", "language": "en"},
            stream=True,
            timeout=300,
        ) as resp:
            if resp.status_code != 200:
                return 0, 0, 0, 0, False, f"HTTP {resp.status_code}"
            for chunk in resp.iter_content(4096):
                if not chunk:
                    continue
                buf += chunk
                off = 0
                while off + 8 <= len(buf):
                    al, ml = struct.unpack_from("<II", buf, off)
                    fs = 8 + al + ml
                    if len(buf) < off + fs:
                        break
                    if ttfa is None and al > 0:
                        ttfa = time.perf_counter() - t0
                    off += fs
        total = time.perf_counter() - t0
        # parse duration from pcm
        import io, wave
        pcm = bytearray()
        sr = 24000
        off = 0
        while off + 8 <= len(buf):
            al, ml = struct.unpack_from("<II", buf, off)
            off += 8
            if al > 0:
                w = buf[off:off + al]
                off += al
                try:
                    with wave.open(io.BytesIO(w), "rb") as wf:
                        sr = wf.getframerate()
                        pcm.extend(wf.readframes(wf.getnframes()))
                except Exception:
                    pcm.extend(w)
            else:
                off += al
            off += ml
        dur = len(pcm) / (sr * 2) if pcm else 0.0
        if dur <= 0:
            return 0, total, 0, 0, False, "zero stream audio"
        ttfa = ttfa or total
        rtf = total / dur
        sustained = max(total - ttfa, 0) / dur
        return (ttfa * 1000, total, dur, rtf, True, "")
    except Exception as e:
        return 0, time.perf_counter() - t0, 0, 0, False, str(e)


def run_config(cfg: dict) -> list[Row]:
    print(f"\n{'='*60}\nCONFIG: {cfg['name']}\n{'='*60}")
    overrides = {k: v for k, v in cfg.items() if k != "name"}
    _write_override(overrides)
    ok, msg = _restart_and_wait()
    if not ok:
        print(f"  SKIP: {msg}")
        return [Row(cfg["name"], label, 0, 0, 0, 0, 0, 0, False, msg) for label, _ in QUICK_TEXTS]

    print(f"  Service ready: {msg}")
    # warmup
    _tts("Hello.")
    rows = []
    for label, text in QUICK_TEXTS:
        adur, gen, tok, err = _tts(text)
        ttfa, stot, sdur, srtf, sok, serr = _stream(text)
        row = Row(
            config=cfg["name"],
            label=label,
            tts_rtf=gen / adur if adur > 0 else 99,
            tts_gen_s=gen,
            tts_audio_s=adur,
            stream_ttfa_ms=ttfa,
            stream_rtf=srtf,
            stream_sustained=max(stot - ttfa / 1000, 0) / sdur if sdur > 0 else 99,
            ok=tok and sok,
            error=err or serr,
        )
        rows.append(row)
        print(
            f"  {label:6s} | /tts RTF {row.tts_rtf:.2f}x ({gen:.1f}s/{adur:.1f}s) "
            f"| stream TTFA {ttfa:.0f}ms RTF {srtf:.2f}x sustained {row.stream_sustained:.2f}x"
            + (f" ERR: {row.error}" if row.error else "")
        )
        time.sleep(0.5)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[Row] = []
    for cfg in CONFIGS:
        try:
            all_rows.extend(run_config(cfg))
        except Exception as e:
            print(f"  CONFIG FAILED: {e}")

    report = [asdict(r) for r in all_rows]
    (OUT / "ab_results.json").write_text(json.dumps(report, indent=2))

    # rank by medium /tts RTF among ok rows
    medium = [r for r in all_rows if r.label == "medium" and r.ok]
    medium.sort(key=lambda r: r.tts_rtf)
    print("\n" + "=" * 60)
    print("RANKING (medium /tts RTF, lower = faster):")
    for r in medium:
        print(f"  {r.config:16s} RTF={r.tts_rtf:.2f}x  stream_ttfa={r.stream_ttfa_ms:.0f}ms  sustained={r.stream_sustained:.2f}x")

    if medium:
        best = medium[0]["config"] if isinstance(medium[0], dict) else medium[0].config
        print(f"\nBEST: {best}")
    print(f"\nFull results: {OUT / 'ab_results.json'}")


if __name__ == "__main__":
    main()
