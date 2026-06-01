#!/usr/bin/env python3
"""Quick A/B: restart moss-tts with env overrides, benchmark medium+short only."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"
OVERRIDE = ROOT / "docker-compose.moss-ab.override.yml"
API = "http://localhost:8013"
OUT = ROOT / "tests/output/loli_rt_ab"

SHORT = "Hello everyone, welcome back!"
MEDIUM = (
    "Good morning, sunshine! The birds are singing and the dew still sparkles on the grass. "
    "Can you hear me clearly? The meadow is so peaceful this morning."
)


def write_override(overrides: dict[str, str]) -> None:
    data = yaml.safe_load(COMPOSE.read_text())
    env_map = {}
    for item in data["services"]["moss-tts"]["environment"]:
        k, v = item.split("=", 1)
        env_map[k] = v
    env_map.update(overrides)
    lines = ["services:", "  moss-tts:", "    environment:"]
    for k, v in env_map.items():
        lines.append(f"      - {k}={v}")
    OVERRIDE.write_text("\n".join(lines) + "\n")


def restart() -> bool:
    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "-f", str(OVERRIDE),
         "up", "-d", "moss-tts", "--force-recreate"],
        cwd=ROOT, capture_output=True,
    )
    for _ in range(60):
        time.sleep(10)
        try:
            if requests.get(f"{API}/health", timeout=5).status_code == 200:
                logs = subprocess.run(
                    ["docker", "logs", "speaker-moss-tts-1"],
                    capture_output=True, text=True,
                ).stdout + subprocess.run(
                    ["docker", "logs", "speaker-moss-tts-1"],
                    capture_output=True, text=True,
                ).stderr
                if "Failed to load worker" in logs or "Application startup failed" in logs:
                    return False
                if "Warmup" in logs and "Done in" in logs:
                    return True
                if _ > 12:
                    return True
        except Exception:
            pass
    return False


def bench_tts(text: str) -> tuple[float, float]:
    t0 = time.perf_counter()
    r = requests.post(f"{API}/tts", json={"text": text, "voice_name": "loli", "language": "en"}, timeout=300)
    gen = time.perf_counter() - t0
    dur = float(r.headers.get("X-Audio-Duration", 0)) if r.status_code == 200 else 0
    return dur, gen


def bench_stream(text: str) -> tuple[float, float, float]:
    t0 = time.perf_counter()
    ttfa = None
    buf = b""
    with requests.post(
        f"{API}/tts/stream",
        json={"text": text, "voice_name": "loli", "language": "en"},
        stream=True, timeout=300,
    ) as resp:
        for chunk in resp.iter_content(4096):
            buf += chunk
            off = 0
            while off + 8 <= len(buf):
                al, ml = struct.unpack_from("<II", buf, off)
                if len(buf) < off + 8 + al + ml:
                    break
                if ttfa is None and al > 0:
                    ttfa = time.perf_counter() - t0
                off += 8 + al + ml
    total = time.perf_counter() - t0
    import io, wave
    pcm = bytearray()
    sr = 24000
    off = 0
    while off + 8 <= len(buf):
        al, ml = struct.unpack_from("<II", buf, off)
        off += 8
        if al:
            try:
                with wave.open(io.BytesIO(buf[off:off + al]), "rb") as wf:
                    sr = wf.getframerate()
                    pcm.extend(wf.readframes(wf.getnframes()))
            except Exception:
                pass
            off += al
        off += ml
    dur = len(pcm) / (sr * 2) if pcm else 0
    return (ttfa or total) * 1000, total, dur


def run_cfg(name: str, overrides: dict[str, str]) -> dict:
    print(f"\n=== {name} ===", flush=True)
    write_override(overrides)
    if not restart():
        print("  LOAD FAILED", flush=True)
        return {"name": name, "ok": False}
    bench_tts("Hi.")  # warmup
    sd, sg = bench_tts(SHORT)
    md, mg = bench_tts(MEDIUM)
    ttfa, st, sdur = bench_stream(MEDIUM)
    row = {
        "name": name,
        "ok": sd > 0 and md > 0 and sdur > 0,
        "short_rtf": sg / sd if sd else 99,
        "medium_rtf": mg / md if md else 99,
        "stream_ttfa_ms": ttfa,
        "stream_rtf": st / sdur if sdur else 99,
        "stream_sustained": max(st - ttfa / 1000, 0) / sdur if sdur else 99,
    }
    print(
        f"  short={row['short_rtf']:.2f}x medium={row['medium_rtf']:.2f}x "
        f"stream_ttfa={ttfa:.0f}ms stream={row['stream_rtf']:.2f}x sustained={row['stream_sustained']:.2f}x",
        flush=True,
    )
    return row


def main():
    configs = [
        ("bf16_baseline", {"MOSS_RT_DTYPE": "bf16", "MOSS_RT_QUANTIZE": "none",
                           "MOSS_RT_FAST_MODE": "false", "MOSS_RT_GREEDY": "false"}),
        ("fp16", {"MOSS_RT_DTYPE": "fp16", "MOSS_RT_QUANTIZE": "none",
                  "MOSS_RT_FAST_MODE": "false", "MOSS_RT_GREEDY": "false"}),
        ("greedy", {"MOSS_RT_DTYPE": "bf16", "MOSS_RT_QUANTIZE": "none",
                    "MOSS_RT_FAST_MODE": "false", "MOSS_RT_GREEDY": "true"}),
        ("fast", {"MOSS_RT_DTYPE": "bf16", "MOSS_RT_QUANTIZE": "none",
                  "MOSS_RT_FAST_MODE": "true", "MOSS_RT_GREEDY": "false"}),
        ("8bit", {"MOSS_RT_DTYPE": "bf16", "MOSS_RT_QUANTIZE": "8bit",
                  "MOSS_RT_FAST_MODE": "false", "MOSS_RT_GREEDY": "false"}),
        ("fp16_greedy", {"MOSS_RT_DTYPE": "fp16", "MOSS_RT_QUANTIZE": "none",
                         "MOSS_RT_FAST_MODE": "false", "MOSS_RT_GREEDY": "true"}),
    ]
    if len(sys.argv) > 1:
        names = set(sys.argv[1:])
        configs = [c for c in configs if c[0] in names]

    OUT.mkdir(parents=True, exist_ok=True)
    results = [run_cfg(n, o) for n, o in configs]
    (OUT / "quick_ab.json").write_text(json.dumps(results, indent=2))

    ok = [r for r in results if r.get("ok")]
    ok.sort(key=lambda r: r["medium_rtf"])
    print("\nRANKING (medium /tts RTF):", flush=True)
    for r in ok:
        print(f"  {r['name']:14s} medium={r['medium_rtf']:.2f}x short={r['short_rtf']:.2f}x "
              f"ttfa={r['stream_ttfa_ms']:.0f}ms sustained={r['stream_sustained']:.2f}x", flush=True)

    if ok:
        best = ok[0]["name"]
        print(f"\nBEST: {best}", flush=True)
        # restore best config
        best_cfg = next(o for n, o in configs if n == best)
        write_override(best_cfg)


if __name__ == "__main__":
    main()
