#!/usr/bin/env python3
"""
Log host health during heavy teacher-gen / GPU workloads.

Writes JSONL samples + scrapes kernel ring buffer for OOM, GPU Xid, hung tasks.
Hard freezes often leave no userspace traceback — this captures the last minutes
before a crash in health.jsonl and kernel_alerts.log.

Usage:
  python3 watchdog_server_health.py --log-dir training/loli_15s_batch3/logs/health
  # or via run_loli15s_teacher_gen_parallel.sh (MONITOR=1)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SPEAKER_ROOT", Path(__file__).resolve().parents[4]))

KERNEL_PATTERNS = [
    r"Out of memory",
    r"Killed process",
    r"oom-kill",
    r"oom_reaper",
    r"NVRM: Xid",
    r"GPU has fallen off the bus",
    r"hung_task",
    r"blocked for more than",
    r"Memory cgroup out of memory",
    r"Underflow",
    r"watchdog:",
    r"soft lockup",
    r"hard LOCKUP",
]


def read_meminfo() -> dict:
    info: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                info[parts[0].rstrip(":")] = int(parts[1])
    except OSError:
        pass
    return info


def read_psi() -> dict:
    out: dict[str, str] = {}
    for kind in ("memory", "cpu", "io"):
        p = Path(f"/proc/pressure/{kind}")
        if not p.is_file():
            continue
        try:
            out[kind] = p.read_text().strip().replace("\n", " | ")
        except OSError:
            pass
    return out


def loadavg() -> list[float]:
    try:
        la = os.getloadavg()
        return [round(x, 2) for x in la]
    except OSError:
        return []


def gpu_stats() -> list[dict]:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return [{"error": r.stderr.strip()[:200]}]
        rows = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            rows.append({
                "index": int(parts[0]),
                "mem_used_mb": int(float(parts[1])),
                "mem_total_mb": int(float(parts[2])),
                "mem_free_mb": int(float(parts[3])),
                "util_pct": int(float(parts[4] or 0)),
                "temp_c": int(float(parts[5] or 0)),
                "power_w": round(float(parts[6] or 0), 1),
            })
        return rows
    except Exception as exc:
        return [{"error": str(exc)}]


def df_shm() -> dict:
    try:
        r = subprocess.run(["df", "-BM", "/dev/shm"], capture_output=True, text=True, timeout=5)
        lines = r.stdout.strip().splitlines()
        if len(lines) < 2:
            return {}
        parts = lines[1].split()
        return {
            "shm_size_mb": parts[1].rstrip("M"),
            "shm_used_mb": parts[2].rstrip("M"),
            "shm_avail_mb": parts[3].rstrip("M"),
            "shm_use_pct": parts[4].rstrip("%"),
        }
    except Exception:
        return {}


def process_counts() -> dict:
    def count(pattern: str) -> int:
        try:
            r = subprocess.run(["pgrep", "-fc", pattern], capture_output=True, text=True, timeout=5)
            return int(r.stdout.strip() or 0)
        except Exception:
            return 0

    return {
        "moss_tts_server": count("moss-tts-server"),
        "teacher_workers": count("build_realtime_finetune_dataset"),
    }


def scrape_kernel(since_boot: bool = False) -> list[str]:
    alerts: list[str] = []
    try:
        cmd = ["dmesg", "--time-format=iso", "-T"] if since_boot else ["dmesg", "--time-format=iso", "-T", "-c"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        text = r.stdout + r.stderr
    except Exception:
        try:
            r = subprocess.run(
                ["journalctl", "-k", "-n", "80", "--no-pager", "-o", "short-iso"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            text = r.stdout
        except Exception as exc:
            return [f"kernel scrape failed: {exc}"]
    for line in text.splitlines():
        for pat in KERNEL_PATTERNS:
            if re.search(pat, line, re.I):
                alerts.append(line.strip())
                break
    return alerts


def wav_counts(staging: Path, disk: Path) -> dict:
    def n(p: Path) -> int:
        if not p.is_dir():
            return 0
        return sum(1 for _ in p.glob("*.wav"))

    return {
        "staging_shm": n(staging / "v15"),
        "disk": n(disk / "v15"),
    }


def check_alerts(sample: dict, *, min_avail_gb: float, max_swap_pct: float) -> list[str]:
    alerts: list[str] = []
    mem = sample.get("mem", {})
    avail_kb = mem.get("MemAvailable", 0)
    if avail_kb and avail_kb < min_avail_gb * 1024 * 1024:
        alerts.append(f"LOW_MEM: MemAvailable={avail_kb // 1024 // 1024}GiB < {min_avail_gb}GiB")
    swap_total = mem.get("SwapTotal", 0)
    swap_free = mem.get("SwapFree", 0)
    if swap_total > 0:
        swap_used_pct = 100.0 * (1.0 - swap_free / swap_total)
        if swap_used_pct > max_swap_pct:
            alerts.append(f"HIGH_SWAP: {swap_used_pct:.0f}% used")
    for g in sample.get("gpu", []):
        if "mem_free_mb" in g and g["mem_free_mb"] < 500:
            alerts.append(f"GPU{g['index']}_VRAM_LOW: {g['mem_free_mb']}MiB free")
        if "temp_c" in g and g["temp_c"] > 85:
            alerts.append(f"GPU{g['index']}_HOT: {g['temp_c']}C")
    return alerts


def main() -> int:
    p = argparse.ArgumentParser(description="Host health watchdog for teacher gen")
    p.add_argument("--interval", type=float, default=15.0)
    p.add_argument("--log-dir", type=Path, default=ROOT / "training/loli_15s/logs/health")
    p.add_argument("--staging", type=Path, default=Path("/dev/shm/loli15s_wavs"))
    p.add_argument("--wav-disk", type=Path, default=ROOT / "training/loli_15s/wavs")
    p.add_argument("--min-avail-gb", type=float, default=float(os.environ.get("HEALTH_MIN_AVAIL_GB", "8")))
    p.add_argument("--max-swap-pct", type=float, default=float(os.environ.get("HEALTH_MAX_SWAP_PCT", "40")))
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    health_path = args.log_dir / "health.jsonl"
    kernel_path = args.log_dir / "kernel_alerts.log"
    alert_path = args.log_dir / "alerts.log"

    print(f"watchdog → {health_path} (interval={args.interval}s)", flush=True)

    while True:
        ts = datetime.now(timezone.utc).isoformat()
        mem = read_meminfo()
        kernel_hits = scrape_kernel(since_boot=False)
        sample = {
            "ts": ts,
            "mem": mem,
            "mem_available_gib": round(mem.get("MemAvailable", 0) / 1024 / 1024, 2),
            "psi": read_psi(),
            "loadavg": loadavg(),
            "gpu": gpu_stats(),
            "shm": df_shm(),
            "wavs": wav_counts(args.staging, args.wav_disk),
            "processes": process_counts(),
        }
        user_alerts = check_alerts(
            sample, min_avail_gb=args.min_avail_gb, max_swap_pct=args.max_swap_pct
        )
        if kernel_hits:
            with kernel_path.open("a", encoding="utf-8") as kf:
                for line in kernel_hits:
                    kf.write(f"{ts} {line}\n")
            user_alerts.extend([f"KERNEL: {l[:200]}" for l in kernel_hits[:5]])

        sample["alerts"] = user_alerts
        with health_path.open("a", encoding="utf-8") as hf:
            hf.write(json.dumps(sample, ensure_ascii=False) + "\n")

        if user_alerts:
            msg = f"{ts} " + " | ".join(user_alerts)
            print(f"ALERT {msg}", flush=True)
            with alert_path.open("a", encoding="utf-8") as af:
                af.write(msg + "\n")

        if args.once:
            break
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
