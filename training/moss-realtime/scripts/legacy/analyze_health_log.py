#!/usr/bin/env python3
"""Summarize watchdog health.jsonl after a crash or long run."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("health_jsonl", type=Path)
    args = p.parse_args()
    if not args.health_jsonl.is_file():
        print(f"Missing {args.health_jsonl}", file=sys.stderr)
        return 1

    rows = []
    alerts = []
    for line in args.health_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append(r)
        alerts.extend(r.get("alerts") or [])

    if not rows:
        print("No samples")
        return 0

    avail = [r.get("mem_available_gib", 0) for r in rows if r.get("mem_available_gib")]
    print(f"Samples: {len(rows)}")
    print(f"Time: {rows[0].get('ts')} → {rows[-1].get('ts')}")
    if avail:
        print(f"MemAvailable GiB: min={min(avail):.1f} median={statistics.median(avail):.1f} last={avail[-1]:.1f}")
    last = rows[-1]
    print(f"Last loadavg: {last.get('loadavg')}")
    print(f"Last wavs: {last.get('wavs')}")
    print(f"Last processes: {last.get('processes')}")
    if alerts:
        print(f"\nAlerts ({len(alerts)}):")
        for a in alerts[-20:]:
            print(f"  {a}")
    else:
        print("\nNo threshold alerts recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
