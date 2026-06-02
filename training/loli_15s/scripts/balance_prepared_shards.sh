#!/usr/bin/env bash
# Equalize prepared rank*.jsonl line counts (DDP hangs if shards differ by even 1 row).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PREPARED_DIR="${PREPARED_DIR:-$ROOT/training/loli_15s/prepared}"
GLOB="${GLOB:-train_with_codes_single.rank*.jsonl}"

python3 - "$PREPARED_DIR" "$GLOB" <<'PY'
import sys
from pathlib import Path

prepared = Path(sys.argv[1])
glob_pat = sys.argv[2]
shards = sorted(prepared.glob(glob_pat))
if not shards:
    raise SystemExit(f"No shards matching {glob_pat} in {prepared}")

def load(p: Path) -> list[str]:
    return [line for line in p.read_text().splitlines() if line.strip()]

def save(p: Path, lines: list[str]) -> None:
    p.write_text("\n".join(lines) + ("\n" if lines else ""))

counts = {p: len(load(p)) for p in shards}
print("Before:", {p.name: n for p, n in counts.items()})

target = max(counts.values())
while min(counts.values()) < target:
    short_p = min(counts, key=counts.get)
    rich_candidates = [p for p in shards if counts[p] > counts[short_p]]
    if not rich_candidates:
        break
    rich_p = max(rich_candidates, key=lambda p: counts[p])
    rich_lines = load(rich_p)
    short_lines = load(short_p)
    short_lines.append(rich_lines.pop())
    save(rich_p, rich_lines)
    save(short_p, short_lines)
    counts[rich_p] -= 1
    counts[short_p] += 1
    print(f"  moved 1 row {rich_p.name} -> {short_p.name}")

print("After:", {p.name: counts[p] for p in shards})
if len(set(counts.values())) != 1:
    raise SystemExit("Could not balance shards")
PY
