#!/usr/bin/env bash
# Build equal 2-way prepared shards for DDP (4-way shards modulo-mapped to 2 GPUs are off by 1 row).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PREPARED="${PREPARED_DIR:-$ROOT/training/loli_15s/prepared}"
SRC_GLOB="${SRC_GLOB:-train_with_codes_single.rank*-of-00004.jsonl}"
OUT_PREFIX="${OUT_PREFIX:-train_with_codes_2gpu}"

python3 - "$PREPARED" "$SRC_GLOB" "$OUT_PREFIX" <<'PY'
import sys
from pathlib import Path

prepared = Path(sys.argv[1])
src_glob = sys.argv[2]
out_prefix = sys.argv[3]
sources = sorted(prepared.glob(src_glob))
if not sources:
    raise SystemExit(f"No sources matching {src_glob} in {prepared}")

lines: list[str] = []
for p in sources:
    for line in p.read_text().splitlines():
        if line.strip():
            lines.append(line)

n = len(lines)
if n % 2 != 0:
    dropped = lines.pop()
    print(f"Dropped 1 row for even 2-GPU split (was {n} rows)")
    n -= 1

half = n // 2
shards = [
    (f"{out_prefix}.rank00000-of-00002.jsonl", lines[:half]),
    (f"{out_prefix}.rank00001-of-00002.jsonl", lines[half:]),
]
for name, chunk in shards:
    out = prepared / name
    out.write_text("\n".join(chunk) + ("\n" if chunk else ""))
    print(f"  {name}: {len(chunk)} rows")

if len(shards[0][1]) != len(shards[1][1]):
    raise SystemExit("Shard sizes still unequal")
print(f"OK: {n} rows -> {half}+{half}")
PY
