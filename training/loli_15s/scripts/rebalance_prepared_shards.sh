#!/usr/bin/env bash
# Rebuild equal N-way prepared shards (DDP hangs if any rank has even 1 extra row).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PREPARED="${PREPARED_DIR:-$ROOT/training/loli_15s/prepared}"
SRC_GLOB="${SRC_GLOB:-train_with_codes_single.rank*-of-00004.jsonl}"
OUT_PREFIX="${OUT_PREFIX:-train_with_codes_single}"
NUM_SHARDS="${NUM_SHARDS:-4}"

python3 - "$PREPARED" "$SRC_GLOB" "$OUT_PREFIX" "$NUM_SHARDS" <<'PY'
import sys
from pathlib import Path

prepared = Path(sys.argv[1])
src_glob = sys.argv[2]
out_prefix = sys.argv[3]
num_shards = int(sys.argv[4])
sources = sorted(prepared.glob(src_glob))
if not sources:
    raise SystemExit(f"No sources matching {src_glob} in {prepared}")

lines: list[str] = []
for p in sources:
    for line in p.read_text().splitlines():
        if line.strip():
            lines.append(line)

n = len(lines)
rem = n % num_shards
if rem:
    lines = lines[:-rem]
    print(f"Dropped {rem} row(s) for {num_shards}-way split (was {n} rows)")

per = len(lines) // num_shards
counts = []
for i in range(num_shards):
    chunk = lines[i * per : (i + 1) * per]
    name = f"{out_prefix}.rank{i:05d}-of-{num_shards:05d}.jsonl"
    out = prepared / name
    out.write_text("\n".join(chunk) + ("\n" if chunk else ""))
    counts.append(len(chunk))
    print(f"  {name}: {len(chunk)} rows")

if len(set(counts)) != 1:
    raise SystemExit(f"Unequal shards: {counts}")
print(f"OK: {sum(counts)} rows across {num_shards} ranks")
PY
