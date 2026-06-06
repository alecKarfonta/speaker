#!/usr/bin/env bash
# Duplicate emotion / expressive-style rows in prepared shards before v2 SFT resume.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
PREPARED="${PREPARED_DIR:-$LOLI/prepared}"
WEIGHT="${EMOTION_OVERSAMPLE:-2}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SRC_GLOB="${SRC_GLOB:-train_with_codes_single.rank*-of-$(printf '%05d' "$NUM_SHARDS").jsonl}"
CORPUS="${CORPUS:-$LOLI/corpus/texts.jsonl}"

python3 - "$PREPARED" "$CORPUS" "$WEIGHT" "$SRC_GLOB" <<'PY'
import json
import sys
from pathlib import Path

prepared = Path(sys.argv[1])
corpus_path = Path(sys.argv[2])
weight = int(sys.argv[3])
src_glob = sys.argv[4]

emotion_ids: set[str] = set()
expressive_styles = {"excited", "playful", "storytelling", "curious"}
if corpus_path.is_file():
    for line in corpus_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("gap_category") == "emotion" or row.get("style") in expressive_styles:
            emotion_ids.add(row["id"])
            emotion_ids.add(f"{row['id']}_v15")

sources = sorted(prepared.glob(src_glob))
if not sources:
    raise SystemExit(f"No shards matching {src_glob}")

total_dup = 0
for path in sources:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    out_lines = []
    for line in lines:
        out_lines.append(line)
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = str(row.get("id", ""))
        base = rid.removesuffix("_v15")
        if base in emotion_ids or rid in emotion_ids:
            for _ in range(weight - 1):
                out_lines.append(line)
                total_dup += 1
    path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    print(f"  {path.name}: {len(lines)} -> {len(out_lines)} rows")

print(f"Oversampled {total_dup} duplicate lines (weight={weight})")
PY

NUM_SHARDS="$NUM_SHARDS" "$SCRIPT_DIR/rebalance_prepared_shards.sh"
echo "Rebalanced shards after emotion oversample."
