#!/usr/bin/env bash
# Build 3000 single-turn loli_15s corpus lines (batch2 IDs, no overlap with v15 st_*).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LEGACY="${LEGACY_DIR:-$ROOT/training/moss-realtime/scripts/legacy}"
CORPUS_DIR="${CORPUS_DIR:-$ROOT/training/loli_15s_batch2/corpus}"
OUT="${OUT:-$CORPUS_DIR/texts.jsonl}"
SEED="${SEED:-20260531}"
SINGLE="${SINGLE:-3000}"

mkdir -p "$CORPUS_DIR"
cd "$ROOT"

python3 "$LEGACY/build_loli15s_corpus.py" \
  --single "$SINGLE" \
  --multi 0 \
  --seed "$SEED" \
  --out "$OUT"

export SEED
python3 - "$OUT" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
seed = int(sys.argv[2])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
for row in rows:
    rid = row["id"]
    if not rid.startswith("b2_"):
        row["id"] = f"b2_{rid}"
path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
stats = {
    "total": len(rows),
    "single": sum(1 for r in rows if r["type"] == "single"),
    "multi": sum(1 for r in rows if r["type"] == "multi"),
    "out": str(path),
    "id_prefix": "b2_",
    "seed": seed,
}
(path.parent / "corpus_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
print(json.dumps(stats, indent=2))
PY

wc -l "$OUT"
