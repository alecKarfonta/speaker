#!/usr/bin/env bash
# Append unique b2_st_* lines so batch2 + original loli can exceed 6k after QC losses.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LEGACY="${LEGACY_DIR:-$ROOT/training/moss-realtime/scripts/legacy}"
CORPUS="$ROOT/training/loli_15s_batch2/corpus/texts.jsonl"
SUPP="${SUPP:-$ROOT/training/loli_15s_batch2/corpus/texts_supplement.jsonl}"
COUNT="${COUNT:-200}"
START_ID="${START_ID:-50000}"
SEED="${SEED:-20260601}"

mkdir -p "$(dirname "$SUPP")"
cd "$ROOT"

python3 - "$LEGACY" "$SUPP" "$COUNT" "$START_ID" "$SEED" <<'PY'
import json
import random
import sys
from pathlib import Path

legacy = Path(sys.argv[1])
sys.path.insert(0, str(legacy))
from build_loli15s_corpus import gen_single_turn  # noqa: E402
from v15_teacher_styles import assign_row_styles  # noqa: E402

out = Path(sys.argv[2])
count = int(sys.argv[3])
start = int(sys.argv[4])
seed = int(sys.argv[5])

rng = random.Random(seed)
rows = []
seen = set()
idx = start
while len(rows) < count:
    row = gen_single_turn(rng, idx)
    row["id"] = f"b2_st_{idx:05d}"
    idx += 1
    key = row["text"].strip().lower()
    if key in seen:
        continue
    seen.add(key)
    assign_row_styles(row, rng)
    rows.append(row)

out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print(json.dumps({"supplement": len(rows), "id_range": f"b2_st_{start:05d}..b2_st_{idx-1:05d}", "out": str(out)}, indent=2))
PY

# Append only ids not already in main corpus
python3 - "$CORPUS" "$SUPP" <<'PY'
import json
import sys
from pathlib import Path

main, supp = map(Path, sys.argv[1:3])
existing = set()
for line in main.read_text().splitlines():
    if line.strip():
        existing.add(json.loads(line)["id"])
added = []
for line in supp.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row["id"] in existing:
        continue
    existing.add(row["id"])
    added.append(row)
if added:
    with main.open("a") as f:
        for row in added:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"appended": len(added), "corpus_lines": sum(1 for _ in main.open())}, indent=2))
PY
