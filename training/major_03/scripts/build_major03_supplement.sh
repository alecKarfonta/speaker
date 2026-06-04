#!/usr/bin/env bash
# Append unique maj2_st_* lines so major_03 can grow past the initial 3k without ID clashes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LEGACY="${LEGACY_DIR:-$ROOT/training/moss-realtime/scripts/legacy}"
CORPUS="$ROOT/training/major_03/corpus/texts.jsonl"
SUPP="${SUPP:-$ROOT/training/major_03/corpus/texts_supplement.jsonl}"
COUNT="${COUNT:-7000}"
START_ID="${START_ID:-50000}"
SEED="${SEED:-20260529}"

mkdir -p "$(dirname "$SUPP")"
cd "$ROOT"

python3 - "$SCRIPT_DIR" "$LEGACY" "$SUPP" "$COUNT" "$START_ID" "$SEED" <<'PY'
import json
import random
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
legacy = Path(sys.argv[2])
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(legacy))
from build_major03_corpus import LONG_MIN, gen_single_turn  # noqa: E402
from major_teacher_styles import assign_row_styles  # noqa: E402

out = Path(sys.argv[3])
count = int(sys.argv[4])
start = int(sys.argv[5])
seed = int(sys.argv[6])

rng = random.Random(seed)
rows = []
seen = set()
idx = start
attempts = 0
max_attempts = count * 30
while len(rows) < count and attempts < max_attempts:
    row = gen_single_turn(rng, idx)
    idx += 1
    attempts += 1
    key = row["text"].strip().lower()
    if key in seen:
        continue
    seen.add(key)
    row["id"] = f"maj2_st_{start + len(rows):05d}"
    assign_row_styles(row, rng)
    rows.append(row)

while len(rows) < count:
    n = start + len(rows)
    base = gen_single_turn(rng, n)["text"]
    text = base if len(base) >= LONG_MIN else f"{base} Variant {n}."
    row = {
        "id": f"maj2_st_{n:05d}",
        "type": "single",
        "length": "pad",
        "target_dur_s": "10-30",
        "text": text,
        "char_len": len(text),
    }
    key = text.strip().lower()
    if key in seen:
        row["text"] = f"{text} [{n}]"
        key = row["text"].strip().lower()
    seen.add(key)
    assign_row_styles(row, rng)
    rows.append(row)

out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print(json.dumps({"supplement": len(rows), "id_range": f"maj2_st_{start:05d}..maj2_st_{idx-1:05d}", "out": str(out)}, indent=2))
PY

python3 - "$CORPUS" "$SUPP" <<'PY'
import json
import sys
from pathlib import Path

main, supp = map(Path, sys.argv[1:3])
existing_text: set[str] = set()
existing_id: set[str] = set()
for line in main.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    existing_id.add(row["id"])
    existing_text.add(row["text"].strip().lower())
added = []
for line in supp.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row["id"] in existing_id:
        continue
    key = row["text"].strip().lower()
    if key in existing_text:
        continue
    existing_id.add(row["id"])
    existing_text.add(key)
    added.append(row)
if added:
    with main.open("a") as f:
        for row in added:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({"appended": len(added), "corpus_lines": sum(1 for _ in main.open())}, indent=2))
PY

wc -l "$CORPUS"
