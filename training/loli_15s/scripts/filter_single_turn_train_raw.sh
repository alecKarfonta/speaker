#!/usr/bin/env bash
# Keep single-turn assistant clips only (drop mt_* multi-turn and multi-conversation rows).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOLI="${MOSS_RT_TRAIN_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
IN="${IN:-$LOLI/train_raw.jsonl}"
OUT="${OUT:-$LOLI/train_raw.jsonl}"
BACKUP="${BACKUP:-$LOLI/train_raw.with_multiturn.jsonl}"

python3 - "$IN" "$OUT" "$BACKUP" <<'PY'
import json
import shutil
import sys
from pathlib import Path

inp, out, backup = map(Path, sys.argv[1:4])
rows_in = []
for line in inp.read_text().splitlines():
    if line.strip():
        rows_in.append(json.loads(line))

if inp.resolve() != backup.resolve():
    if not backup.is_file():
        shutil.copy2(inp, backup)
        print(f"backup → {backup} ({len(rows_in)} rows)")

kept = []
dropped_mt = dropped_multi = 0
for row in rows_in:
    rid = row.get("id", "")
    conv = row.get("conversations") or []
    if rid.startswith("mt_"):
        dropped_mt += 1
        continue
    if len(conv) != 1:
        dropped_multi += 1
        continue
    if conv[0].get("role") != "assistant":
        dropped_multi += 1
        continue
    kept.append(row)

out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n")
stats = {
    "input_rows": len(rows_in),
    "kept": len(kept),
    "dropped_mt": dropped_mt,
    "dropped_multi": dropped_multi,
    "out": str(out),
}
(Path(out).parent / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
print(json.dumps(stats, indent=2))
PY

python3 "$LOLI/scripts/legacy/finetune/build_moss_rt_train_raw_noref.py" \
  --input "$OUT" \
  --output "$LOLI/train_raw.noref.jsonl"
