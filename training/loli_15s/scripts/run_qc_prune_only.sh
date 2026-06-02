#!/usr/bin/env bash
# Full QC on v15_pruned (requires STT on :8603). Run after merge/assemble.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
STT_API="${STT_API:-http://localhost:8603/v1/audio/transcriptions}"
LOG="${LOG:-$LOLI/logs/qc_prune.log}"

exec > >(tee -a "$LOG") 2>&1
echo "=== QC prune $(date -Is) STT=$STT_API ==="

STT_BASE="${STT_API%/v1/audio/transcriptions}"
curl -sf --max-time 5 "${STT_BASE}/v1/models" >/dev/null || {
  echo "ERROR: STT not reachable at ${STT_BASE}/v1/models" >&2
  exit 1
}

python3 "$ROOT/training/moss-realtime/scripts/legacy/prune_loli15s_teacher_dataset.py" \
  --wav-dir "$LOLI/wavs/v15_pruned" \
  --corpus "$LOLI/corpus/texts.jsonl" \
  --qc-dir "$LOLI/qc" \
  --train-ids "$LOLI/corpus/.no_train_ids" \
  --workers "${QC_WORKERS:-8}" \
  --stt-api "$STT_API" \
  --apply \
  --in-place

echo "=== Drop quarantined IDs from train_raw ==="
python3 - "$LOLI" <<'PY'
import json
from pathlib import Path

loli = Path(__import__("sys").argv[1])
qfile = loli / "qc/quarantine_ids.txt"
train = loli / "train_raw.jsonl"
if not qfile.is_file():
    print("No quarantine_ids.txt")
    raise SystemExit(0)
bad = {line.strip() for line in qfile.read_text().splitlines() if line.strip()}
rows, dropped = [], 0
for line in train.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    wav = Path((row.get("conversations") or [{}])[0].get("wav", "")).name
    if wav in bad:
        dropped += 1
        continue
    rows.append(row)
train.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print(json.dumps({"quarantined": len(bad), "dropped": dropped, "kept": len(rows)}, indent=2))
PY

"$SCRIPT_DIR/filter_single_turn_train_raw.sh"
echo "Done. Report: $LOLI/qc/prune_report.html"
