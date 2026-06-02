#!/usr/bin/env bash
# Assemble merged loli WAVs, filter train_raw, run full QC (STT trim + WER + wav outliers).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
LOG="$LOLI/logs/merge_and_qc_all.log"
STT_API="${STT_API:-http://localhost:8603/v1/audio/transcriptions}"

mkdir -p "$LOLI/logs" "$LOLI/qc"
exec > >(tee -a "$LOG") 2>&1
echo "=== merge_and_qc_all $(date -Is) ==="

# Tear down leftover MOSS teacher servers if any
if [[ -x "$ROOT/training/moss-realtime/scripts/legacy/teardown_openmoss.sh" ]]; then
  PORTS=8014,8015,8016,8017 SPEAKER_ROOT="$ROOT" \
    "$ROOT/training/moss-realtime/scripts/legacy/teardown_openmoss.sh" 2>/dev/null || true
fi

echo "=== 1) Assemble v15_pruned + align train_raw ==="
"$SCRIPT_DIR/assemble_loli_dataset.sh"

echo "=== 2) Single-turn filter + noref jsonl ==="
"$SCRIPT_DIR/filter_single_turn_train_raw.sh"

PRUNE="$ROOT/training/moss-realtime/scripts/legacy/prune_loli15s_teacher_dataset.py"
QC_ARGS=(
  --wav-dir "$LOLI/wavs/v15_pruned"
  --corpus "$LOLI/corpus/texts.jsonl"
  --qc-dir "$LOLI/qc"
  --train-ids "$LOLI/corpus/.no_train_ids"
  --workers "${QC_WORKERS:-8}"
  --apply
  --in-place
)

echo "=== 3) Full QC on v15_pruned (STT API) ==="
STT_BASE="${STT_API%/v1/audio/transcriptions}"
if ! curl -sf --max-time 5 "${STT_BASE}/v1/models" >/dev/null 2>&1; then
  echo "ERROR: STT not reachable at ${STT_BASE}/v1/models" >&2
  exit 1
fi
echo "STT API: $STT_API"
python3 "$PRUNE" --stt-api "$STT_API" "${QC_ARGS[@]}"

echo "=== 5) Drop quarantined IDs from train_raw ==="
python3 - "$LOLI" <<'PY'
import json
from pathlib import Path

loli = Path(__import__("sys").argv[1])
qfile = loli / "qc/quarantine_ids.txt"
train = loli / "train_raw.jsonl"
if not qfile.is_file():
    print("No quarantine_ids.txt — skip filter")
    raise SystemExit(0)
bad = {line.strip() for line in qfile.read_text().splitlines() if line.strip()}
rows = []
dropped = 0
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
print(json.dumps({"quarantined": len(bad), "dropped_from_train_raw": dropped, "kept": len(rows)}, indent=2))
PY

echo "=== 6) Rebuild noref + stats ==="
"$SCRIPT_DIR/filter_single_turn_train_raw.sh"

echo "=== Done $(date -Is) ==="
echo "Report: $LOLI/qc/prune_report.html"
echo "Quarantine list: $LOLI/qc/quarantine_ids.txt"
