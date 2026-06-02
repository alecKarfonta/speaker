#!/usr/bin/env bash
# Restore merged train_raw, re-assemble pruned WAVs, STT tail-trim only + quarantine WER>threshold.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
STT_API="${STT_API:-http://localhost:8603/v1/audio/transcriptions}"
WER_Q="${WER_QUARANTINE_THRESHOLD:-0.75}"
LOG="${LOG:-$LOLI/logs/qc_trim_high_wer.log}"
QC_DIR="${QC_DIR:-$LOLI/qc_trim}"

exec > >(tee "$LOG") 2>&1
echo "=== qc_trim_high_wer $(date -Is) WER>${WER_Q} ==="

STT_BASE="${STT_API%/v1/audio/transcriptions}"
curl -sf --max-time 5 "${STT_BASE}/v1/models" >/dev/null || {
  echo "ERROR: STT not up at ${STT_BASE}/v1/models" >&2
  exit 1
}

echo "=== Restore train_raw from pre-gap3 merge (undo aggressive QC filter) ==="
cp -a "$LOLI/train_raw.pre_gap3_merge.jsonl" "$LOLI/train_raw.jsonl"

echo "=== Re-assemble v15_pruned from v15 + quarantine + premerge ==="
"$SCRIPT_DIR/assemble_loli_dataset.sh"

echo "=== Single-turn filter ==="
"$SCRIPT_DIR/filter_single_turn_train_raw.sh"

echo "=== QC: trim-only + quarantine only if WER > ${WER_Q} ==="
mkdir -p "$QC_DIR"
python3 "$ROOT/training/moss-realtime/scripts/legacy/prune_loli15s_teacher_dataset.py" \
  --wav-dir "$LOLI/wavs/v15_pruned" \
  --corpus "$LOLI/corpus/texts.jsonl" \
  --qc-dir "$QC_DIR" \
  --train-ids "$LOLI/corpus/.no_train_ids" \
  --workers "${QC_WORKERS:-8}" \
  --stt-api "$STT_API" \
  --trim-only \
  --wer-quarantine-threshold "$WER_Q" \
  --quarantine-dir "$LOLI/wavs/v15_quarantine_high_wer" \
  --apply \
  --in-place

echo "=== Drop high-WER quarantines from train_raw ==="
python3 - "$LOLI" "$QC_DIR" <<'PY'
import json
from pathlib import Path

loli = Path(__import__("sys").argv[1])
qc = Path(__import__("sys").argv[2])
qfile = qc / "quarantine_ids.txt"
train = loli / "train_raw.jsonl"
bad = {line.strip() for line in qfile.read_text().splitlines() if line.strip()} if qfile.is_file() else set()
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
print(json.dumps({"wer_quarantine": len(bad), "dropped": dropped, "kept": len(rows)}, indent=2))
PY

"$SCRIPT_DIR/filter_single_turn_train_raw.sh"
echo "Done. Report: $QC_DIR/prune_report.html"
