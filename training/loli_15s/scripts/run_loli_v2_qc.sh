#!/usr/bin/env bash
# v2 QC: raw v15 → v15_pruned (750ms buffer, cutoff quarantine, voice floors).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
PRUNE="$ROOT/training/moss-realtime/scripts/legacy/prune_loli15s_teacher_dataset.py"
STT_API="${STT_API:-http://192.168.1.196:8603/v1/audio/transcriptions}"
WAV_DIR="${WAV_DIR:-$LOLI/wavs/v15}"
OUT_DIR="${OUT_DIR:-$LOLI/wavs/v15_pruned}"
END_BUFFER_MS="${END_BUFFER_MS:-750}"
MIN_COS_REF="${MIN_COS_REF:-0.5}"
MIN_COS_TEACHER="${MIN_COS_TEACHER:-0.5}"
QC_WORKERS="${QC_WORKERS:-8}"
TEACHER_POOL="${TEACHER_POOL:-$LOLI/wavs/v15_pruned.pre_reqc}"

mkdir -p "$OUT_DIR" "$LOLI/wavs/v15_quarantine" "$LOLI/qc"
export STT_API

python3 - "$STT_API" "$WAV_DIR" <<'PY'
import sys
from pathlib import Path
import requests
api, d = sys.argv[1], Path(sys.argv[2])
s = next(d.glob("*.wav"))
with s.open("rb") as f:
    r = requests.post(api, files={"file": (s.name, f, "audio/wav")},
        data={"model": "base", "language": "en", "response_format": "verbose_json"}, timeout=120)
if r.status_code != 200:
    raise SystemExit(f"STT failed: {r.status_code}")
print(f"STT OK: {s.name}")
PY

pool_args=()
if [[ -d "$TEACHER_POOL" ]]; then
  pool_args=(--teacher-pool "$TEACHER_POOL")
fi

python3 "$PRUNE" \
  --root "$ROOT" \
  --wav-dir "$WAV_DIR" \
  --out-dir "$OUT_DIR" \
  --quarantine-dir "$LOLI/wavs/v15_quarantine" \
  --corpus "$LOLI/corpus/texts.jsonl" \
  --qc-dir "$LOLI/qc" \
  --apply \
  --trim-only \
  --wer-quarantine-threshold 0.75 \
  --end-buffer-ms "$END_BUFFER_MS" \
  --quarantine-cutoff \
  --tail-gap-fail-s 0.30 \
  --min-missing-tail-words 2 \
  --stt-api "$STT_API" \
  --voice-qc \
  --ref-wav "$ROOT/data/voices/loli/loli_15s.wav" \
  --teacher-train-raw "$LOLI/train_raw.jsonl" \
  "${pool_args[@]}" \
  --min-cos-ref "$MIN_COS_REF" \
  --min-cos-teacher "$MIN_COS_TEACHER" \
  --workers "$QC_WORKERS"

echo "v2 QC done → $OUT_DIR (buffer=${END_BUFFER_MS}ms, cutoff quarantine on)"
