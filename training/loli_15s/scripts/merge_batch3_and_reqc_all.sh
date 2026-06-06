#!/usr/bin/env bash
# Merge batch3 into loli_15s, then re-run v2 QC on the full combined raw corpus.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
PRUNE="$ROOT/training/moss-realtime/scripts/legacy/prune_loli15s_teacher_dataset.py"
STT_API="${STT_API:-http://192.168.1.196:8603/v1/audio/transcriptions}"
QC_WORKERS="${QC_WORKERS:-8}"
LOG="$LOLI/logs/merge_reqc_v2_$(date +%Y%m%d_%H%M%S).log"
PRE_POOL="$LOLI/wavs/v15_pruned.pre_reqc"

mkdir -p "$LOLI/logs"

FINETUNE_VENV="${FINETUNE_VENV:-$ROOT/.venv-finetune}"
PYTHON="${PYTHON:-$FINETUNE_VENV/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== merge + re-QC v2 @ $(date -Is) ==="
echo "Log: $LOG"

if [[ "${SKIP_MERGE:-0}" != "1" ]]; then
  echo "=== 1) Merge batch3 → loli_15s ==="
  "$SCRIPT_DIR/merge_batch3_into_loli15s.sh"
else
  echo "=== 1) Merge (skip, SKIP_MERGE=1) ==="
fi

n_raw=$(find "$LOLI/wavs/v15" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ')
echo "Raw v15 WAVs: $n_raw"

echo "=== 2) Preserve prior pruned set as ECAPA teacher pool ==="
if [[ -d "$LOLI/wavs/v15_pruned" ]] && [[ ! -e "$PRE_POOL" ]]; then
  mv "$LOLI/wavs/v15_pruned" "$PRE_POOL"
  echo "Backed up → $PRE_POOL"
elif [[ -d "$PRE_POOL" ]]; then
  echo "Using existing pool: $PRE_POOL"
else
  echo "WARN: no prior v15_pruned pool — voice QC teacher match may be weak" >&2
fi

rm -rf "$LOLI/wavs/v15_pruned" "$LOLI/wavs/v15_quarantine"
mkdir -p "$LOLI/wavs/v15_pruned" "$LOLI/wavs/v15_quarantine" "$LOLI/qc"

echo "=== 3) STT preflight ($STT_API) ==="
export STT_API
"$PYTHON" - "$STT_API" "$LOLI/wavs/v15" <<'PY'
import sys
from pathlib import Path
import requests

api, wav_dir = sys.argv[1], Path(sys.argv[2])
sample = next(wav_dir.glob("*.wav"), None)
if sample is None:
    raise SystemExit("No WAVs in v15")
with sample.open("rb") as f:
    r = requests.post(
        api,
        files={"file": (sample.name, f, "audio/wav")},
        data={"model": "base", "language": "en", "response_format": "verbose_json"},
        timeout=120,
    )
if r.status_code != 200:
    raise SystemExit(f"STT HTTP {r.status_code}: {r.text[:200]}")
print(f"STT OK: {sample.name}")
PY

POOL_ARG=()
if [[ -d "$PRE_POOL" ]]; then
  POOL_ARG=(--teacher-pool "$PRE_POOL")
fi

echo "=== 4) Full v2 QC on combined raw v15 ($n_raw clips, workers=$QC_WORKERS) ==="
"$PYTHON" "$PRUNE" \
  --root "$ROOT" \
  --wav-dir "$LOLI/wavs/v15" \
  --out-dir "$LOLI/wavs/v15_pruned" \
  --quarantine-dir "$LOLI/wavs/v15_quarantine" \
  --corpus "$LOLI/corpus/texts.jsonl" \
  --qc-dir "$LOLI/qc" \
  --apply \
  --trim-only \
  --wer-quarantine-threshold 0.75 \
  --end-buffer-ms 750 \
  --quarantine-cutoff \
  --tail-gap-fail-s 0.30 \
  --min-missing-tail-words 2 \
  --stt-api "$STT_API" \
  --voice-qc \
  --ref-wav "$ROOT/data/voices/loli/loli_15s.wav" \
  --teacher-train-raw "$LOLI/train_raw.jsonl" \
  "${POOL_ARG[@]}" \
  --min-cos-ref 0.5 \
  --min-cos-teacher 0.5 \
  --workers "$QC_WORKERS" \
  --stt-max-retries 5 \
  --defer-stt-fail \
  --no-local-whisper-fallback

STT_PENDING="$LOLI/qc/stt_pending_wavs.txt"
if [[ -s "$STT_PENDING" ]]; then
  echo "=== 4b) STT pending mop-up ($(wc -l < "$STT_PENDING") clips) ==="
  "$SCRIPT_DIR/recover_loli_qc.sh"
fi

echo "=== 5) Drop quarantined clips from train_raw ==="
"$PYTHON" - "$LOLI" <<'PY'
import json
import sys
from pathlib import Path

loli = Path(sys.argv[1])
qfile = loli / "qc/quarantine_ids.txt"
train = loli / "train_raw.jsonl"
backup = loli / "train_raw.pre_reqc.jsonl"

if train.is_file() and not backup.is_file():
    backup.write_bytes(train.read_bytes())

bad = set()
if qfile.is_file():
    bad = {line.strip() for line in qfile.read_text().splitlines() if line.strip()}

rows = []
dropped = 0
for line in train.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    names = [
        Path(t["wav"]).name
        for t in row.get("conversations") or []
        if t.get("role") == "assistant" and t.get("wav")
    ]
    if bad and any(n in bad for n in names):
        dropped += 1
        continue
    for t in row.get("conversations") or []:
        if t.get("role") == "assistant" and t.get("wav"):
            t["wav"] = f"training/loli_15s/wavs/v15_pruned/{Path(t['wav']).name}"
    row["ref_wav"] = "data/voices/loli/loli_15s.wav"
    rows.append(row)

train.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print(json.dumps({
    "quarantined_wavs": len(bad),
    "dropped_rows": dropped,
    "kept_rows": len(rows),
    "pruned_on_disk": len(list((loli / "wavs/v15_pruned").glob("*.wav"))),
}, indent=2))
PY

if [[ -x "$SCRIPT_DIR/filter_single_turn_train_raw.sh" ]]; then
  echo "=== 6) Rebuild noref jsonl ==="
  "$SCRIPT_DIR/filter_single_turn_train_raw.sh"
fi

n_pruned=$(find "$LOLI/wavs/v15_pruned" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ')
n_quar=$(find "$LOLI/wavs/v15_quarantine" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ')
echo "=== Done $(date -Is) ==="
echo "Raw: $n_raw  Pruned: $n_pruned  Quarantine: $n_quar"
echo "Report: $LOLI/qc/prune_report.html"
echo "Next: run_sft_v2.sh or preprocess"
