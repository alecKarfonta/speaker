#!/usr/bin/env bash
# QC prune (STT tail-trim) → align train_raw → no-ref jsonl → multi-GPU code preprocess.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MAJOR="$ROOT/training/major_03"
LEGACY="$ROOT/training/moss-realtime/scripts/legacy"
FINETUNE="$LEGACY/finetune"
PRUNE="$LEGACY/prune_loli15s_teacher_dataset.py"
export MOSS_RT_TRAIN_DIR="$MAJOR"
export SPEAKER_ROOT="$ROOT"
NUM_GPUS="${NUM_GPUS:-4}"
QC_WORKERS="${QC_WORKERS:-8}"
STT_API="${STT_API:-http://localhost:8603/v1/audio/transcriptions}"
# Major clips are 10–32s; default wav-outlier QC quarantined ~99.9% (duration/click false positives).
QC_TRIM_ONLY="${QC_TRIM_ONLY:-1}"
QC_WER_QUARANTINE="${QC_WER_QUARANTINE:-0.75}"
LOG="${LOG:-$MAJOR/logs/qc_and_preprocess.log}"

mkdir -p "$MAJOR/qc" "$MAJOR/logs" "$MAJOR/prepared" "$MAJOR/configs"
exec > >(tee -a "$LOG") 2>&1
echo "=== major_03 qc + preprocess $(date -Is) ==="

if [[ -x "$LEGACY/teardown_openmoss.sh" ]]; then
  PORTS=8014,8015,8016,8017 "$LEGACY/teardown_openmoss.sh" 2>/dev/null || true
fi

STT_BASE="${STT_API%/v1/audio/transcriptions}"
if ! curl -sf --max-time 10 "${STT_BASE}/v1/models" >/dev/null; then
  echo "ERROR: STT not reachable at ${STT_BASE}/v1/models" >&2
  exit 1
fi
echo "STT OK: $STT_API"

echo "=== 1) QC prune → wavs/v15_pruned (trim_only=${QC_TRIM_ONLY}) ==="
PRUNE_ARGS=(
  --root "$ROOT"
  --wav-dir "$MAJOR/wavs/v15"
  --corpus "$MAJOR/corpus/texts.jsonl"
  --qc-dir "$MAJOR/qc"
  --out-dir "$MAJOR/wavs/v15_pruned"
  --quarantine-dir "$MAJOR/wavs/v15_quarantine"
  --apply
  --end-buffer-ms 500
  --min-dur 9.0
  --max-dur 32.0
  --workers "$QC_WORKERS"
  --stt-api "$STT_API"
)
if [[ "$QC_TRIM_ONLY" == "1" ]]; then
  PRUNE_ARGS+=(--trim-only --wer-quarantine-threshold "$QC_WER_QUARANTINE")
fi
python3 "$PRUNE" "${PRUNE_ARGS[@]}"

echo "=== 2) Update train_raw (drop quarantine, point at v15_pruned) ==="
python3 "$SCRIPT_DIR/apply_qc_to_train_raw.py"

echo "=== 3) train_raw.noref.jsonl ==="
python3 "$FINETUNE/build_moss_rt_train_raw_noref.py" \
  --input "$MAJOR/train_raw.jsonl" \
  --output "$MAJOR/train_raw.noref.jsonl"

if [[ "${SKIP_PREPROCESS:-0}" != "1" ]]; then
  echo "=== 4) Preprocess audio codes ($NUM_GPUS GPUs) ==="
  if [[ ! -f "$MAJOR/configs/accelerate_ddp_4gpu.yaml" ]]; then
    cp "$ROOT/training/moss-realtime/configs/accelerate_ddp_4gpu.yaml" \
      "$MAJOR/configs/accelerate_ddp_4gpu.yaml"
  fi
  TRAIN_RAW="$MAJOR/train_raw.noref.jsonl" \
  PREPARED="$MAJOR/prepared/train_with_codes.jsonl" \
  NUM_GPUS="$NUM_GPUS" \
    "$FINETUNE/run_moss_rt_finetune_preprocess.sh"
fi

echo "=== Done $(date -Is) ==="
echo "QC report: $MAJOR/qc/prune_report.html"
echo "Prepared: $MAJOR/prepared/train_with_codes.jsonl"
