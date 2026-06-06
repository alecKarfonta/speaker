#!/usr/bin/env bash
# Generate batch3 teacher WAVs (4× GPU) then build train_raw in batch3 dir.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
B3="$ROOT/training/loli_15s_batch3"
LOLI_LEGACY="$ROOT/training/loli_15s/scripts/legacy"

export MOSS_RT_TRAIN_DIR="$B3"
export LOG_DIR="$B3"
export CORPUS="$B3/corpus/texts.jsonl"
export OUT_DIR="$B3"
export WAV_REL_ROOT="training/loli_15s_batch3/wavs"
export REF_WAV="$ROOT/data/voices/loli/loli_15s.wav"
export SPEAKER_ROOT="$ROOT"
export VOICE_QC="${VOICE_QC:-1}"
export MIN_COS_REF="${MIN_COS_REF:-0.5}"
export MIN_COS_TEACHER="${MIN_COS_TEACHER:-0.5}"
export TEACHER_TRAIN_RAW="${TEACHER_TRAIN_RAW:-$ROOT/training/loli_15s/train_raw.jsonl}"
export TEACHER_POOL="${TEACHER_POOL:-$ROOT/training/loli_15s/wavs/v15_pruned}"

if [[ ! -f "$B3/corpus/texts.jsonl" ]]; then
  "$SCRIPT_DIR/build_loli_batch3.sh"
fi

mkdir -p "$B3/wavs/v15" "$B3/logs"

echo "=== Teacher gen (batch3 → $B3) ==="
"$LOLI_LEGACY/run_loli15s_teacher_gen_parallel.sh"

echo "train_raw: $B3/train_raw.jsonl ($(wc -l < "$B3/train_raw.jsonl" 2>/dev/null || echo 0) rows)"
echo "Next: $SCRIPT_DIR/merge_batch3_into_loli15s.sh && run_loli_v2_qc.sh"
