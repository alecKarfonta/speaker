#!/usr/bin/env bash
# Resume after crash: verify single-turn data, preprocess if needed, run 4-GPU SFT.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
LOG="$LOLI/logs/recover_sft_single.log"

export NUM_GPUS="${NUM_GPUS:-4}"
export MOSS_RT_TRAIN_DIR="$LOLI"
export SPEAKER_ROOT="$ROOT"
export OUTPUT_DIR="$LOLI/output/sft_ddp_single"
export PREPARED_OUT="$LOLI/prepared/train_with_codes_single.jsonl"
export PREPARED_GLOB="$LOLI/prepared/train_with_codes_single.rank*.jsonl"
export REFRESH_PREPARED=0

echo "=== Recover $(date -Is) ===" | tee -a "$LOG"
exec >>"$LOG" 2>&1

# Ensure single-turn jsonl (idempotent)
"$SCRIPT_DIR/filter_single_turn_train_raw.sh"

if ! compgen -G "$PREPARED_GLOB" >/dev/null; then
  echo "Prepared shards missing — running preprocess..."
  REFRESH_PREPARED=0 RUN_SFT=0 "$SCRIPT_DIR/run_sft_4gpu.sh"
else
  echo "Prepared shards present:"
  wc -l "$LOLI"/prepared/train_with_codes_single.rank*.jsonl
fi

echo "=== Starting SFT ==="
RUN_PREPROCESS=0 "$SCRIPT_DIR/run_sft_4gpu.sh"
