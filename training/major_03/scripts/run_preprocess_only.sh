#!/usr/bin/env bash
set -euo pipefail
ROOT="${SPEAKER_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
MAJOR="$ROOT/training/major_03"
FINETUNE="$ROOT/training/moss-realtime/scripts/legacy/finetune"
export MOSS_RT_TRAIN_DIR="$MAJOR"
export SPEAKER_ROOT="$ROOT"
export NUM_GPUS="${NUM_GPUS:-4}"
export TRAIN_RAW="${TRAIN_RAW:-$MAJOR/train_raw.noref.jsonl}"
export PREPARED="${PREPARED:-$MAJOR/prepared/train_with_codes.jsonl}"
LOG="${LOG:-$MAJOR/logs/preprocess.log}"
mkdir -p "$MAJOR/prepared" "$MAJOR/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== major_03 preprocess $(date -Is) rows=$(wc -l < "$TRAIN_RAW") gpus=$NUM_GPUS ==="
"$FINETUNE/run_moss_rt_finetune_preprocess.sh"
echo "=== Done $(date -Is) ==="
ls -la "$MAJOR/prepared/"
