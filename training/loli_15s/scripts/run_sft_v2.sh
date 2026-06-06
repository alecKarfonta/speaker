#!/usr/bin/env bash
# Loli v2: filter train_raw → preprocess → emotion oversample → resume SFT from epoch-7 → merge v2.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"

export SPEAKER_ROOT="$ROOT"
export MOSS_RT_TRAIN_DIR="$LOLI"
export SFT_V2=1

CKPT="${RESUME_CHECKPOINT:-$LOLI/output/sft_ddp_single/checkpoint-epoch-7}"
NUM_EPOCHS="${NUM_EPOCHS:-6}"
LR="${LEARNING_RATE:-5e-6}"
EPOCH_OFFSET="${EPOCH_OFFSET:-7}"

echo "=== filter single-turn train_raw ==="
"$LOLI/scripts/filter_single_turn_train_raw.sh" || true
python3 "$LOLI/scripts/legacy/finetune/build_moss_rt_train_raw_noref.py" 2>/dev/null || true

echo "=== preprocess (4 GPU) ==="
PREPARED_OUT="$LOLI/prepared/train_with_codes_single.jsonl" \
  TRAIN_RAW="$LOLI/train_raw.noref.jsonl" \
  python3 "$LOLI/scripts/distill.py" train preprocess --gpus 4 --noref

echo "=== emotion oversample + rebalance ==="
EMOTION_OVERSAMPLE="${EMOTION_OVERSAMPLE:-2}" \
  "$SCRIPT_DIR/oversample_emotion_prepared.sh"

echo "=== resume SFT from epoch-7 ==="
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OUTPUT_DIR="$LOLI/output/sft_ddp_single"
export RESUME_CHECKPOINT="$CKPT"
export EPOCH_OFFSET="$EPOCH_OFFSET"
export NUM_EPOCHS="$NUM_EPOCHS"
export LEARNING_RATE="$LR"
export PREPARED_GLOB="$LOLI/prepared/train_with_codes_single.rank*.jsonl"

if [[ ! -f "$CKPT/adapter_model.safetensors" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${FINETUNE_VENV:-$ROOT/.venv-finetune}/bin/activate"
SFT_LOG="${SFT_LOG:-$LOLI/logs/sft_v2_resume.log}"
mkdir -p "$(dirname "$SFT_LOG")"
SPEAKER_ROOT="$ROOT" MOSS_RT_TRAIN_DIR="$LOLI" \
  "$LOLI/scripts/legacy/finetune/run_moss_rt_finetune_noref.sh" 2>&1 | tee -a "$SFT_LOG"

latest=$(ls -d "$OUTPUT_DIR"/checkpoint-epoch-* 2>/dev/null | sort -V | tail -1)
echo "=== merge → loli15s-v2-merged ==="
MERGE_CHECKPOINT="${latest:-$CKPT}" \
MERGE_OUTPUT="$LOLI/exports/loli15s-v2-merged" \
SFT_V2=1 \
  python3 "$LOLI/scripts/distill.py" export merge

echo "v2 merged: $LOLI/exports/loli15s-v2-merged"
