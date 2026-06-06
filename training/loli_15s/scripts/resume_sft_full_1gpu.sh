#!/usr/bin/env bash
# Resume 1-GPU full SFT after crash — one remaining epoch, then merge.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
LEGACY="$ROOT/training/moss-realtime/scripts/legacy"
FINETUNE="$LOLI/scripts/legacy/finetune"

export SPEAKER_ROOT="$ROOT"
export MOSS_RT_TRAIN_DIR="$LOLI"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export NUM_GPUS=1
export OUTPUT_DIR="${OUTPUT_DIR:-$LOLI/output/sft_1gpu_full}"
export PREPARED_GLOB="${PREPARED_GLOB:-$LOLI/prepared/train_with_codes_1gpu.rank00000-of-00001.jsonl}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$OUTPUT_DIR/checkpoint-epoch-6}"
export EPOCH_OFFSET="${EPOCH_OFFSET:-7}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
SFT_LOG="${SFT_LOG:-$LOLI/logs/sft_full_1gpu_resume_$(date +%Y%m%d_%H%M%S).log}"
export FINETUNE_LOG="$SFT_LOG"

if [[ ! -f "$RESUME_CHECKPOINT/adapter_model.safetensors" ]]; then
  echo "Missing resume checkpoint: $RESUME_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$PREPARED_GLOB" ]]; then
  echo "Missing prepared shard: $PREPARED_GLOB" >&2
  exit 1
fi

mkdir -p "$(dirname "$SFT_LOG")" "$OUTPUT_DIR"
exec > >(tee -a "$SFT_LOG") 2>&1
echo "=== resume 1-GPU full SFT @ $(date -Is) ==="
echo "  GPU: $CUDA_VISIBLE_DEVICES  resume: $RESUME_CHECKPOINT"
echo "  epochs: $NUM_EPOCHS (offset $EPOCH_OFFSET)  log: $SFT_LOG"

PORTS="${PORTS:-8014,8015,8016,8017}" "$LEGACY/teardown_openmoss.sh" 2>/dev/null || true
pkill -f 'app.moss_api' 2>/dev/null || true
sleep 2
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# shellcheck disable=SC1091
source "${FINETUNE_VENV:-$ROOT/.venv-finetune}/bin/activate"
SPEAKER_ROOT="$ROOT" MOSS_RT_TRAIN_DIR="$LOLI" \
  "$FINETUNE/run_moss_rt_finetune_train.sh"

latest=$(ls -d "$OUTPUT_DIR"/checkpoint-epoch-* 2>/dev/null | sort -V | tail -1)
if [[ -n "$latest" && -f "$latest/adapter_model.safetensors" ]]; then
  echo "=== merge → loli15s-full-1gpu ==="
  MERGE_CHECKPOINT="$latest" \
  MERGE_OUTPUT="$LOLI/exports/loli15s-full-1gpu" \
    "$ROOT/.venv-finetune/bin/python3" "$LOLI/scripts/distill.py" export merge
fi

echo "=== Done @ $(date -Is) ==="
echo "  latest: $latest"
echo "  merged: $LOLI/exports/loli15s-full-1gpu"
echo "  log: $SFT_LOG"
