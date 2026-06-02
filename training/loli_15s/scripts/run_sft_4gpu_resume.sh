#!/usr/bin/env bash
# Resume LoRA SFT on 4 GPUs with equal prepared shards.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
FINETUNE="$LOLI/scripts/legacy/finetune"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NUM_GPUS=4
export MOSS_RT_TRAIN_DIR="$LOLI"
export SPEAKER_ROOT="$ROOT"
export OUTPUT_DIR="${OUTPUT_DIR:-$LOLI/output/sft_ddp_single}"
export PREPARED_GLOB="${PREPARED_GLOB:-$LOLI/prepared/train_with_codes_single.rank*.jsonl}"
export SFT_LOG="${SFT_LOG:-$LOLI/logs/sft_4gpu_resume.log}"
export FINETUNE_LOG="$SFT_LOG"

export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-4}"
export EPOCH_OFFSET="${EPOCH_OFFSET:-4}"
export RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-$OUTPUT_DIR/checkpoint-epoch-3}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

bak="$LOLI/prepared/pre_rebalance_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$bak"
cp -a "$LOLI"/prepared/train_with_codes_single.rank*.jsonl "$bak/" 2>/dev/null || true
echo "Backed up shards -> $bak"

echo "=== Rebalance prepared shards (4 equal ranks) ==="
NUM_SHARDS=4 "$SCRIPT_DIR/rebalance_prepared_shards.sh"
wc -l "$LOLI"/prepared/train_with_codes_single.rank*.jsonl

if [[ ! -f "$RESUME_CHECKPOINT/adapter_model.safetensors" ]]; then
  echo "Missing resume checkpoint: $RESUME_CHECKPOINT"
  exit 1
fi

echo "=== Resume 4-GPU SFT ==="
echo "  GPUs:        $CUDA_VISIBLE_DEVICES"
echo "  resume:      $RESUME_CHECKPOINT"
echo "  epochs:      $NUM_EPOCHS new (names checkpoint-epoch-$EPOCH_OFFSET ..)"
echo "  log:         $SFT_LOG"

# shellcheck disable=SC1091
source "${FINETUNE_VENV:-$ROOT/.venv-finetune}/bin/activate"
mkdir -p "$(dirname "$SFT_LOG")" "$OUTPUT_DIR"

SPEAKER_ROOT="$ROOT" MOSS_RT_TRAIN_DIR="$LOLI" \
  "$FINETUNE/run_moss_rt_finetune_train.sh" 2>&1 | tee -a "$SFT_LOG"
