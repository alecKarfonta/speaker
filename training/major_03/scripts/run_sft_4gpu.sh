#!/usr/bin/env bash
# 4-GPU no-ref LoRA SFT for major_03 (expects prepared/train_with_codes.rank*.jsonl).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MAJOR="$ROOT/training/major_03"
LEGACY="$ROOT/training/moss-realtime/scripts/legacy"
FINETUNE="$LEGACY/finetune"

export NUM_GPUS="${NUM_GPUS:-4}"
export MOSS_RT_TRAIN_DIR="$MAJOR"
export SPEAKER_ROOT="$ROOT"
export PREPARED_GLOB="${PREPARED_GLOB:-$MAJOR/prepared/train_with_codes.rank*.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-$MAJOR/output/sft_ddp}"
export CHECKPOINT_LINK="${CHECKPOINT_LINK:-$MAJOR/checkpoints/latest}"
export NUM_EPOCHS="${NUM_EPOCHS:-12}"
export LEARNING_RATE="${LEARNING_RATE:-3e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
SFT_LOG="${SFT_LOG:-$MAJOR/logs/sft.log}"

mkdir -p "$MAJOR/logs" "$(dirname "$OUTPUT_DIR")"

echo "=== Stop MOSS inference / teacher servers ==="
PORTS="${PORTS:-8014,8015,8016,8017}" SPEAKER_ROOT="$ROOT" "$LEGACY/teardown_openmoss.sh" 2>/dev/null || true
pkill -f '[a]pp.moss_api' 2>/dev/null || true

echo "=== GPU memory ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

if ! compgen -G "$PREPARED_GLOB" >/dev/null; then
  echo "Missing prepared shards — run: ./scripts/run_preprocess_only.sh" >&2
  exit 1
fi

echo "=== LoRA SFT ($NUM_GPUS GPUs) ==="
echo "  shards: $PREPARED_GLOB"
echo "  epochs: $NUM_EPOCHS lr=$LEARNING_RATE"
echo "  output: $OUTPUT_DIR"
echo "  log:    $SFT_LOG"

cd "$ROOT"
SPEAKER_ROOT="$ROOT" MOSS_RT_TRAIN_DIR="$MAJOR" \
  "$FINETUNE/run_moss_rt_finetune_noref.sh" 2>&1 | tee -a "$SFT_LOG"
