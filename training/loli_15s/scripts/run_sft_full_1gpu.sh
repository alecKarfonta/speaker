#!/usr/bin/env bash
# Full overnight LoRA SFT on 1 GPU — fresh train on 8.9k QC'd corpus (no resume).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
LEGACY="$ROOT/training/moss-realtime/scripts/legacy"

export SPEAKER_ROOT="$ROOT"
export MOSS_RT_TRAIN_DIR="$LOLI"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export NUM_GPUS=1
export NUM_SHARDS=1

OUTPUT_DIR="${OUTPUT_DIR:-$LOLI/output/sft_1gpu_full}"
PREPARED_PREFIX="${PREPARED_PREFIX:-train_with_codes_1gpu}"
PREPARED_GLOB="$LOLI/prepared/${PREPARED_PREFIX}.rank00000-of-00001.jsonl"
NUM_EPOCHS="${NUM_EPOCHS:-8}"
LR="${LEARNING_RATE:-1e-5}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
SFT_LOG="${SFT_LOG:-$LOLI/logs/sft_full_1gpu_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOLI/prepared" "$LOLI/logs" "$OUTPUT_DIR"
exec > >(tee -a "$SFT_LOG") 2>&1
echo "=== loli full SFT (1 GPU) @ $(date -Is) ==="
echo "  GPU: $CUDA_VISIBLE_DEVICES  epochs: $NUM_EPOCHS  lr: $LR"
echo "  output: $OUTPUT_DIR  log: $SFT_LOG"

echo "=== Stop MOSS inference / teacher servers ==="
PORTS="${PORTS:-8014,8015,8016,8017}" "$LEGACY/teardown_openmoss.sh" 2>/dev/null || true
pkill -f 'app.moss_api' 2>/dev/null || true
sleep 2
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

echo "=== filter single-turn train_raw ==="
"$SCRIPT_DIR/filter_single_turn_train_raw.sh"
python3 "$LOLI/scripts/legacy/finetune/build_moss_rt_train_raw_noref.py" 2>/dev/null || true

if [[ "${RUN_PREPROCESS:-0}" == "1" ]]; then
  echo "=== preprocess (1 GPU) ==="
  PREPARED_OUT="$LOLI/prepared/${PREPARED_PREFIX}.jsonl" \
    TRAIN_RAW="$LOLI/train_raw.noref.jsonl" \
    python3 "$LOLI/scripts/distill.py" train preprocess --gpus 1 --noref
  echo "=== emotion oversample ==="
  EMOTION_OVERSAMPLE="${EMOTION_OVERSAMPLE:-2}" NUM_SHARDS=1 \
    SRC_GLOB="${PREPARED_PREFIX}.rank*-of-00001.jsonl" \
    OUT_PREFIX="$PREPARED_PREFIX" \
    "$SCRIPT_DIR/oversample_emotion_prepared.sh"
elif [[ -f "$PREPARED_GLOB" && "${REFRESH_PREPARED:-0}" != "1" ]]; then
  n=$(wc -l < "$PREPARED_GLOB")
  echo "=== Reuse prepared shard ($n rows) ==="
else
  echo "=== Build 1-GPU shard from 2-GPU oversampled prepared ==="
  SRC_GLOB="${SRC_GLOB:-train_with_codes_single.rank*-of-00002.jsonl}"
  if ! compgen -G "$LOLI/prepared/$SRC_GLOB" >/dev/null; then
    echo "Missing $LOLI/prepared/$SRC_GLOB — set RUN_PREPROCESS=1" >&2
    exit 1
  fi
  NUM_SHARDS=1 OUT_PREFIX="$PREPARED_PREFIX" SRC_GLOB="$SRC_GLOB" \
    "$SCRIPT_DIR/rebalance_prepared_shards.sh"
fi

if [[ ! -f "$PREPARED_GLOB" ]]; then
  echo "Prepared shard missing: $PREPARED_GLOB" >&2
  exit 1
fi
echo "Prepared: $PREPARED_GLOB ($(wc -l < "$PREPARED_GLOB") rows)"

echo "=== LoRA SFT (fresh, $NUM_EPOCHS epochs) ==="
export OUTPUT_DIR
export PREPARED_GLOB
export NUM_EPOCHS
export LEARNING_RATE="$LR"
export GRAD_ACCUM
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
unset RESUME_CHECKPOINT
unset EPOCH_OFFSET
export FINETUNE_LOG="$SFT_LOG"

# shellcheck disable=SC1091
source "${FINETUNE_VENV:-$ROOT/.venv-finetune}/bin/activate"
SPEAKER_ROOT="$ROOT" MOSS_RT_TRAIN_DIR="$LOLI" \
  "$LOLI/scripts/legacy/finetune/run_moss_rt_finetune_noref.sh"

latest=$(ls -d "$OUTPUT_DIR"/checkpoint-epoch-* 2>/dev/null | sort -V | tail -1)
if [[ -n "$latest" ]]; then
  echo "=== merge → loli15s-full-1gpu ==="
  MERGE_CHECKPOINT="$latest" \
  MERGE_OUTPUT="$LOLI/exports/loli15s-full-1gpu" \
    .venv-finetune/bin/python3 "$LOLI/scripts/distill.py" export merge
fi

echo "=== Done @ $(date -Is) ==="
echo "  checkpoints: $OUTPUT_DIR"
echo "  merged: $LOLI/exports/loli15s-full-1gpu"
echo "  log: $SFT_LOG"
