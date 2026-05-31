#!/usr/bin/env bash
# No-ref native-voice LoRA SFT: strip ref_wav → preprocess → train.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
TRAIN_DIR="${MOSS_RT_TRAIN_DIR:-$ROOT/training/moss-realtime}"
FINETUNE="$SCRIPT_DIR"

export TRAIN_RAW="${TRAIN_RAW:-$TRAIN_DIR/train_raw.noref.jsonl}"
export PREPARED_GLOB="${PREPARED_GLOB:-$TRAIN_DIR/prepared/train_with_codes.rank*.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-$TRAIN_DIR/output/sft_ddp}"
export CHECKPOINT_LINK="${CHECKPOINT_LINK:-$TRAIN_DIR/checkpoints/latest}"

export NUM_GPUS="${NUM_GPUS:-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-12}"
export LEARNING_RATE="${LEARNING_RATE:-3e-6}"
export WARMUP_RATIO="${WARMUP_RATIO:-0}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export USE_LORA=1

mkdir -p "$TRAIN_DIR/logs" "$(dirname "$OUTPUT_DIR")"

if [[ ! -f "$TRAIN_RAW" ]]; then
  python3 "$FINETUNE/build_moss_rt_train_raw_noref.py"
fi
if ! compgen -G "$PREPARED_GLOB" >/dev/null; then
  echo "Preparing audio codes ($NUM_GPUS GPU)..."
  PREPARED="$TRAIN_DIR/prepared/train_with_codes.jsonl" \
    SPEAKER_ROOT="$ROOT" MOSS_RT_TRAIN_DIR="$TRAIN_DIR" \
    "$FINETUNE/run_moss_rt_finetune_preprocess.sh"
fi

echo "=== MOSS-RT no-ref LoRA SFT ==="
echo "  data:   $TRAIN_RAW"
echo "  epochs: $NUM_EPOCHS @ lr=$LEARNING_RATE"
echo "  output: $OUTPUT_DIR"
echo "  log:    $TRAIN_DIR/logs/finetune.log"
echo ""

SPEAKER_ROOT="$ROOT" MOSS_RT_TRAIN_DIR="$TRAIN_DIR" \
  "$FINETUNE/run_moss_rt_finetune_train.sh" 2>&1 | tee "$TRAIN_DIR/logs/finetune.log"
