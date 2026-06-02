#!/usr/bin/env bash
# LoRA SFT on MOSS-TTS-Realtime (multi-GPU DDP).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
MOSS_DIR="${MOSS_TTS_DIR:-$ROOT/third_party/MOSS-TTS}"
VENV="${FINETUNE_VENV:-$ROOT/.venv-finetune}"
NUM_GPUS="${NUM_GPUS:-4}"
TRAIN_DIR="${MOSS_RT_TRAIN_DIR:-$ROOT/training/loli_15s}"
USE_LORA="${USE_LORA:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
EPOCH_OFFSET="${EPOCH_OFFSET:-0}"
PREPARED_GLOB="${PREPARED_GLOB:-$TRAIN_DIR/prepared/train_with_codes.rank*.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$TRAIN_DIR/output/sft_ddp}"
CHECKPOINT_LINK="${CHECKPOINT_LINK:-$TRAIN_DIR/checkpoints/latest}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -d "$MOSS_DIR" ]]; then
  echo "Run: python training/loli_15s/scripts/distill.py env setup"
  exit 1
fi
shopt -s nullglob
shards=($PREPARED_GLOB)
if [[ ${#shards[@]} -eq 0 ]]; then
  echo "No prepared shards at $PREPARED_GLOB — run: distill.py train preprocess"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$OUTPUT_DIR" "$(dirname "$CHECKPOINT_LINK")"

ACCEL_CFG="$MOSS_DIR/moss_tts_realtime/finetuning/configs/accelerate_ddp_8gpu.yaml"
if [[ "$NUM_GPUS" -eq 2 && -f "$TRAIN_DIR/configs/accelerate_ddp_2gpu.yaml" ]]; then
  ACCEL_CFG="$TRAIN_DIR/configs/accelerate_ddp_2gpu.yaml"
elif [[ "$NUM_GPUS" -lt 8 && -f "$TRAIN_DIR/configs/accelerate_ddp_4gpu.yaml" ]]; then
  ACCEL_CFG="$TRAIN_DIR/configs/accelerate_ddp_4gpu.yaml"
fi

cd "$MOSS_DIR"
SFT_ARGS=(
  --model-path OpenMOSS-Team/MOSS-TTS-Realtime
  --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer
  --train-jsonl "$PREPARED_GLOB"
  --output-dir "$OUTPUT_DIR"
  --per-device-batch-size 1
  --gradient-accumulation-steps "$GRAD_ACCUM"
  --learning-rate "$LEARNING_RATE"
  --warmup-ratio "$WARMUP_RATIO"
  --lr-scheduler-type "$LR_SCHEDULER_TYPE"
  --num-epochs "$NUM_EPOCHS"
  --mixed-precision bf16
  --gradient-checkpointing
)
if [[ "$USE_LORA" == "1" ]]; then
  SFT_ARGS+=(--lora --lora-r 16 --lora-alpha 32 --lora-dropout 0.05)
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  SFT_ARGS+=(--resume-from-checkpoint "$RESUME_CHECKPOINT" --epoch-offset "$EPOCH_OFFSET")
fi

accelerate launch \
  --num_processes "$NUM_GPUS" \
  ${ACCEL_CFG:+--config_file "$ACCEL_CFG"} \
  moss_tts_realtime/finetuning/sft.py \
  "${SFT_ARGS[@]}"

latest=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1)
if [[ -n "$latest" ]]; then
  ln -sfn "$latest" "$CHECKPOINT_LINK"
  echo "Checkpoint linked: $CHECKPOINT_LINK -> $latest"
fi
