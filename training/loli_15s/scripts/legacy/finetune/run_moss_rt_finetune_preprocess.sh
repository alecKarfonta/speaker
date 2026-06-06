#!/usr/bin/env bash
# Preprocess train_raw.jsonl → audio codes (multi-GPU).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
MOSS_DIR="${MOSS_TTS_DIR:-$ROOT/third_party/MOSS-TTS}"
VENV="${FINETUNE_VENV:-$ROOT/.venv-finetune}"
NUM_GPUS="${NUM_GPUS:-4}"
TRAIN_DIR="${MOSS_RT_TRAIN_DIR:-$ROOT/training/loli_15s}"
TRAIN_RAW="${TRAIN_RAW:-$TRAIN_DIR/train_raw.jsonl}"
PREPARED="${PREPARED:-${PREPARED_OUT:-$TRAIN_DIR/prepared/train_with_codes.jsonl}}"

if [[ ! -d "$MOSS_DIR" ]]; then
  echo "Run: python training/loli_15s/scripts/distill.py env setup"
  exit 1
fi
if [[ ! -f "$TRAIN_RAW" ]]; then
  echo "Missing $TRAIN_RAW — run: distill.py teacher gen"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
mkdir -p "$(dirname "$PREPARED")"

ACCEL_CFG="$MOSS_DIR/moss_tts_realtime/finetuning/configs/accelerate_ddp_8gpu.yaml"
if [[ "$NUM_GPUS" -eq 1 && -f "$TRAIN_DIR/configs/accelerate_ddp_1gpu.yaml" ]]; then
  ACCEL_CFG="$TRAIN_DIR/configs/accelerate_ddp_1gpu.yaml"
elif [[ "$NUM_GPUS" -eq 2 && -f "$TRAIN_DIR/configs/accelerate_ddp_2gpu.yaml" ]]; then
  ACCEL_CFG="$TRAIN_DIR/configs/accelerate_ddp_2gpu.yaml"
elif [[ "$NUM_GPUS" -lt 8 && -f "$TRAIN_DIR/configs/accelerate_ddp_4gpu.yaml" ]]; then
  ACCEL_CFG="$TRAIN_DIR/configs/accelerate_ddp_4gpu.yaml"
fi

export PYTHONPATH="${MOSS_DIR}${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
accelerate launch \
  --num_processes "$NUM_GPUS" \
  ${ACCEL_CFG:+--config_file "$ACCEL_CFG"} \
  "$MOSS_DIR/moss_tts_realtime/finetuning/prepare_data.py" \
  --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --device auto \
  --input-jsonl "$TRAIN_RAW" \
  --output-jsonl "$PREPARED"

echo "Prepared shards under $(dirname "$PREPARED")"
