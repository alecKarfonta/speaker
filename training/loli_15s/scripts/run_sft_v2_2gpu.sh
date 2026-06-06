#!/usr/bin/env bash
# Loli v2 SFT on 2 GPUs: preprocess → emotion oversample → resume from epoch-7.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
LEGACY="$ROOT/training/moss-realtime/scripts/legacy"

export SPEAKER_ROOT="$ROOT"
export MOSS_RT_TRAIN_DIR="$LOLI"
export SFT_V2=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS=2
export NUM_SHARDS=2

CKPT="${RESUME_CHECKPOINT:-$LOLI/output/sft_ddp_single/checkpoint-epoch-7}"
NUM_EPOCHS="${NUM_EPOCHS:-6}"
LR="${LEARNING_RATE:-5e-6}"
EPOCH_OFFSET="${EPOCH_OFFSET:-7}"
PREPARED_OUT="$LOLI/prepared/train_with_codes_single.jsonl"
PREPARED_GLOB="$LOLI/prepared/train_with_codes_single.rank*-of-00002.jsonl"
SFT_LOG="${SFT_LOG:-$LOLI/logs/sft_v2_2gpu_$(date +%Y%m%d_%H%M%S).log}"

if [[ ! -f "$CKPT/adapter_model.safetensors" ]]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 1
fi

mkdir -p "$LOLI/prepared" "$LOLI/logs"
exec > >(tee -a "$SFT_LOG") 2>&1
echo "=== loli v2 SFT (2 GPU) @ $(date -Is) ==="
echo "  GPUs: $CUDA_VISIBLE_DEVICES  log: $SFT_LOG"

echo "=== Stop MOSS inference servers ==="
PORTS="${PORTS:-8014,8015,8016,8017}" "$LEGACY/teardown_openmoss.sh" 2>/dev/null || true
if [[ -x "$LEGACY/lighten_host_for_teacher_gen.sh" ]]; then
  "$LEGACY/lighten_host_for_teacher_gen.sh" restore 2>/dev/null || true
fi
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

echo "=== filter single-turn train_raw ==="
"$SCRIPT_DIR/filter_single_turn_train_raw.sh"
python3 "$LOLI/scripts/legacy/finetune/build_moss_rt_train_raw_noref.py" 2>/dev/null || true

if [[ "${REFRESH_PREPARED:-1}" == "1" ]]; then
  echo "=== Clear stale prepared shards ==="
  if compgen -G "$LOLI/prepared/train_with_codes_single.rank*.jsonl" >/dev/null; then
    bak="$LOLI/prepared/pre_8902_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$bak"
    mv "$LOLI/prepared"/train_with_codes_single.rank*.jsonl "$bak/" 2>/dev/null || true
    echo "  archived old shards → $bak"
  fi
fi

echo "=== preprocess (2 GPU, 8902 clips) ==="
PREPARED_OUT="$PREPARED_OUT" TRAIN_RAW="$LOLI/train_raw.noref.jsonl" \
  python3 "$LOLI/scripts/distill.py" train preprocess --gpus 2 --noref

echo "=== emotion oversample (weight=${EMOTION_OVERSAMPLE:-2}) ==="
EMOTION_OVERSAMPLE="${EMOTION_OVERSAMPLE:-2}" NUM_SHARDS=2 \
  SRC_GLOB="train_with_codes_single.rank*-of-00002.jsonl" \
  "$SCRIPT_DIR/oversample_emotion_prepared.sh"

echo "=== resume SFT from epoch-7 ==="
export OUTPUT_DIR="$LOLI/output/sft_ddp_single"
export RESUME_CHECKPOINT="$CKPT"
export EPOCH_OFFSET="$EPOCH_OFFSET"
export NUM_EPOCHS="$NUM_EPOCHS"
export LEARNING_RATE="$LR"
export PREPARED_GLOB="$PREPARED_GLOB"
export FINETUNE_LOG="$SFT_LOG"

# shellcheck disable=SC1091
source "${FINETUNE_VENV:-$ROOT/.venv-finetune}/bin/activate"
SPEAKER_ROOT="$ROOT" MOSS_RT_TRAIN_DIR="$LOLI" \
  "$LOLI/scripts/legacy/finetune/run_moss_rt_finetune_noref.sh"

latest=$(ls -d "$OUTPUT_DIR"/checkpoint-epoch-* 2>/dev/null | sort -V | tail -1)
echo "=== merge → loli15s-v2-merged ==="
MERGE_CHECKPOINT="${latest:-$CKPT}" \
MERGE_OUTPUT="$LOLI/exports/loli15s-v2-merged" \
SFT_V2=1 \
  python3 "$LOLI/scripts/distill.py" export merge

echo "=== Done @ $(date -Is) ==="
echo "  merged: $LOLI/exports/loli15s-v2-merged"
echo "  log: $SFT_LOG"
