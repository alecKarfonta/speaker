#!/usr/bin/env bash
# Stop MOSS inference servers, refresh no-ref jsonl + prepared shards, run 4-GPU preprocess → SFT.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
LEGACY="$ROOT/training/moss-realtime/scripts/legacy"
FINETUNE="$LOLI/scripts/legacy/finetune"

export NUM_GPUS="${NUM_GPUS:-4}"
export MOSS_RT_TRAIN_DIR="$LOLI"
export SPEAKER_ROOT="$ROOT"
export OUTPUT_DIR="${OUTPUT_DIR:-$LOLI/output/sft_ddp_single}"
export PREPARED_OUT="${PREPARED_OUT:-$LOLI/prepared/train_with_codes_single.jsonl}"
export PREPARED_GLOB="${PREPARED_GLOB:-$LOLI/prepared/train_with_codes_single.rank*.jsonl}"

echo "=== Stop MOSS teacher / inference servers ==="
PORTS="${PORTS:-8014,8015,8016,8017}" "$LEGACY/teardown_openmoss.sh"
if [[ -x "$LEGACY/lighten_host_for_teacher_gen.sh" ]]; then
  "$LEGACY/lighten_host_for_teacher_gen.sh" restore 2>/dev/null || true
fi

echo "=== Single-turn train_raw only (drop mt_* multi-turn) ==="
"$SCRIPT_DIR/filter_single_turn_train_raw.sh"

if [[ "${REFRESH_PREPARED:-1}" == "1" ]]; then
  echo "=== Clear stale prepared shards (old ~3.5k run) ==="
  mkdir -p "$LOLI/prepared"
  if compgen -G "$LOLI/prepared/train_with_codes_single.rank*.jsonl" >/dev/null; then
    bak="$LOLI/prepared/pre_6k_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$bak"
    mv "$LOLI/prepared"/train_with_codes_single.rank*.jsonl "$bak/" 2>/dev/null || true
    echo "  moved old shards → $bak"
  fi
fi

echo "=== GPU memory (want ~24G free on 0-3) ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

cd "$ROOT"
if [[ "${RUN_PREPROCESS:-1}" == "1" ]]; then
  echo "=== Preprocess audio codes ($NUM_GPUS GPUs) ==="
  python3 "$LOLI/scripts/distill.py" train preprocess --noref --gpus "$NUM_GPUS"
fi

if [[ "${RUN_SFT:-1}" == "1" ]]; then
  SFT_LOG="${SFT_LOG:-$LOLI/logs/sft_single_turn.log}"
  export FINETUNE_LOG="$SFT_LOG"
  echo "=== LoRA SFT ($NUM_GPUS GPUs) ==="
  echo "  progress log: $SFT_LOG"
  NUM_EPOCHS="${NUM_EPOCHS:-8}" LEARNING_RATE="${LEARNING_RATE:-1e-5}" \
    WARMUP_RATIO="${WARMUP_RATIO:-0.05}" LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}" \
    "$LOLI/scripts/legacy/finetune/run_moss_rt_finetune_noref.sh" \
    >> "$SFT_LOG" 2>&1
fi
