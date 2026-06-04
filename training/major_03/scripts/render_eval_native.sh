#!/usr/bin/env bash
# Merge epoch-11 LoRA → serve native voice (no ref) → write eval WAVs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MAJOR="$ROOT/training/major_03"
CKPT="${CKPT:-$MAJOR/checkpoints/latest}"
MERGED="${MERGED:-$MAJOR/exports/major03-epoch11-merged}"
PORT="${MOSS_RT_PORT:-8016}"
GPU="${MOSS_RT_GPU:-1}"
LOG="$MAJOR/logs/eval_native.log"
FINETUNE="$ROOT/training/moss-realtime/scripts/legacy/finetune"

mkdir -p "$MAJOR/logs" "$MAJOR/eval/listen" "$(dirname "$MERGED")"

exec > >(tee -a "$LOG") 2>&1
echo "=== major_03 native eval $(date -Is) ==="

PORTS=8014,8015,8016,8017 SPEAKER_ROOT="$ROOT" \
  "$ROOT/training/moss-realtime/scripts/legacy/teardown_openmoss.sh" 2>/dev/null || true
pkill -f '[a]pp.moss_api' 2>/dev/null || true
sleep 2

if [[ ! -f "$MERGED/config.json" ]]; then
  echo "=== Merge LoRA ==="
  SPEAKER_ROOT="$ROOT" MOSS_RT_TRAIN_DIR="$MAJOR" \
    "$ROOT/.venv-finetune/bin/python3" "$FINETUNE/merge_moss_rt_lora.py" \
      --checkpoint "$CKPT" \
      --output "$MERGED"
fi

fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

echo "=== Start MOSS-RT (native voice, merged weights) ==="
cd "$ROOT"
nohup env \
  MOSS_RT_GPU="$GPU" \
  MOSS_RT_PORT="$PORT" \
  MOSS_RT_MODEL_ID="$MERGED" \
  MOSS_RT_NATIVE_VOICE=true \
  MOSS_RT_STREAM_CODEC_BACKEND=torch \
  MOSS_RT_STREAM_DECODER_OVERLAP_FRAMES=0 \
  ./scripts/start-moss-realtime.sh \
  >> "$MAJOR/logs/moss_rt_eval.log" 2>&1 &
echo $! > "$MAJOR/logs/moss_rt_eval.pid"

echo "=== Generate samples ==="
MOSS_RT_TRAIN_DIR="$MAJOR" MOSS_RT_API="http://127.0.0.1:${PORT}" \
  "$ROOT/.venv-finetune/bin/python3" "$SCRIPT_DIR/generate_eval_samples.py" \
  --wait-health 900

echo "=== Done. Listen: file://$MAJOR/eval/listen/epoch11_native/index.html ==="
