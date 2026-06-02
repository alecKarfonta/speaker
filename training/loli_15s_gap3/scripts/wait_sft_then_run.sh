#!/usr/bin/env bash
# Wait for current 4-GPU SFT to finish, then build gap corpus + teacher gen.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
LOG="$ROOT/training/loli_15s_gap3/logs/wait_sft_then_run.log"
FINETUNE_LOG="${FINETUNE_LOG:-$LOLI/logs/finetune.log}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOLI/output/sft_ddp_single}"
POLL_SEC="${POLL_SEC:-120}"
NUM_EPOCHS="${NUM_EPOCHS:-8}"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo "=== wait_sft_then_run $(date -Is) ==="
echo "Watching SFT: $OUTPUT_DIR (epochs=$NUM_EPOCHS)"

sft_running() {
  pgrep -f 'moss_tts_realtime/finetuning/sft.py.*sft_ddp_single' >/dev/null 2>&1 \
    || pgrep -f 'accelerate launch.*sft_ddp_single' >/dev/null 2>&1
}

epoch_done() {
  local n=0
  if [[ -d "$OUTPUT_DIR" ]]; then
    n=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-epoch-*' 2>/dev/null | wc -l | tr -d ' ')
  fi
  echo "$n"
}

while sft_running; do
  ep=$(epoch_done)
  tail -1 "$FINETUNE_LOG" 2>/dev/null || true
  echo "[$(date -Is)] SFT still running (checkpoints=$ep / $NUM_EPOCHS)"
  sleep "$POLL_SEC"
done

echo "[$(date -Is)] SFT processes gone. checkpoints=$(epoch_done)"

# Extra guard: wait until GPUs mostly free
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  if [[ "$used" -lt 8000 ]]; then
    break
  fi
  echo "[$(date -Is)] GPUs still busy (${used} MiB total used), waiting..."
  sleep 30
done

echo "[$(date -Is)] Building gap corpus..."
"$SCRIPT_DIR/build_corpus.sh"

echo "[$(date -Is)] Starting gap3 teacher generation (4 GPU)..."
"$SCRIPT_DIR/run_teacher_gen_parallel.sh"

echo "[$(date -Is)] Teacher gen finished — merging into loli_15s..."
"$SCRIPT_DIR/finish_merge_to_loli15s.sh"

echo "=== All done $(date -Is) ==="
