#!/usr/bin/env bash
# Wait for major_03 teacher gen to finish, then build + run loli_15s batch2 (3000 clips).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MAJOR_DIR="$ROOT/training/major_03"
LOLI_DIR="$ROOT/training/loli_15s_batch2"
LOG="${CHAIN_LOG:-$MAJOR_DIR/chain_loli15s.log}"
PID_FILE="${CHAIN_PID_FILE:-$MAJOR_DIR/chain_loli15s.pid}"
TARGET_WAVS="${TARGET_WAVS:-3000}"
POLL_SEC="${POLL_SEC:-120}"

log() {
  echo "$(date -Is) $*" | tee -a "$LOG"
}

count_major_wavs() {
  find "$MAJOR_DIR/wavs/v15" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' '
}

major_workers() {
  pgrep -f '[b]uild_realtime_finetune_dataset.py.*major_03' 2>/dev/null | wc -l | tr -d ' '
}

major_launcher() {
  pgrep -f '[r]un_major03_teacher_gen_parallel' 2>/dev/null | wc -l | tr -d ' '
}

major_train_rows() {
  if [[ -f "$MAJOR_DIR/train_raw.jsonl" ]]; then
    wc -l < "$MAJOR_DIR/train_raw.jsonl" | tr -d ' '
  else
    echo 0
  fi
}

wait_for_major() {
  log "Watching major_03 (target ${TARGET_WAVS} wavs)..."
  while true; do
    n_wav=$(count_major_wavs)
    n_row=$(major_train_rows)
    workers=$(major_workers)
    launcher=$(major_launcher)
    if [[ "$n_wav" -ge "$TARGET_WAVS" ]] && [[ "$workers" -eq 0 ]] && [[ "$launcher" -eq 0 ]]; then
      if [[ "$n_row" -ge "$TARGET_WAVS" ]]; then
        log "major_03 complete: wavs=$n_wav rows=$n_row"
        return 0
      fi
      log "major_03 idle but train_raw only $n_row rows — waiting for merge..."
    fi
    if grep -q 'Done:.*train_raw.jsonl' "$MAJOR_DIR/teacher_gen_full.log" 2>/dev/null \
      && [[ "$workers" -eq 0 ]] && [[ "$launcher" -eq 0 ]]; then
      n_wav=$(count_major_wavs)
      n_row=$(major_train_rows)
      log "major_03 launcher finished (log Done); wavs=$n_wav rows=$n_row"
      if [[ "$n_wav" -ge "$((TARGET_WAVS * 95 / 100))" ]]; then
        return 0
      fi
    fi
    log "  major_03: wavs=$n_wav/$TARGET_WAVS rows=$n_row workers=$workers launcher=$launcher"
    sleep "$POLL_SEC"
  done
}

run_loli_batch2() {
  log "Building loli_15s_batch2 corpus (3000 lines, b2_* ids)..."
  SEED=20260531 SINGLE=3000 "$LOLI_DIR/scripts/build_corpus.sh" 2>&1 | tee -a "$LOG"

  log "Starting loli_15s_batch2 teacher generation..."
  STAGGER_SEC=20 CLEAR_SWAP=0 \
    "$LOLI_DIR/scripts/run_teacher_gen_parallel.sh" 2>&1 | tee -a "$LOG"

  local n
  n=$(find "$LOLI_DIR/wavs/v15" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
  log "loli_15s_batch2 finished: $n wavs, $(wc -l < "$LOLI_DIR/train_raw.jsonl") train_raw rows"
}

main() {
  mkdir -p "$(dirname "$LOG")" "$LOLI_DIR"
  log "chain_loli15s_after_major03 started (pid $$)"
  echo "$$" > "$PID_FILE"
  wait_for_major
  run_loli_batch2
  log "Pipeline complete."
}

main "$@"
