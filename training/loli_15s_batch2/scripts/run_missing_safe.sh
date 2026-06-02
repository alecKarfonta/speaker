#!/usr/bin/env bash
# Generate only missing batch2 WAVs with reduced GPU load (2 shards, long stagger).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOG_DIR="$ROOT/training/loli_15s_batch2"

if [[ -d /dev/shm/loli15s_wavs ]]; then
  echo "Removing legacy tmpfs /dev/shm/loli15s_wavs"
  rm -rf /dev/shm/loli15s_wavs
fi

"$SCRIPT_DIR/build_missing_corpus.sh"
MISSING="$LOG_DIR/corpus/texts_missing.jsonl"
n=$(wc -l <"$MISSING" | tr -d ' ')
if [[ "$n" -eq 0 ]]; then
  echo "Nothing missing — batch2 complete."
  exit 0
fi

have=$(find "$LOG_DIR/wavs/v15" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
merged=$(find "$ROOT/training/loli_15s/wavs/v15_pruned" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
echo "Safe missing-only run: batch2=$have wavs, missing=$n lines, merged_loli=$merged (target 6000+)"
echo "Settings: 2 GPUs, 60s stagger, QC=lenient (no min-duration reject), lighten_host=1"

export CORPUS="$MISSING"
export NUM_SHARDS="${NUM_SHARDS:-2}"
export GPUS="${GPUS:-0,1}"
export PORTS="${PORTS:-8014,8015}"
export STAGGER_SEC="${STAGGER_SEC:-60}"
export MAX_DUR="${MAX_DUR:-32.0}"
export LIGHT_HOST="${LIGHT_HOST:-1}"
export CLEAR_SWAP=0
export MONITOR=1
# Teacher gen: reject only silent/empty/too-long — not short (~4–8s) MOSS clips.
export TEACHER_GEN_EXTRA_ARGS="${TEACHER_GEN_EXTRA_ARGS:---qc-lenient}"

exec "$SCRIPT_DIR/run_teacher_gen_parallel.sh" >>"$LOG_DIR/teacher_gen_missing_safe.log" 2>&1
