#!/usr/bin/env bash
# Resume batch2 teacher gen after a crash (--skip-existing, frees legacy tmpfs).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOG_DIR="$ROOT/training/loli_15s_batch2"

if [[ -d /dev/shm/loli15s_wavs ]]; then
  echo "Removing legacy tmpfs /dev/shm/loli15s_wavs (original run is on disk under training/loli_15s/wavs)"
  rm -rf /dev/shm/loli15s_wavs
fi

n=$(find "$LOG_DIR/wavs/v15" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
echo "Resuming loli_15s_batch2: $n wavs on disk, target 3000"

STAGGER_SEC="${STAGGER_SEC:-20}" \
LIGHT_HOST="${LIGHT_HOST:-1}" \
CLEAR_SWAP=0 \
exec "$SCRIPT_DIR/run_teacher_gen_parallel.sh" "$@"
