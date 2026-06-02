#!/usr/bin/env bash
# Log GPU + RAM usage while teacher gen runs.
# Usage: ./scripts/monitor_teacher_gen_resources.sh [interval_sec] [log_file]

set -uo pipefail
INTERVAL="${1:-30}"
LOG="${2:-training/loli_15s/resource_monitor.log}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p "$(dirname "$LOG")"

echo "=== resource monitor started $(date -Is) interval=${INTERVAL}s ===" >>"$LOG"

while true; do
  {
    echo "--- $(date -Is) ---"
    free -h | grep -E '^(Mem|Swap):'
    if command -v nvidia-smi >/dev/null; then
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    fi
    n_batch2_disk=0
    n_batch2_shm=0
    if [[ -d training/loli_15s_batch2/wavs/v15 ]]; then
      n_batch2_disk=$(find training/loli_15s_batch2/wavs/v15 -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l)
    fi
    if [[ -d /dev/shm/loli15s_batch2_wavs/v15 ]]; then
      n_batch2_shm=$(find /dev/shm/loli15s_batch2_wavs/v15 -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l)
    fi
    echo "wavs: batch2_disk=$n_batch2_disk batch2_shm=$n_batch2_shm"
    moss_n=$(pgrep -f '[m]oss-tts-server' 2>/dev/null | wc -l)
    worker_n=$(pgrep -f '[b]uild_realtime_finetune_dataset' 2>/dev/null | wc -l)
    echo "moss-tts-server processes: $moss_n"
    echo "teacher workers: $worker_n"
    if [[ "$moss_n" -gt 4 ]]; then
      echo "ALERT: orphan moss-tts-server (run ./scripts/teardown_openmoss.sh)"
    fi
  } >>"$LOG" 2>&1
  sleep "$INTERVAL"
done
