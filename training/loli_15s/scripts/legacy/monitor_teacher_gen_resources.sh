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
    n_v15_disk=$(ls training/loli_15s/wavs/v15/*.wav 2>/dev/null | wc -l)
    n_v15_shm=0
    if [[ -d /dev/shm/loli15s_wavs/v15 ]]; then
      n_v15_shm=$(ls /dev/shm/loli15s_wavs/v15/*.wav 2>/dev/null | wc -l)
    fi
    echo "wavs: v15_disk=$n_v15_disk v15_shm=$n_v15_shm"
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
