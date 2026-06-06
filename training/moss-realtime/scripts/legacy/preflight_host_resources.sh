#!/usr/bin/env bash
# Refuse teacher-gen if host is too tight on RAM/VRAM (prevents hard freeze).
set -euo pipefail

MIN_AVAIL_GB="${MIN_AVAIL_GB:-12}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-14000}"
GPUS="${GPUS:-0,1,2}"

avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
avail_gb=$((avail_kb / 1024 / 1024))
if [[ "$avail_gb" -lt "$MIN_AVAIL_GB" ]]; then
  echo "ERROR: MemAvailable=${avail_gb}GiB < ${MIN_AVAIL_GB}GiB — reduce NUM_SHARDS or free RAM first" >&2
  echo "  tip: stop docker/airflow, use OPENMOSS_AUX_CPU=0, or run 2 GPUs not 3" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
for gpu in "${GPU_ARR[@]}"; do
  free_mb=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' -v g="$gpu" '$1==g {print $2; exit}')
  if [[ -z "$free_mb" || "$free_mb" -lt "$MIN_GPU_FREE_MB" ]]; then
    echo "ERROR: GPU $gpu only ${free_mb:-?} MiB free (need >= ${MIN_GPU_FREE_MB})" >&2
    exit 1
  fi
done

echo "preflight OK: MemAvailable=${avail_gb}GiB GPUs=${GPUS}"
