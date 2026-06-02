#!/usr/bin/env bash
# Resume v15 teacher gen without restarting moss or redoing completed WAVs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEGACY="${LEGACY_DIR:-$SCRIPT_DIR}"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
cd "$ROOT"

LOG_DIR="${LOG_DIR:-${MOSS_RT_TRAIN_DIR:-$ROOT/training/loli_15s}}"
WAV_STAGING="${WAV_STAGING:-/dev/shm/loli15s_wavs}"
WAV_SYNC_SEC="${WAV_SYNC_SEC:-120}"
TEACHERS="${TEACHERS:-v15}"
NUM_SHARDS="${NUM_SHARDS:-4}"
PORTS="${PORTS:-8014,8015,8016,8017}"
LIGHT_HOST="${LIGHT_HOST:-1}"
REUSE_MOSS="${REUSE_MOSS:-1}"

IFS=',' read -r -a PORT_ARR <<< "$PORTS"

mkdir -p "$WAV_STAGING/v15" "${LOG_DIR}/wavs/v15"
rsync -a "${LOG_DIR}/wavs/v15/" "$WAV_STAGING/v15/"
rsync -a "$WAV_STAGING/v15/" "${LOG_DIR}/wavs/v15/"

if [[ "$LIGHT_HOST" == "1" ]]; then
  "${LEGACY}/lighten_host_for_teacher_gen.sh" stop 2>/dev/null || true
fi

moss_ok=0
if [[ "$REUSE_MOSS" == "1" ]]; then
  moss_ok=1
  for port in "${PORT_ARR[@]}"; do
    if ! curl -sf "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q ok; then
      moss_ok=0
      break
    fi
  done
fi

if [[ "$moss_ok" != "1" ]]; then
  echo "Moss not healthy on all ports — running full launcher..."
  exec SPEAKER_ROOT="$ROOT" MOSS_RT_TRAIN_DIR="$LOG_DIR" "${LEGACY}/run_loli15s_teacher_gen_parallel.sh"
fi

echo "Reusing moss servers; starting workers only."
pkill -f '[b]uild_realtime_finetune_dataset' 2>/dev/null || true
sleep 1

if ! pgrep -f '[m]onitor_teacher_gen_resources' >/dev/null 2>&1; then
  setsid bash "${LEGACY}/monitor_teacher_gen_resources.sh" 30 "${LOG_DIR}/resource_monitor.log" \
    >> "${LOG_DIR}/resource_monitor.log" 2>&1 < /dev/null &
fi

if ! pgrep -f 'sync_teacher_wavs_from_staging' >/dev/null 2>&1; then
  (
    while true; do
      sleep "$WAV_SYNC_SEC"
      STAGING="$WAV_STAGING" DEST="${LOG_DIR}/wavs" "${LEGACY}/sync_teacher_wavs_from_staging.sh" \
        >> "${LOG_DIR}/wav_sync.log" 2>&1 || true
    done
  ) &
fi

PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  port="${PORT_ARR[$i]}"
  log="${LOG_DIR}/teacher_gen.shard${i}.log"
  echo "Resume shard $i -> :${port}"
  SPEAKER_ROOT="$ROOT" MOSS_RT_TRAIN_DIR="$LOG_DIR" PYTHONUNBUFFERED=1 \
    nohup python3 "${LEGACY}/build_realtime_finetune_dataset.py" \
      --teachers "$TEACHERS" \
      --wav-dir "$WAV_STAGING" \
      --wav-rel-root "training/loli_15s/wavs" \
      --api "http://127.0.0.1:${port}/tts" \
      --shard-id "$i" \
      --num-shards "$NUM_SHARDS" \
      --skip-existing --no-stt --no-auto-start \
      >> "$log" 2>&1 &
  PIDS+=($!)
  sleep 2
done

echo "Resumed workers: ${PIDS[*]}"
wait "${PIDS[@]}"
pkill -f 'sync_teacher_wavs_from_staging' 2>/dev/null || true
STAGING="$WAV_STAGING" DEST="${LOG_DIR}/wavs" "${LEGACY}/sync_teacher_wavs_from_staging.sh"
cat "${LOG_DIR}"/train_raw.shard*.jsonl > "${LOG_DIR}/train_raw.jsonl" 2>/dev/null || true
[[ "$LIGHT_HOST" == "1" ]] && "${LEGACY}/lighten_host_for_teacher_gen.sh" restore 2>/dev/null || true
echo "Done."
