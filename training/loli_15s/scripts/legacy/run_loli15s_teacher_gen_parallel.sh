#!/usr/bin/env bash
# Parallel teacher WAV generation — one openmoss server per GPU shard.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEGACY="${LEGACY_DIR:-$SCRIPT_DIR}"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
OPENMOSS_START="${OPENMOSS_START:-$ROOT/training/moss-realtime/scripts/legacy/start-openmoss.sh}"
cd "$ROOT"

NUM_SHARDS="${NUM_SHARDS:-2}"
GPUS="${GPUS:-0,1}"
PORTS="${PORTS:-8014,8015}"
MIN_AVAIL_GB="${MIN_AVAIL_GB:-12}"
# aux on CPU saves VRAM but duplicates codec RAM per server — avoid when NUM_SHARDS>1
OPENMOSS_AUX_CPU="${OPENMOSS_AUX_CPU:-0}"
# Extra codec headroom so teacher clips don't cut off the last word
export OPENMOSS_DUR_SLACK="${OPENMOSS_DUR_SLACK:-1.25}"
export OPENMOSS_MAX_EXTRA="${OPENMOSS_MAX_EXTRA:-40}"
TEACHERS="${TEACHERS:-v15}"
LOG_DIR="${LOG_DIR:-${MOSS_RT_TRAIN_DIR:-$ROOT/training/loli_15s}}"
STAGGER_SEC="${STAGGER_SEC:-45}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-16000}"
WAV_STAGING="${WAV_STAGING:-/dev/shm/loli15s_wavs}"
WAV_SYNC_SEC="${WAV_SYNC_SEC:-120}"
LIGHT_HOST="${LIGHT_HOST:-0}"
CLEAR_SWAP="${CLEAR_SWAP:-0}"
VOICE_QC="${VOICE_QC:-1}"
MIN_COS_REF="${MIN_COS_REF:-0.5}"
MIN_COS_TEACHER="${MIN_COS_TEACHER:-0.5}"
EXTRA_ARGS=(--skip-existing --no-stt --no-auto-start)
if [[ "$VOICE_QC" == "1" ]]; then
  EXTRA_ARGS+=(--voice-qc --min-cos-ref "$MIN_COS_REF" --min-cos-teacher "$MIN_COS_TEACHER")
  EXTRA_ARGS+=(
    --teacher-train-raw "${TEACHER_TRAIN_RAW:-$ROOT/training/loli_15s/train_raw.jsonl}"
    --teacher-pool "${TEACHER_POOL:-$ROOT/training/loli_15s/wavs/v15_pruned}"
  )
fi
MONITOR="${MONITOR:-1}"
HEALTH_LOG_DIR="${HEALTH_LOG_DIR:-${LOG_DIR}/logs/health}"
WATCHDOG="$ROOT/training/moss-realtime/scripts/legacy/watchdog_server_health.py"
PREFLIGHT="$ROOT/training/moss-realtime/scripts/legacy/preflight_host_resources.sh"
SYNC_PID=""

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
IFS=',' read -r -a PORT_ARR <<< "$PORTS"
if [[ "${#GPU_ARR[@]}" -ne "$NUM_SHARDS" || "${#PORT_ARR[@]}" -ne "$NUM_SHARDS" ]]; then
  echo "GPUS and PORTS must each have NUM_SHARDS=$NUM_SHARDS entries" >&2
  exit 1
fi

gpu_free_mb() {
  local gpu=$1
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' -v g="$gpu" '$1==g {print $2; exit}'
}

wait_openmoss() {
  local port=$1
  for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${port}/health" 2>/dev/null | grep -q ok; then
      echo "openmoss :${port} ready"
      return 0
    fi
    sleep 2
  done
  echo "openmoss :${port} failed to start" >&2
  tail -30 "/tmp/openmoss-server-${port}.log" >&2 || true
  return 1
}

start_openmoss() {
  local gpu=$1 port=$2
  local free_mb
  free_mb=$(gpu_free_mb "$gpu" || echo 0)
  if [[ "$free_mb" -lt "$MIN_GPU_FREE_MB" ]]; then
    echo "WARN: GPU $gpu only ${free_mb} MiB free (want >= ${MIN_GPU_FREE_MB})" >&2
  fi
  fuser -k "${port}/tcp" 2>/dev/null || true
  sleep 1
  OPENMOSS_MAIN_GPU="$gpu" OPENMOSS_PORT="$port" OPENMOSS_AUX_CPU="$OPENMOSS_AUX_CPU" \
    OPENMOSS_MODEL_VERSION=v15 SPEAKER_ROOT="$ROOT" nohup "$OPENMOSS_START" \
    >> "/tmp/openmoss-server-${port}.log" 2>&1 &
  wait_openmoss "$port"
}

moss_count() {
  local n
  n=$(pgrep -f '[m]oss-tts-server' 2>/dev/null | wc -l | tr -d ' ')
  echo "${n:-0}"
}

gpu_ghost_mb() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 >= 200 && $2 < 1200 {print $1":"$2}'
}

stop_all() {
  pkill -f "build_realtime_finetune_dataset.py" 2>/dev/null || true
  pkill -f "monitor_teacher_gen_resources.sh" 2>/dev/null || true
  pkill -f "watchdog_server_health.py" 2>/dev/null || true
  PORTS="${PORTS}" SPEAKER_ROOT="$ROOT" "${LEGACY}/teardown_openmoss.sh"
}

preflight_gpus() {
  local ghosts
  ghosts=$(gpu_ghost_mb | tr '\n' ' ')
  if [[ -n "${ghosts// }" ]]; then
    echo "WARN: GPUs with ghost VRAM allocations: $ghosts" >&2
    echo "Run: distill.py teacher teardown" >&2
  fi
  for gpu in "${GPU_ARR[@]}"; do
    local free_mb
    free_mb=$(gpu_free_mb "$gpu" || echo 0)
    if [[ "$free_mb" -lt "$MIN_GPU_FREE_MB" ]]; then
      echo "ERROR: GPU $gpu only ${free_mb} MiB free (need >= ${MIN_GPU_FREE_MB})" >&2
      exit 1
    fi
  done
}

cleanup() {
  if [[ -n "${SYNC_PID:-}" ]] && kill -0 "$SYNC_PID" 2>/dev/null; then
    kill "$SYNC_PID" 2>/dev/null || true
  fi
  if [[ "${LIGHT_HOST:-0}" == "1" ]]; then
    "${LEGACY}/lighten_host_for_teacher_gen.sh" restore 2>/dev/null || true
  fi
  STAGING="$WAV_STAGING" DEST="${LOG_DIR}/wavs" "${LEGACY}/sync_teacher_wavs_from_staging.sh" 2>/dev/null || true
}
trap cleanup EXIT

echo "Stopping prior teacher-gen workers..."
stop_all
if [[ -x "$PREFLIGHT" ]]; then
  MIN_AVAIL_GB="$MIN_AVAIL_GB" GPUS="$GPUS" MIN_GPU_FREE_MB="$MIN_GPU_FREE_MB" "$PREFLIGHT"
fi
preflight_gpus
echo "OPENMOSS_AUX_CPU=$OPENMOSS_AUX_CPU (0 = aux on GPU, saves RAM when multi-shard)"

if [[ "$LIGHT_HOST" == "1" ]]; then
  "${LEGACY}/lighten_host_for_teacher_gen.sh" stop
fi

if [[ "${CLEAR_SWAP:-0}" == "1" ]] && command -v swapoff >/dev/null; then
  echo "CLEAR_SWAP=1: resetting swap..."
  sudo swapoff -a && sudo swapon -a || echo "WARN: swap reset failed" >&2
fi

if [[ "$MONITOR" == "1" ]]; then
  pkill -f "monitor_teacher_gen_resources.sh" 2>/dev/null || true
  pkill -f "watchdog_server_health.py" 2>/dev/null || true
  mkdir -p "$HEALTH_LOG_DIR"
  nohup "${LEGACY}/monitor_teacher_gen_resources.sh" 30 "${LOG_DIR}/resource_monitor.log" \
    >> "${LOG_DIR}/resource_monitor.log" 2>&1 &
  HEALTH_MIN_AVAIL_GB="${HEALTH_MIN_AVAIL_GB:-8}" \
    nohup python3 "$WATCHDOG" \
      --log-dir "$HEALTH_LOG_DIR" \
      --staging "$WAV_STAGING" \
      --wav-disk "${LOG_DIR}/wavs" \
      --interval "${HEALTH_INTERVAL:-15}" \
    >> "${HEALTH_LOG_DIR}/watchdog.stdout.log" 2>&1 &
  echo "Resource monitor -> ${LOG_DIR}/resource_monitor.log"
  echo "Health watchdog  -> ${HEALTH_LOG_DIR}/health.jsonl"
fi

echo "Starting $NUM_SHARDS openmoss servers (stagger ${STAGGER_SEC}s)..."
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  if [[ "$i" -gt 0 ]]; then
    sleep "$STAGGER_SEC"
  fi
  start_openmoss "${GPU_ARR[$i]}" "${PORT_ARR[$i]}"
  expected=$((i + 1))
  actual=$(moss_count)
  if [[ "$actual" -ne "$expected" ]]; then
    echo "ERROR: expected $expected moss-tts-server, got $actual" >&2
    exit 1
  fi
done

mkdir -p "$WAV_STAGING/v15" "${LOG_DIR}/wavs/v15"
if [[ -d "${LOG_DIR}/wavs/v15" ]]; then
  rsync -a "${LOG_DIR}/wavs/v15/" "$WAV_STAGING/v15/"
fi
(
  while true; do
    sleep "$WAV_SYNC_SEC"
    STAGING="$WAV_STAGING" DEST="${LOG_DIR}/wavs" "${LEGACY}/sync_teacher_wavs_from_staging.sh" \
      >> "${LOG_DIR}/wav_sync.log" 2>&1 || true
  done
) &
SYNC_PID=$!

mkdir -p "$LOG_DIR"
PIDS=()
CORPUS="${CORPUS:-$LOG_DIR/corpus/texts.jsonl}"
OUT_DIR="${OUT_DIR:-$LOG_DIR}"
REF_WAV="${REF_WAV:-$ROOT/data/voices/loli/loli_15s.wav}"
WAV_REL="${WAV_REL_ROOT:-training/loli_15s/wavs}"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_ARR[$i]}"
  port="${PORT_ARR[$i]}"
  api="http://127.0.0.1:${port}/tts"
  log="${LOG_DIR}/teacher_gen.shard${i}.log"
  echo "Launching shard $i gpu=$gpu -> $log"
  SPEAKER_ROOT="$ROOT" MOSS_RT_TRAIN_DIR="$LOG_DIR" PYTHONUNBUFFERED=1 \
    OPENMOSS_MAIN_GPU="$gpu" OPENMOSS_PORT="$port" \
    nohup python3 "${LEGACY}/build_realtime_finetune_dataset.py" \
      --corpus "$CORPUS" \
      --ref "$REF_WAV" \
      --out-dir "$OUT_DIR" \
      --teachers "$TEACHERS" \
      --wav-dir "$WAV_STAGING" \
      --wav-rel-root "$WAV_REL" \
      --api "$api" \
      --shard-id "$i" \
      --num-shards "$NUM_SHARDS" \
      "${EXTRA_ARGS[@]}" \
      >> "$log" 2>&1 &
  PIDS+=($!)
  sleep 2
done

echo "Teacher gen PIDs: ${PIDS[*]}"
wait "${PIDS[@]}"
kill "$SYNC_PID" 2>/dev/null || true
SYNC_PID=""
STAGING="$WAV_STAGING" DEST="${LOG_DIR}/wavs" "${LEGACY}/sync_teacher_wavs_from_staging.sh"
cat "${LOG_DIR}"/train_raw.shard*.jsonl > "${LOG_DIR}/train_raw.jsonl"
echo "Done: ${LOG_DIR}/train_raw.jsonl ($(wc -l < "${LOG_DIR}/train_raw.jsonl") rows)"
pkill -f "monitor_teacher_gen_resources.sh" 2>/dev/null || true
