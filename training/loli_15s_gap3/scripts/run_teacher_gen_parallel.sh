#!/usr/bin/env bash
# Parallel MOSS v1.5 teacher WAV generation for loli_15s gap3 (~2500 targeted clips).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LEGACY="${LEGACY_DIR:-$ROOT/training/moss-realtime/scripts/legacy}"
cd "$ROOT"

NUM_SHARDS="${NUM_SHARDS:-4}"
GPUS="${GPUS:-0,1,2,3}"
PORTS="${PORTS:-8014,8015,8016,8017}"
TEACHERS="${TEACHERS:-v15}"
LOG_DIR="${LOG_DIR:-$ROOT/training/loli_15s_gap3}"
REF="${REF:-$ROOT/data/voices/loli_15s/loli_15s.wav}"
CORPUS="${CORPUS:-$LOG_DIR/corpus/texts.jsonl}"
STAGGER_SEC="${STAGGER_SEC:-45}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-16000}"
WAV_STAGING="${WAV_STAGING:-/dev/shm/loli15s_gap3_wavs}"
WAV_SYNC_SEC="${WAV_SYNC_SEC:-120}"
LIGHT_HOST="${LIGHT_HOST:-1}"
EXTRA_ARGS=(--skip-existing --no-stt --no-auto-start)
TEACHER_GEN_EXTRA_ARGS="${TEACHER_GEN_EXTRA_ARGS:---qc-lenient}"
if [[ -n "$TEACHER_GEN_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS+=($TEACHER_GEN_EXTRA_ARGS)
fi
LIMIT="${LIMIT:-0}"
if [[ "$LIMIT" -gt 0 ]]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi
MONITOR="${MONITOR:-1}"
SYNC_PID=""

export OPENMOSS_MAX_SEC="${OPENMOSS_MAX_SEC:-32}"
export OPENMOSS_MAX_CODEC_TOKENS="${OPENMOSS_MAX_CODEC_TOKENS:-450}"
export OPENMOSS_CHARS_PER_SEC="${OPENMOSS_CHARS_PER_SEC:-12.5}"
export OPENMOSS_DUR_SLACK="${OPENMOSS_DUR_SLACK:-1.15}"
export TEACHER_STYLES_MODULE=v15_teacher_styles

MIN_DUR="${MIN_DUR:-5.0}"
MAX_DUR="${MAX_DUR:-32.0}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
IFS=',' read -r -a PORT_ARR <<< "$PORTS"
if [[ "${#GPU_ARR[@]}" -ne "$NUM_SHARDS" || "${#PORT_ARR[@]}" -ne "$NUM_SHARDS" ]]; then
  echo "GPUS and PORTS must each have NUM_SHARDS=$NUM_SHARDS entries" >&2
  exit 1
fi

if [[ ! -f "$CORPUS" ]]; then
  echo "Missing corpus: $CORPUS — run: $SCRIPT_DIR/build_corpus.sh" >&2
  exit 1
fi
if [[ ! -f "$REF" ]]; then
  echo "Missing reference: $REF" >&2
  exit 1
fi

_legacy_loli="$ROOT/training/loli_15s"
if [[ "$LOG_DIR" == "$_legacy_loli" ]]; then
  echo "ERROR: LOG_DIR must not be training/loli_15s" >&2
  exit 1
fi
if [[ "$LOG_DIR" != *"loli_15s_gap3"* ]]; then
  echo "ERROR: LOG_DIR must be under training/loli_15s_gap3 (got $LOG_DIR)" >&2
  exit 1
fi
if [[ "$WAV_STAGING" == *"loli15s_wavs"* && "$WAV_STAGING" != *"gap3"* ]]; then
  echo "ERROR: use WAV_STAGING=/dev/shm/loli15s_gap3_wavs" >&2
  exit 1
fi
if [[ -d "$_legacy_loli/wavs/v15" ]] && [[ "$(realpath "$LOG_DIR/wavs" 2>/dev/null)" == "$(realpath "$_legacy_loli/wavs" 2>/dev/null)" ]]; then
  echo "ERROR: LOG_DIR/wavs resolves to legacy training/loli_15s/wavs" >&2
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
  OPENMOSS_MAIN_GPU="$gpu" OPENMOSS_PORT="$port" OPENMOSS_AUX_CPU=1 \
    OPENMOSS_MODEL_VERSION=v15 SPEAKER_ROOT="$ROOT" nohup "${LEGACY}/start-openmoss.sh" \
    >> "/tmp/openmoss-server-${port}.log" 2>&1 &
  wait_openmoss "$port"
}

stop_all() {
  pkill -f '[b]uild_realtime_finetune_dataset.py.*loli_15s_gap3' 2>/dev/null || true
  pkill -f '[m]onitor_teacher_gen_resources.sh.*loli_15s_gap3' 2>/dev/null || true
  PORTS="${PORTS}" SPEAKER_ROOT="$ROOT" "${LEGACY}/teardown_openmoss.sh"
}

cleanup() {
  if [[ -n "${SYNC_PID:-}" ]] && kill -0 "$SYNC_PID" 2>/dev/null; then
    kill "$SYNC_PID" 2>/dev/null || true
  fi
  if [[ "${LIGHT_HOST:-0}" == "1" ]] && [[ -x "${LEGACY}/lighten_host_for_teacher_gen.sh" ]]; then
    "${LEGACY}/lighten_host_for_teacher_gen.sh" restore 2>/dev/null || true
  fi
  STAGING="$WAV_STAGING" DEST="${LOG_DIR}/wavs" "${LEGACY}/sync_teacher_wavs_from_staging.sh" 2>/dev/null || true
}
trap cleanup EXIT

echo "Stopping prior loli_15s_gap3 teacher-gen workers..."
stop_all

if [[ "$LIGHT_HOST" == "1" ]] && [[ -x "${LEGACY}/lighten_host_for_teacher_gen.sh" ]]; then
  "${LEGACY}/lighten_host_for_teacher_gen.sh" stop
fi

if [[ "$MONITOR" == "1" ]]; then
  pkill -f "monitor_teacher_gen_resources.sh.*loli_15s_gap3" 2>/dev/null || true
  nohup "${LEGACY}/monitor_teacher_gen_resources.sh" 30 "${LOG_DIR}/resource_monitor.log" \
    >> "${LOG_DIR}/resource_monitor.log" 2>&1 &
fi

echo "Corpus: $CORPUS  limit=${LIMIT:-0}  ref=$REF"
echo "Starting $NUM_SHARDS openmoss servers (stagger ${STAGGER_SEC}s)..."
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  if [[ "$i" -gt 0 ]]; then
    sleep "$STAGGER_SEC"
  fi
  start_openmoss "${GPU_ARR[$i]}" "${PORT_ARR[$i]}"
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
WAV_REL="training/loli_15s_gap3/wavs"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_ARR[$i]}"
  port="${PORT_ARR[$i]}"
  api="http://127.0.0.1:${port}/tts"
  log="${LOG_DIR}/teacher_gen.shard${i}.log"
  echo "Launching shard $i gpu=$gpu port=$port -> $log"
  SPEAKER_ROOT="$ROOT" MOSS_RT_TRAIN_DIR="$LOG_DIR" PYTHONUNBUFFERED=1 \
    nohup python3 "${LEGACY}/build_realtime_finetune_dataset.py" \
      --corpus "$CORPUS" \
      --ref "$REF" \
      --out-dir "$LOG_DIR" \
      --teachers "$TEACHERS" \
      --teacher-styles v15_teacher_styles \
      --wav-dir "$WAV_STAGING" \
      --wav-rel-root "$WAV_REL" \
      --api "$api" \
      --min-dur "$MIN_DUR" \
      --max-dur "$MAX_DUR" \
      --shard-id "$i" \
      --num-shards "$NUM_SHARDS" \
      "${EXTRA_ARGS[@]}" \
      >> "$log" 2>&1 &
  PIDS+=($!)
  sleep 2
done

echo "Teacher gen PIDs: ${PIDS[*]}"
echo "Monitor: tail -f ${LOG_DIR}/teacher_gen.shard0.log"
wait "${PIDS[@]}"
kill "$SYNC_PID" 2>/dev/null || true
SYNC_PID=""
STAGING="$WAV_STAGING" DEST="${LOG_DIR}/wavs" "${LEGACY}/sync_teacher_wavs_from_staging.sh"
cat "${LOG_DIR}"/train_raw.shard*.jsonl > "${LOG_DIR}/train_raw.jsonl"
echo "Done: ${LOG_DIR}/train_raw.jsonl ($(wc -l < "${LOG_DIR}/train_raw.jsonl") rows)"
pkill -f "monitor_teacher_gen_resources.sh" 2>/dev/null || true
