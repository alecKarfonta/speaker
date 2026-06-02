#!/usr/bin/env bash
# Compare MOSS-Realtime RTF: LoRA vs merged weights, with optional backbone compile.
#
# Requires a free GPU. Stops any process on MOSS_RT_PORT, starts server twice, runs benchmarks.
#
# Usage:
#   MOSS_RT_GPU=3 ./scripts/benchmark_rt_merged_compile.sh
#   SKIP_LORA=1 ./scripts/benchmark_rt_merged_compile.sh   # merged only
#   SKIP_COMPILE=1 ./scripts/benchmark_rt_merged_compile.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${MOSS_RT_PORT:-8016}"
GPU="${MOSS_RT_GPU:-3}"
MERGED="${MOSS_RT_MERGED_DIR:-$ROOT/training/loli_15s/exports/loli15s-v2-noref-max-merged}"
LORA_CKPT="${MOSS_RT_LORA_CKPT:-$ROOT/training/loli_15s/checkpoints/loli15s-v2-noref-max}"
LOG_DIR="${MOSS_RT_BENCH_LOG_DIR:-$ROOT/training/loli_15s/logs}"
WAIT_SEC="${MOSS_RT_BENCH_WAIT_SEC:-240}"

PYTHON="${MOSS_RT_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/.venv-finetune/bin/python3" ]]; then
    PYTHON="$ROOT/.venv-finetune/bin/python3"
  elif [[ -x "$ROOT/.venv/bin/python3" ]]; then
    PYTHON="$ROOT/.venv/bin/python3"
  else
    PYTHON="python3"
  fi
fi

mkdir -p "$LOG_DIR"

stop_port() {
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
  fi
  sleep 2
}

wait_health() {
  local label="$1"
  echo "Waiting for $label (up to ${WAIT_SEC}s) ..."
  for ((i=0; i<WAIT_SEC; i+=5)); do
    if curl -sf "http://127.0.0.1:${PORT}/health" | grep -q '"realtime_enabled":true'; then
      curl -s "http://127.0.0.1:${PORT}/health" | "$PYTHON" -m json.tool
      return 0
    fi
    sleep 5
  done
  echo "Server did not become ready for $label"
  return 1
}

run_bench() {
  local tag="$1"
  echo ""
  echo "========== $tag: stream =========="
  "$PYTHON" "$ROOT/scripts/benchmark_rt_clean.py" 2>&1 | tee "$LOG_DIR/bench_stream_${tag}.log"
  echo ""
  echo "========== $tag: POST /tts =========="
  "$PYTHON" "$ROOT/scripts/benchmark_rt_tts.py" --url "http://127.0.0.1:${PORT}" 2>&1 | tee "$LOG_DIR/bench_tts_${tag}.log"
}

start_server() {
  local model_id="$1"
  local compile_flag="$2"
  local log="$3"
  stop_port
  export MOSS_RT_GPU="$GPU"
  export MOSS_RT_PORT="$PORT"
  export MOSS_RT_MODEL_ID="$model_id"
  export MOSS_RT_NATIVE_VOICE=true
  export MOSS_RT_CODEC_BACKEND=auto
  export MOSS_RT_ONNX_GPU=true
  export MOSS_RT_EXPERIMENTAL_COMPILE_BACKBONE="$compile_flag"
  export MOSS_RT_DEVICES=0
  export CUDA_VISIBLE_DEVICES="$GPU"
  export MOSS_ENABLE_MAIN_MODEL=false
  export MOSS_ENABLE_VOICE_GEN=false
  export MOSS_ENABLE_REALTIME=true
  export MOSS_ENABLE_STREAMING=true
  export MOSS_TTS_DIR="$ROOT/third_party/MOSS-TTS"
  export MOSS_RT_ONNX_CODEC_DIR="$ROOT/training/weights/MOSS-Audio-Tokenizer-ONNX"
  export VOICES_DIR="${VOICES_DIR:-$ROOT/data/voices}"
  export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT"
  export PORT="$PORT"
  export HOST=127.0.0.1
  export TORCHINDUCTOR_FX_GRAPH_CACHE=1
  export TORCHINDUCTOR_AUTOGRAD_CACHE=1
  nohup "$PYTHON" -m app.moss_api >"$log" 2>&1 &
  echo "Started server pid=$! model=$model_id compile=$compile_flag log=$log"
}

if [[ ! -d "$MERGED" || ! -f "$MERGED/model.safetensors" ]]; then
  echo "Merged weights missing — run: $PYTHON scripts/merge_loli_rt_lora.py"
  exit 1
fi

SUMMARY="$LOG_DIR/benchmark_merged_compile_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$SUMMARY") 2>&1
echo "Benchmark log: $SUMMARY"
echo "GPU=$GPU PORT=$PORT"

if [[ "${SKIP_LORA:-0}" != "1" ]]; then
  start_server "$LORA_CKPT" false "$LOG_DIR/moss_rt_lora.log"
  wait_health "LoRA (no compile)"
  run_bench "lora"
fi

start_server "$MERGED" false "$LOG_DIR/moss_rt_merged.log"
wait_health "merged (no compile)"
run_bench "merged"

if [[ "${SKIP_COMPILE:-0}" != "1" ]]; then
  start_server "$MERGED" true "$LOG_DIR/moss_rt_merged_compile.log"
  wait_health "merged + compile backbone"
  run_bench "merged_compile"
fi

stop_port
echo ""
echo "Done. Summary: $SUMMARY"
