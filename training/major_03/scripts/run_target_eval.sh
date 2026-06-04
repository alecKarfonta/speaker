#!/usr/bin/env bash
# Targeted sweep: T≈0.92 p≈0.72 k=40, story clips, 2× consistency runs per cell.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MAJOR="$ROOT/training/major_03"
PYTHON="${BENCH_PYTHON:-$ROOT/.venv-finetune/bin/python3}"
PORT="${MOSS_RT_PORT:-8016}"
API="${MOSS_RT_API:-http://127.0.0.1:${PORT}}"

echo "=== Target sweep TTS (${API}) ==="
echo "  11 presets × 4 story clips × 2 runs = 88 WAVs"
MOSS_RT_TRAIN_DIR="$MAJOR" MOSS_RT_API="$API" \
  "$PYTHON" "$SCRIPT_DIR/generate_eval_samples.py" \
  --sweep-target --wait-health 900

echo "=== ECAPA bench + scored listen page ==="
TARGET=1 NO_STT=1 "$SCRIPT_DIR/run_voice_similarity_bench.sh"

echo "=== Done ==="
echo "  Listen: file://${MAJOR}/eval/listen/epoch11_target_sweep/index.html"
echo "  Bench:  file://${MAJOR}/eval/bench/epoch11_target_sweep/report.html"
