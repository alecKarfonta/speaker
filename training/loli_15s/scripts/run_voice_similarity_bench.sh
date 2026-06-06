#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="${MOSS_RT_TRAIN_DIR:-$ROOT/training/loli_15s}"

export MOSS_RT_TRAIN_DIR="$LOLI"
export SPEAKER_ROOT="$ROOT"
export GEN_DIR="${GEN_DIR:-$LOLI/eval/listen/v2_emotion}"
export OUT_DIR="${OUT_DIR:-$LOLI/eval/bench/v2_emotion_ecapa}"
export REF_WAV="${REF_WAV:-$ROOT/data/voices/loli/loli_15s.wav}"

# shellcheck disable=SC1091
source "${FINETUNE_VENV:-$ROOT/.venv-finetune}/bin/activate"
cd "$ROOT"
python3 "$SCRIPT_DIR/bench_voice_similarity.py" "$@"
