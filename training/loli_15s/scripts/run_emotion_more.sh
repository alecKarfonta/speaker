#!/usr/bin/env bash
# Finish batch3 + append 1k emotion lines + resume 3× GPU gen.
# Output after QC: training/loli_15s_batch3/wavs/v15_pruned/
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

export SPEAKER_ROOT="$ROOT"
export SKIP_CORPUS=1
export FRESH_WAVS=0
export LIGHT_HOST=0
export CLEAR_SWAP=0
export VOICE_QC=0
export GPUS="${GPUS:-0,1,2}"
export PORTS="${PORTS:-8014,8015,8016}"
export NUM_SHARDS="${NUM_SHARDS:-3}"
export MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-14000}"

EMOTION_MORE="${EMOTION_MORE:-1000}"
if [[ "${APPEND_EMOTION:-1}" == "1" ]]; then
  echo "=== Append $EMOTION_MORE emotion lines to batch3 corpus ==="
  python3 "$SCRIPT_DIR/build_emotion_supplement.py" --count "$EMOTION_MORE"
  python3 "$ROOT/training/loli_15s/scripts/legacy/enrich_loli15s_corpus_styles.py" \
    --corpus "$ROOT/training/loli_15s_batch3/corpus/texts.jsonl" || true
fi

echo "=== Resume teacher gen (skip-existing) ==="
bash "$SCRIPT_DIR/run_loli_batch3_3k.sh"

echo "=== QC + listen page ==="
bash "$SCRIPT_DIR/run_batch3_qc_listen.sh"
