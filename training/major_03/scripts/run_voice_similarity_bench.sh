#!/usr/bin/env bash
# ECAPA speaker-similarity bench for major_03 eval WAVs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MAJOR="$ROOT/training/major_03"
PYTHON="${BENCH_PYTHON:-$ROOT/.venv-finetune/bin/python3}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON — set BENCH_PYTHON or create .venv-finetune" >&2
  exit 1
fi

if ! "$PYTHON" -c "import speechbrain" 2>/dev/null; then
  echo "Installing bench deps into $(dirname "$PYTHON")…"
  "$PYTHON" -m pip install -q -r "$MAJOR/requirements-bench.txt"
fi

if [[ "${TARGET:-0}" == "1" ]]; then
  GEN_DIR="${GEN_DIR:-eval/listen/epoch11_target_sweep}"
  OUT_DIR="${OUT_DIR:-eval/bench/epoch11_target_sweep}"
elif [[ "${HOT_ZONE:-0}" == "1" ]]; then
  GEN_DIR="${GEN_DIR:-eval/listen/epoch11_hot_zone_sweep}"
  OUT_DIR="${OUT_DIR:-eval/bench/epoch11_hot_zone_sweep}"
elif [[ "${SWEEP:-0}" == "1" ]]; then
  GEN_DIR="${GEN_DIR:-eval/listen/epoch11_sampling_sweep}"
  OUT_DIR="${OUT_DIR:-eval/bench/epoch11_sampling_sweep}"
else
  GEN_DIR="${GEN_DIR:-eval/listen/epoch11_native}"
  OUT_DIR="${OUT_DIR:-eval/bench/epoch11_native}"
fi
EXTRA=()
[[ "${NO_STT:-0}" == "1" ]] && EXTRA+=(--no-stt)

exec "$PYTHON" "$SCRIPT_DIR/bench_voice_similarity.py" \
  --gen-dir "$MAJOR/$GEN_DIR" \
  --out-dir "$MAJOR/$OUT_DIR" \
  "${EXTRA[@]}" \
  "$@"
