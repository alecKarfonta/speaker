#!/usr/bin/env bash
# Clone OpenMOSS/MOSS-TTS and install finetune dependencies for Realtime SFT.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
MOSS_DIR="${MOSS_TTS_DIR:-$ROOT/third_party/MOSS-TTS}"
PIN="${MOSS_TTS_PIN:-main}"

mkdir -p "$ROOT/third_party"
if [[ ! -d "$MOSS_DIR/.git" ]]; then
  echo "Cloning MOSS-TTS into $MOSS_DIR ..."
  git clone --depth 1 --branch "$PIN" https://github.com/OpenMOSS/MOSS-TTS.git "$MOSS_DIR"
else
  echo "MOSS-TTS already at $MOSS_DIR"
fi

echo "$PIN" > "$MOSS_DIR/.speaker_pin"

VENV="${FINETUNE_VENV:-$ROOT/.venv-finetune}"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip wheel
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e "$MOSS_DIR[torch-runtime,finetune]"

echo ""
echo "Done. Activate with: source $VENV/bin/activate"
echo "MOSS-TTS path: $MOSS_DIR"
echo "Next: python3 scripts/build_loli15s_corpus.py"
