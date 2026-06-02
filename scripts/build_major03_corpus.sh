#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export SPEAKER_ROOT="$ROOT"
exec python3 "$ROOT/training/major_03/scripts/build_major03_corpus.py" "$@"
