#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export SPEAKER_ROOT="$ROOT"
exec "$ROOT/training/major_03/scripts/run_major03_teacher_gen_parallel.sh" "$@"
