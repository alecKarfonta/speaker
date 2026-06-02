#!/usr/bin/env bash
# Copy WAVs from tmpfs staging to durable disk (rsync, incremental).
#
# Usage:
#   ./scripts/sync_teacher_wavs_from_staging.sh
#   STAGING=/dev/shm/loli15s_wavs DEST=training/loli_15s/wavs ./scripts/sync_teacher_wavs_from_staging.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="${STAGING:-/dev/shm/loli15s_wavs}"
DEST="${DEST:-$ROOT/training/loli_15s/wavs}"

if [[ ! -d "$STAGING/v15" ]]; then
  echo "No staging dir $STAGING/v15" >&2
  exit 0
fi

mkdir -p "$DEST/v15"
rsync -a --info=stats2 "$STAGING/v15/" "$DEST/v15/"
