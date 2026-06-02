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

# Block accidental cross-sync (batch2 staging → legacy disk or vice versa).
_staging_base="$(basename "$STAGING")"
_dest_norm="$(realpath -m "$DEST" 2>/dev/null || echo "$DEST")"
if [[ "$_staging_base" == *"batch2"* && "$_dest_norm" == *"/training/loli_15s/wavs" && "$_dest_norm" != *"batch2"* ]]; then
  echo "ERROR: refusing to sync batch2 staging ($STAGING) into legacy $DEST" >&2
  exit 1
fi
if [[ "$_staging_base" != *"batch2"* && "$_dest_norm" == *"loli_15s_batch2"* ]]; then
  echo "ERROR: refusing to sync legacy staging ($STAGING) into batch2 $DEST" >&2
  exit 1
fi

if [[ ! -d "$STAGING/v15" ]]; then
  echo "No staging dir $STAGING/v15" >&2
  exit 0
fi

mkdir -p "$DEST/v15"
rsync -a --info=stats2 "$STAGING/v15/" "$DEST/v15/"
