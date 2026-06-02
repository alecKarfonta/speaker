#!/usr/bin/env bash
# Supplement corpus if needed, then 2-GPU missing-only teacher gen toward 6k+ loli clips.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
B2="$ROOT/training/loli_15s_batch2"

merged=$(find "$LOLI/wavs/v15_pruned" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
merged="${merged:-0}"
target="${TARGET_WAVS:-6000}"
need=$((target - merged))
echo "Merged loli clips: $merged  target: $target  need: $need (approx, before merge)"

if [[ "$need" -gt 0 && ! -f "$B2/corpus/texts_supplement.jsonl" ]]; then
  supp_count=$((need + 250))
  if [[ "$supp_count" -lt 150 ]]; then
    supp_count=150
  fi
  echo "Appending up to $supp_count supplemental corpus lines (once)..."
  COUNT="$supp_count" "$SCRIPT_DIR/build_supplement_corpus.sh"
fi

exec "$SCRIPT_DIR/run_missing_safe.sh"
