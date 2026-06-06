#!/usr/bin/env bash
# Build loli_15s_batch3 corpus (~2k emotion-heavy lines) + enrich styles.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
B3="$ROOT/training/loli_15s_batch3"
COUNT="${COUNT:-3000}"

cd "$ROOT"
python3 "$SCRIPT_DIR/build_loli_batch3_corpus.py" --count "$COUNT"

if [[ -f "$SCRIPT_DIR/legacy/enrich_loli15s_corpus_styles.py" ]]; then
  python3 "$SCRIPT_DIR/legacy/enrich_loli15s_corpus_styles.py" \
    --corpus "$B3/corpus/texts.jsonl"
fi

echo "Batch3 corpus ready: $B3/corpus/texts.jsonl"
