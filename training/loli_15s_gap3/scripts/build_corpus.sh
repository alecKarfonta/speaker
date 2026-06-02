#!/usr/bin/env bash
# Build gap-targeted corpus (g3_st_* IDs, deduped against loli + batch2).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CORPUS_DIR="${CORPUS_DIR:-$ROOT/training/loli_15s_gap3/corpus}"
OUT="${OUT:-$CORPUS_DIR/texts.jsonl}"
TOTAL="${TOTAL:-2500}"
SEED="${SEED:-20260602}"
MIX="${MIX:-long:0.40,numbers:0.15,names:0.15,questions:0.15,emotion:0.15}"

mkdir -p "$CORPUS_DIR"
cd "$ROOT"

python3 "$SCRIPT_DIR/analyze_coverage_gaps.py"

python3 "$SCRIPT_DIR/build_gap_corpus.py" \
  --out "$OUT" \
  --total "$TOTAL" \
  --seed "$SEED" \
  --mix "$MIX"

wc -l "$OUT"
cat "$CORPUS_DIR/corpus_stats.json"
