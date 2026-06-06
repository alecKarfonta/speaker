#!/usr/bin/env bash
# Full loli v2 data pipeline (GPU-heavy steps). Run phases individually if needed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"$SCRIPT_DIR/build_loli_batch3.sh"
"$SCRIPT_DIR/run_loli_batch3_teacher_gen.sh"
"$SCRIPT_DIR/merge_batch3_into_loli15s.sh"
"$SCRIPT_DIR/run_loli_v2_qc.sh"
"$SCRIPT_DIR/run_sft_v2.sh"
echo "Serve exports/loli15s-v2-merged with warm_092, then:"
COMPARE_API="${COMPARE_API:-}" MOSS_RT_API="${MOSS_RT_API:-http://127.0.0.1:8016}" \
  "$SCRIPT_DIR/run_eval_v2.sh"
