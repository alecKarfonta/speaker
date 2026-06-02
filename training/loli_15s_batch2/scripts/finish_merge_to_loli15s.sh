#!/usr/bin/env bash
# After batch2 teacher gen completes: merge new WAVs into training/loli_15s and rebuild train_raw.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

cat "${ROOT}/training/loli_15s_batch2"/train_raw.shard*.jsonl > "${ROOT}/training/loli_15s_batch2/train_raw.jsonl"

"$ROOT/training/loli_15s/scripts/merge_batch2_into_loli15s.sh"
"$ROOT/training/loli_15s/scripts/assemble_loli_dataset.sh"

python3 - <<'PY'
import json
from pathlib import Path
loli = Path("/home/alec/git/speaker/training/loli_15s")
n = len(list((loli / "wavs/v15_pruned").glob("*.wav")))
rows = sum(1 for _ in (loli / "train_raw.jsonl").open() if _.strip())
print(json.dumps({"v15_pruned_wavs": n, "train_raw_rows": rows, "target_met": n >= 6000}, indent=2))
PY
