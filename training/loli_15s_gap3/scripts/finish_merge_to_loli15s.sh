#!/usr/bin/env bash
# After gap3 teacher gen: merge WAVs + train_raw into training/loli_15s.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
G3="$ROOT/training/loli_15s_gap3"

cat "${G3}"/train_raw.shard*.jsonl > "${G3}/train_raw.jsonl" 2>/dev/null || true

"$ROOT/training/loli_15s/scripts/merge_gap3_into_loli15s.sh"
"$ROOT/training/loli_15s/scripts/assemble_loli_dataset.sh"

python3 - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

loli = Path(sys.argv[1])
pruned = loli / "wavs/v15_pruned"
n_wav = len(list(pruned.glob("*.wav"))) if pruned.is_dir() else 0
rows = sum(1 for _ in (loli / "train_raw.jsonl").open() if _.strip())
print(json.dumps({
    "v15_pruned_wavs": n_wav,
    "train_raw_rows": rows,
    "next": [
        "filter_single_turn_train_raw.sh",
        "distill.py qc trim --trim-only",
        "preprocess + SFT v2",
    ],
}, indent=2))
PY
