#!/usr/bin/env bash
# Build final SFT dataset: pruned passes + quarantine (user-approved) + remaining raw v15.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
WAVS="$LOLI/wavs"
FINAL="$WAVS/v15_pruned"
PREMERGE="$WAVS/v15_pruned.premerge"
QUAR="$WAVS/v15_quarantine"
RAW="$WAVS/v15"

mkdir -p "$FINAL"

echo "=== Assemble $FINAL ==="
if [[ -d "$PREMERGE" ]]; then
  echo "  + premerge ($(find "$PREMERGE" -maxdepth 1 -name '*.wav' | wc -l) wavs)"
  rsync -a "$PREMERGE/" "$FINAL/"
fi
if [[ -d "$QUAR" ]]; then
  echo "  + quarantine ($(find "$QUAR" -maxdepth 1 -name '*.wav' | wc -l) wavs)"
  rsync -a "$QUAR/" "$FINAL/"
fi
# Batch2 / never-pruned clips still in raw v15
echo "  + raw v15 (only names not already in final)"
rsync -a --ignore-existing "$RAW/" "$FINAL/"

n_final=$(find "$FINAL" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ')
echo "Final WAV count: $n_final"

echo "=== Filter train_raw.jsonl to existing WAVs ==="
python3 - "$LOLI" <<'PY'
import json
import sys
from pathlib import Path

loli = Path(sys.argv[1])
final = loli / "wavs/v15_pruned"
train_in = loli / "train_raw.jsonl"
train_out = loli / "train_raw.jsonl"
backup = loli / "train_raw.pre_assemble.jsonl"

# Prefer freshly merged train_raw (post batch2 merge); stale pre_assemble loses rows.
source = train_in
if backup.is_file() and train_in.is_file():
    if backup.stat().st_size > train_in.stat().st_size:
        source = backup
if source == train_in and train_in.is_file() and not backup.is_file():
    backup.write_bytes(train_in.read_bytes())

have = {p.stem for p in final.glob("*.wav")}


def row_wav_names(row: dict) -> list[str]:
    names = []
    for turn in row.get("conversations") or []:
        if turn.get("role") == "assistant" and turn.get("wav"):
            names.append(Path(turn["wav"]).name)
    return names


rows = []
dropped = 0
for line in source.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    names = row_wav_names(row)
    if not names or not all(Path(n).stem in have for n in names):
        dropped += 1
        continue
    for turn in row.get("conversations") or []:
        if turn.get("role") == "assistant" and turn.get("wav"):
            turn["wav"] = f"training/loli_15s/wavs/v15_pruned/{Path(turn['wav']).name}"
    row["ref_wav"] = "data/voices/loli_15s/loli_15s.wav"
    rows.append(row)

train_out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
stats = {
    "wavs_on_disk": len(have),
    "train_raw_rows": len(rows),
    "dropped_no_wav": dropped,
    "out": str(train_out),
}
(loli / "dataset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
print(json.dumps(stats, indent=2))
PY

echo "Done. Use wavs/v15_pruned + train_raw.jsonl for preprocess/SFT."
