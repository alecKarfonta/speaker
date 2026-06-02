#!/usr/bin/env bash
# Merge teacher WAVs + train_raw from loli_15s_gap3 into training/loli_15s.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
G3="$ROOT/training/loli_15s_gap3"
WAV_DIR="$LOLI/wavs/v15"
G3_WAV="$G3/wavs/v15"

if [[ ! -d "$G3_WAV" ]]; then
  echo "Missing gap3 wavs: $G3_WAV" >&2
  exit 1
fi

mkdir -p "$WAV_DIR" "$LOLI/corpus"

echo "=== Copy gap3 WAVs into $WAV_DIR (no overwrite) ==="
rsync -a --ignore-existing "$G3_WAV/" "$WAV_DIR/"
n_wav=$(find "$WAV_DIR" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ')
echo "Total WAVs in $WAV_DIR: $n_wav"

echo "=== Merge train_raw.jsonl ==="
python3 - "$LOLI" "$G3" <<'PY'
import json
import shutil
import sys
from pathlib import Path

loli = Path(sys.argv[1])
g3 = Path(sys.argv[2])
out = loli / "train_raw.jsonl"
backup = loli / "train_raw.pre_gap3_merge.jsonl"


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip().replace("\x00", "")
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def fix_paths(row: dict, wav_prefix: str) -> dict:
    conv = row.get("conversations") or []
    if conv and "wav" in conv[0]:
        name = Path(conv[0]["wav"]).name
        conv[0]["wav"] = f"{wav_prefix}/{name}"
    row["conversations"] = conv
    if "ref_wav" in row:
        row["ref_wav"] = "data/voices/loli_15s/loli_15s.wav"
    return row


if out.is_file() and not backup.is_file():
    shutil.copy2(out, backup)
    print(f"Backed up → {backup}")

by_id: dict[str, dict] = {}
for row in load_rows(loli / "train_raw.jsonl"):
    by_id[row["id"]] = fix_paths(row, "training/loli_15s/wavs/v15")
for row in load_rows(g3 / "train_raw.jsonl"):
    by_id[row["id"]] = fix_paths(row, "training/loli_15s/wavs/v15")

merged = list(by_id.values())
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n")
print(f"Merged train_raw: {len(merged)} rows → {out}")
PY

echo "=== Refresh corpus/texts.jsonl (gap3 styles) ==="
python3 - "$LOLI" "$G3" <<'PY'
import json
import sys
from pathlib import Path

loli = Path(sys.argv[1])
g3 = Path(sys.argv[2])
styles: dict[str, dict] = {}
for corpus_path in (g3 / "corpus/texts.jsonl", loli / "corpus/texts.jsonl"):
    if not corpus_path.is_file():
        continue
    for line in corpus_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            styles[row["id"]] = row

rows = []
for line in (loli / "train_raw.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    tr = json.loads(line)
    cid = tr["id"].removesuffix("_v15")
    text = tr["conversations"][0]["text"]
    base = styles.get(cid, {})
    row = {
        "id": cid,
        "type": base.get("type", "single"),
        "text": text,
        "gap_category": base.get("gap_category"),
        "style": base.get("style"),
        "instruction": base.get("instruction"),
    }
    rows.append({k: v for k, v in row.items() if v is not None})

out = loli / "corpus/texts.jsonl"
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
(loli / "corpus/corpus_stats.json").write_text(
    json.dumps({"total": len(rows), "source": "merged train_raw + gap3 styles"}, indent=2) + "\n"
)
print(f"Wrote {len(rows)} corpus lines → {out}")
PY

echo "Done. Next: filter_single_turn → qc trim → preprocess → SFT"
