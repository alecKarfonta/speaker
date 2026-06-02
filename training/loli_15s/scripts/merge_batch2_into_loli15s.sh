#!/usr/bin/env bash
# Merge teacher WAVs + train_raw from loli_15s_batch2 into training/loli_15s.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
B2="$ROOT/training/loli_15s_batch2"
WAV_DIR="$LOLI/wavs/v15"
B2_WAV="$B2/wavs/v15"

if [[ ! -d "$B2_WAV" ]]; then
  echo "Missing batch2 wavs: $B2_WAV" >&2
  exit 1
fi

mkdir -p "$WAV_DIR" "$LOLI/corpus"

echo "=== Copy batch2 WAVs into $WAV_DIR (no overwrite) ==="
rsync -a --ignore-existing "$B2_WAV/" "$WAV_DIR/"
n_wav=$(find "$WAV_DIR" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ')
echo "Total WAVs in $WAV_DIR: $n_wav"

echo "=== Merge train_raw.jsonl ==="
python3 - "$LOLI" "$B2" <<'PY'
import json
import shutil
import sys
from pathlib import Path

loli = Path(sys.argv[1])
b2 = Path(sys.argv[2])
out = loli / "train_raw.jsonl"
backup = loli / "train_raw.premerge.jsonl"

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
    row["ref_wav"] = "data/voices/loli_15s/loli_15s.wav"
    return row

if out.is_file() and not backup.is_file():
    shutil.copy2(out, backup)
    print(f"Backed up → {backup}")

by_id: dict[str, dict] = {}
for row in load_rows(loli / "train_raw.jsonl"):
    by_id[row["id"]] = fix_paths(row, "training/loli_15s/wavs/v15")
for row in load_rows(b2 / "train_raw.jsonl"):
    by_id[row["id"]] = fix_paths(row, "training/loli_15s/wavs/v15")

merged = list(by_id.values())
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n")
print(f"Merged train_raw: {len(merged)} rows → {out}")
PY

echo "=== Build corpus/texts.jsonl for QC (from train_raw + batch2 styles) ==="
python3 - "$LOLI" "$B2" <<'PY'
import json
import sys
from pathlib import Path

loli = Path(sys.argv[1])
b2 = Path(sys.argv[2])
styles: dict[str, dict] = {}
b2_corpus = b2 / "corpus/texts.jsonl"
if b2_corpus.is_file():
    for line in b2_corpus.read_text().splitlines():
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
    rows.append({
        "id": cid,
        "type": base.get("type", "single"),
        "text": text,
        "style": base.get("style"),
        "instruction": base.get("instruction"),
    })
    rows[-1] = {k: v for k, v in rows[-1].items() if v is not None}

out = loli / "corpus/texts.jsonl"
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
(loli / "corpus/corpus_stats.json").write_text(
    json.dumps({"total": len(rows), "source": "merged train_raw + batch2 styles"}, indent=2) + "\n"
)
print(f"Wrote {len(rows)} corpus lines → {out}")
PY

echo "Done. Next: python training/loli_15s/scripts/distill.py qc prune"
