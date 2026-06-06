#!/usr/bin/env bash
# Merge batch3 teacher WAVs + train_raw into training/loli_15s (same pattern as batch2).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
B3="$ROOT/training/loli_15s_batch3"
WAV_DIR="$LOLI/wavs/v15"
B3_WAV="$B3/wavs/v15"

if [[ ! -d "$B3_WAV" ]]; then
  echo "Missing batch3 wavs: $B3_WAV — run run_loli_batch3_teacher_gen.sh first" >&2
  exit 1
fi

mkdir -p "$WAV_DIR" "$LOLI/corpus"

echo "=== Copy batch3 WAVs into $WAV_DIR (no overwrite) ==="
rsync -a --ignore-existing "$B3_WAV/" "$WAV_DIR/"
n_wav=$(find "$WAV_DIR" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ')
echo "Total WAVs in $WAV_DIR: $n_wav"

echo "=== Merge train_raw.jsonl ==="
python3 - "$LOLI" "$B3" <<'PY'
import json
import shutil
import sys
from pathlib import Path

loli = Path(sys.argv[1])
b3 = Path(sys.argv[2])
out = loli / "train_raw.jsonl"
backup = loli / "train_raw.pre_batch3.jsonl"

def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip().replace("\x00", "")
        if line:
            rows.append(json.loads(line))
    return rows

def fix_paths(row: dict) -> dict:
    conv = row.get("conversations") or []
    if conv and "wav" in conv[0]:
        name = Path(conv[0]["wav"]).name
        conv[0]["wav"] = f"training/loli_15s/wavs/v15/{name}"
    row["conversations"] = conv
    row["ref_wav"] = "data/voices/loli/loli_15s.wav"
    return row

if out.is_file() and not backup.is_file():
    shutil.copy2(out, backup)
    print(f"Backed up → {backup}")

by_id: dict[str, dict] = {}
for row in load_rows(loli / "train_raw.jsonl"):
    by_id[row["id"]] = fix_paths(row)
for row in load_rows(b3 / "train_raw.jsonl"):
    by_id[row["id"]] = fix_paths(row)

merged = list(by_id.values())
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n")
print(f"Merged train_raw: {len(merged)} rows → {out}")
PY

echo "=== Refresh corpus/texts.jsonl ==="
python3 - "$LOLI" "$B3" <<'PY'
import json
import sys
from pathlib import Path

# corpus_id_from_wav_name not needed; train_raw id is loli3_st_XXXXX_v15

loli = Path(sys.argv[1])
b3 = Path(sys.argv[2])
styles: dict[str, dict] = {}
for src in (b3 / "corpus/texts.jsonl", loli / "corpus/texts.jsonl"):
    if src.is_file():
        for line in src.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                styles[row["id"]] = row

rows = []
for line in (loli / "train_raw.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    tr = json.loads(line)
    cid = tr["id"].removesuffix("_v15")
    wav_name = Path(tr["conversations"][0]["wav"]).name
    text = tr["conversations"][0]["text"]
    base = styles.get(cid, {})
    row = {
        "id": cid,
        "type": base.get("type", "single"),
        "text": text,
        "style": base.get("style"),
        "instruction": base.get("instruction"),
        "gap_category": base.get("gap_category"),
        "length": base.get("length"),
        "wav_file": wav_name,
    }
    rows.append({k: v for k, v in row.items() if v is not None})

out = loli / "corpus/texts.jsonl"
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
(loli / "corpus/corpus_stats.json").write_text(
    json.dumps({"total": len(rows), "source": "merged train_raw + batch3"}, indent=2) + "\n"
)
print(f"Wrote {len(rows)} corpus lines → {out}")
PY

echo "Done. Next: training/loli_15s/scripts/run_loli_v2_qc.sh"
