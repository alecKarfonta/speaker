#!/usr/bin/env bash
# Corpus lines that do not yet have a WAV under wavs/v15/.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOG_DIR="$ROOT/training/loli_15s_batch2"
CORPUS="${CORPUS:-$LOG_DIR/corpus/texts.jsonl}"
OUT="${OUT:-$LOG_DIR/corpus/texts_missing.jsonl}"
WAV_DIR="$LOG_DIR/wavs/v15"

python3 - "$CORPUS" "$WAV_DIR" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

corpus, wav_dir, out = map(Path, sys.argv[1:4])
seen_ids: set[str] = set()
rows = []
corpus_lines = 0
for line in corpus.read_text().splitlines():
    if not line.strip():
        continue
    corpus_lines += 1
    row = json.loads(line)
    rid = row["id"]
    if rid in seen_ids:
        continue
    seen_ids.add(rid)
    wav = wav_dir / f"{rid}.wav"
    if not wav.is_file() or wav.stat().st_size < 1024:
        rows.append(row)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""))
print(
    json.dumps(
        {
            "corpus_lines": corpus_lines,
            "corpus_unique_ids": len(seen_ids),
            "missing_unique": len(rows),
            "out": str(out),
        },
        indent=2,
    )
)
PY
