#!/usr/bin/env bash
# Recover clips lost to STT outages; import batch3 emotion supplement; rebuild train_raw.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
B3="$ROOT/training/loli_15s_batch3"
PRUNE="$ROOT/training/moss-realtime/scripts/legacy/prune_loli15s_teacher_dataset.py"
STT_API="${STT_API:-http://192.168.1.196:8603/v1/audio/transcriptions}"
PYTHON="${PYTHON:-$ROOT/.venv-finetune/bin/python3}"
[[ -x "$PYTHON" ]] || PYTHON=python3
QC_WORKERS="${QC_WORKERS:-8}"
MANIFEST="${MANIFEST:-$LOLI/qc/prune_manifest.jsonl}"
LOG="$LOLI/logs/recover_qc_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOLI/logs" "$LOLI/wavs/v15_pruned"
exec > >(tee -a "$LOG") 2>&1
echo "=== recover_loli_qc @ $(date -Is) ==="

echo "=== 1) Import batch3 pre-QC emotion supplement (63000+) ==="
if [[ -d "$B3/wavs/v15_pruned" ]]; then
  n=0
  for f in "$B3/wavs/v15_pruned"/loli3_st_63*.wav; do
    [[ -f "$f" ]] || continue
    base="$(basename "$f")"
    dest="$LOLI/wavs/v15_pruned/$base"
    if [[ ! -f "$dest" ]]; then
      cp -a "$f" "$dest"
      n=$((n + 1))
    fi
    rm -f "$LOLI/wavs/v15_quarantine/$base"
  done
  echo "Imported $n batch3 63000+ pruned WAVs"
fi

STT_OK=0
if [[ "${SKIP_STT_RETRY:-0}" != "1" ]]; then
  echo "=== 2) STT preflight ==="
  if "$PYTHON" - "$STT_API" "$LOLI/wavs/v15" <<'PY'
import sys
from pathlib import Path
import requests
api, d = sys.argv[1], Path(sys.argv[2])
s = next(d.glob("*.wav"))
with s.open("rb") as f:
    r = requests.post(api, files={"file": (s.name, f, "audio/wav")},
        data={"model": "base", "language": "en", "response_format": "verbose_json"}, timeout=120)
if r.status_code != 200:
    raise SystemExit(f"STT HTTP {r.status_code}")
print("STT OK")
PY
  then
    STT_OK=1
  else
    echo "WARN: STT unavailable — skipping retry pass (set SKIP_STT_RETRY=1 to silence)"
  fi
else
  echo "=== 2) STT retry skipped (SKIP_STT_RETRY=1) ==="
fi

echo "=== 3) Build retry manifest (stt_failed, not already pruned) ==="
RETRY_MANIFEST="$LOLI/qc/retry_stt_manifest.jsonl"
"$PYTHON" - "$MANIFEST" "$LOLI/wavs/v15_pruned" "$RETRY_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
src, pruned, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
have = {p.name for p in pruned.glob("*.wav")}
rows = []
for line in src.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("wav") in have:
        continue
    if any(str(x).startswith("stt_failed") for x in r.get("reasons") or []):
        rows.append(line)
out.write_text("\n".join(rows) + ("\n" if rows else ""))
print(f"retry candidates: {len(rows)} -> {out}")
PY

if [[ "$STT_OK" == "1" && -s "$RETRY_MANIFEST" ]]; then
  echo "=== 4) Re-QC STT failures (retries=8, no local fallback) ==="
  POOL="$LOLI/wavs/v15_pruned.pre_reqc"
  pool_args=()
  [[ -d "$POOL" ]] && pool_args=(--teacher-pool "$POOL")
  "$PYTHON" "$PRUNE" \
    --root "$ROOT" \
    --wav-dir "$LOLI/wavs/v15" \
    --out-dir "$LOLI/wavs/v15_pruned" \
    --quarantine-dir "$LOLI/wavs/v15_quarantine" \
    --corpus "$LOLI/corpus/texts.jsonl" \
    --qc-dir "$LOLI/qc" \
    --apply \
    --trim-only \
    --wer-quarantine-threshold 0.75 \
    --end-buffer-ms 750 \
    --quarantine-cutoff \
    --tail-gap-fail-s 0.30 \
    --min-missing-tail-words 2 \
    --stt-api "$STT_API" \
    --stt-max-retries 8 \
    --no-local-whisper-fallback \
    --retry-stt-from-manifest "$RETRY_MANIFEST" \
    --voice-qc \
    --ref-wav "$ROOT/data/voices/loli/loli_15s.wav" \
    --teacher-train-raw "$LOLI/train_raw.pre_reqc.jsonl" \
    "${pool_args[@]}" \
    --min-cos-ref 0.5 \
    --min-cos-teacher 0.5 \
    --workers "$QC_WORKERS"
elif [[ -s "$RETRY_MANIFEST" ]]; then
  echo "=== 4) STT retry deferred ($(wc -l < "$RETRY_MANIFEST") candidates) — start STT and re-run ==="
else
  echo "=== 4) No STT retry candidates ==="
fi

echo "=== 4b) Drop quarantine copies superseded by pruned ==="
"$PYTHON" - "$LOLI" <<'PY'
from pathlib import Path
import sys
loli = Path(sys.argv[1])
pruned = {p.name for p in (loli / "wavs/v15_pruned").glob("*.wav")}
quar = loli / "wavs/v15_quarantine"
n = 0
for p in quar.glob("*.wav"):
    if p.name in pruned:
        p.unlink()
        n += 1
print(f"removed {n} stale quarantine dupes")
PY

echo "=== 5) Rebuild train_raw from pre_reqc + batch3 supplement + pruned WAVs ==="
"$PYTHON" - "$LOLI" "$B3" <<'PY'
import json
import sys
from pathlib import Path

loli, b3 = Path(sys.argv[1]), Path(sys.argv[2])
pruned = {p.name for p in (loli / "wavs/v15_pruned").glob("*.wav")}
quar = {p.name for p in (loli / "wavs/v15_quarantine").glob("*.wav")}
pruned_ok = pruned - quar
pruned_prefix = "training/loli_15s/wavs/v15_pruned"

def assistant_wavs(row):
    return [
        Path(t["wav"]).name
        for t in row.get("conversations") or []
        if t.get("role") == "assistant" and t.get("wav")
    ]

def normalize_row(row):
    for t in row.get("conversations") or []:
        if t.get("role") == "assistant" and t.get("wav"):
            t["wav"] = f"{pruned_prefix}/{Path(t['wav']).name}"
    row["ref_wav"] = "data/voices/loli/loli_15s.wav"
    return row

rows_by_id: dict[str, dict] = {}

def score_row(row) -> tuple[int, int]:
    names = assistant_wavs(row)
    if not names or not all(n in pruned_ok for n in names):
        return (-1, 0)
    return (len(names), sum(1 for n in names if n in pruned_ok))

def ingest(path):
    added = 0
    if not path.is_file():
        return added
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rid = row["id"]
        s = score_row(row)
        if s[0] < 0:
            continue
        prev = rows_by_id.get(rid)
        if prev is None or s > score_row(prev):
            rows_by_id[rid] = row
            added += 1
    return added

source = loli / "train_raw.pre_reqc.jsonl"
if not source.is_file():
    source = loli / "train_raw.jsonl"
n_pre = ingest(source)
n_b3 = ingest(b3 / "train_raw.jsonl")
rows = [normalize_row(r) for r in rows_by_id.values()]
covered_wavs = set()
for row in rows:
    covered_wavs.update(assistant_wavs(row))

out = loli / "train_raw.jsonl"
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
emo = sum(
    1 for r in rows
    if any("__emotion__" in (t.get("wav") or "") for t in r.get("conversations") or [] if t.get("role") == "assistant")
)
missing_pruned = sorted(pruned_ok - covered_wavs)
print(json.dumps({
    "train_raw_rows": len(rows),
    "from_pre_reqc": n_pre,
    "from_batch3": n_b3,
    "pruned_wavs": len(pruned_ok),
    "pruned_without_train_row": len(missing_pruned),
    "emotion_rows": emo,
}, indent=2))
PY

if [[ -x "$SCRIPT_DIR/filter_single_turn_train_raw.sh" ]]; then
  echo "=== 6) noref jsonl ==="
  "$SCRIPT_DIR/filter_single_turn_train_raw.sh"
fi

n_pr=$(find "$LOLI/wavs/v15_pruned" -name '*.wav' | wc -l)
n_emo=$(find "$LOLI/wavs/v15_pruned" -name '*__emotion__*.wav' | wc -l)
echo "=== Done: pruned=$n_pr emotion=$n_emo ==="
echo "Log: $LOG"
