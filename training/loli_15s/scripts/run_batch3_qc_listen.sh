#!/usr/bin/env bash
# QC batch3 teacher WAVs → v15_pruned + browser listen page (smoke-style layout).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
B3="$ROOT/training/loli_15s_batch3"
LOLI="$ROOT/training/loli_15s"
LISTEN="${LISTEN:-$B3/eval/listen/emotion}"
PRUNE="$ROOT/training/moss-realtime/scripts/legacy/prune_loli15s_teacher_dataset.py"

mkdir -p "$B3/wavs/v15_pruned" "$B3/wavs/v15_quarantine" "$B3/qc" "$LISTEN"

STT_API="${STT_API:-http://192.168.1.196:8603/v1/audio/transcriptions}"
export STT_API

# shellcheck disable=SC1091
source "${FINETUNE_VENV:-$ROOT/.venv-finetune}/bin/activate" 2>/dev/null || true

echo "=== STT preflight ($STT_API) ==="
python3 - "$STT_API" "$B3/wavs/v15" <<'PY'
import sys
from pathlib import Path
import requests

api, wav_dir = sys.argv[1], Path(sys.argv[2])
sample = next(wav_dir.glob("*.wav"), None)
if sample is None:
    raise SystemExit("No WAVs to QC")
try:
    with sample.open("rb") as f:
        r = requests.post(
            api,
            files={"file": (sample.name, f, "audio/wav")},
            data={"model": "base", "language": "en", "response_format": "verbose_json"},
            timeout=60,
        )
    if r.status_code != 200:
        raise SystemExit(f"STT HTTP {r.status_code}: {r.text[:200]}")
    print(f"STT OK on {sample.name}: {str(r.json().get('text', ''))[:60]!r}")
except Exception as exc:
    raise SystemExit(
        f"STT not reachable at {api}: {exc}\n"
        "Start faster-whisper on :8603 before QC (see docker-compose STT_API_URL)."
    ) from exc
PY

echo "=== QC batch3 → v15_pruned (750ms buffer + last-word cutoff quarantine) ==="
python3 "$PRUNE" \
  --root "$ROOT" \
  --wav-dir "$B3/wavs/v15" \
  --out-dir "$B3/wavs/v15_pruned" \
  --quarantine-dir "$B3/wavs/v15_quarantine" \
  --corpus "$B3/corpus/texts.jsonl" \
  --qc-dir "$B3/qc" \
  --apply \
  --trim-only \
  --wer-quarantine-threshold 0.75 \
  --end-buffer-ms 750 \
  --quarantine-cutoff \
  --tail-gap-fail-s 0.30 \
  --min-missing-tail-words 2 \
  --stt-api "$STT_API" \
  --voice-qc \
  --ref-wav "$ROOT/data/voices/loli/loli_15s.wav" \
  --teacher-train-raw "$LOLI/train_raw.jsonl" \
  --teacher-pool "$LOLI/wavs/v15_pruned" \
  --min-cos-ref 0.5 \
  --min-cos-teacher 0.5 \
  --workers 4 \
  --stt-max-retries 5 \
  --defer-stt-fail \
  --no-local-whisper-fallback

echo "=== Listen page ==="
python3 "$SCRIPT_DIR/build_smoke_listen_page.py" \
  --smoke-dir "$B3" \
  --out "$LISTEN" \
  --max-each 40 \
  --emotion-only

echo ""
echo "Pruned WAVs: $B3/wavs/v15_pruned/"
echo "Listen:      $LISTEN/index.html"
echo "Report:      $B3/qc/prune_report.html"
