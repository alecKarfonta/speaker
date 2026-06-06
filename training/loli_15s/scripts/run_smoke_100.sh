#!/usr/bin/env bash
# Smoke: 100 teacher clips → voice QC → listen page.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SMOKE="$ROOT/training/loli_15s_smoke"
LOLI="$ROOT/training/loli_15s"
LEGACY="$LOLI/scripts/legacy"
OPENMOSS_START="$ROOT/training/moss-realtime/scripts/legacy/start-openmoss.sh"
PORT="${OPENMOSS_PORT:-8014}"
GPU="${OPENMOSS_MAIN_GPU:-0}"
COUNT="${COUNT:-100}"
LISTEN="$SMOKE/eval/listen/smoke_100"

mkdir -p "$SMOKE/corpus" "$SMOKE/wavs/v15" "$SMOKE/qc" "$LISTEN/pass" "$LISTEN/quarantine"

echo "=== Corpus ($COUNT lines) ==="
head -n "$COUNT" "$ROOT/training/loli_15s_batch3/corpus/texts.jsonl" > "$SMOKE/corpus/texts.jsonl"
python3 "$LEGACY/enrich_loli15s_corpus_styles.py" --corpus "$SMOKE/corpus/texts.jsonl" 2>/dev/null || true

echo "=== STT check ==="
STT_API="${STT_API:-http://localhost:8603/v1/audio/transcriptions}"
curl -sf --max-time 5 "${STT_API%/v1/audio/transcriptions}/v1/models" >/dev/null

echo "=== openmoss :$PORT ==="
if ! curl -sf --max-time 3 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q ok; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
  sleep 1
  OPENMOSS_MAIN_GPU="$GPU" OPENMOSS_PORT="$PORT" OPENMOSS_MODEL_VERSION=v15 \
    SPEAKER_ROOT="$ROOT" nohup "$OPENMOSS_START" >> "/tmp/openmoss-smoke-${PORT}.log" 2>&1 &
  for _ in $(seq 1 90); do
    if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q ok; then
      echo "openmoss ready"
      break
    fi
    sleep 2
  done
fi
curl -sf "http://127.0.0.1:${PORT}/health" | head -c 120 || { echo "openmoss failed"; exit 1; }

echo "=== Generate $COUNT teacher WAVs ==="
export SPEAKER_ROOT="$ROOT"
export TEACHER_TRAIN_RAW="$LOLI/train_raw.jsonl"
export TEACHER_POOL="$LOLI/wavs/v15_pruned"
# shellcheck disable=SC1091
source "${FINETUNE_VENV:-$ROOT/.venv-finetune}/bin/activate" 2>/dev/null || true

python3 "$LEGACY/build_realtime_finetune_dataset.py" \
  --corpus "$SMOKE/corpus/texts.jsonl" \
  --ref "$ROOT/data/voices/loli/loli_15s.wav" \
  --out-dir "$SMOKE" \
  --wav-dir "$SMOKE/wavs" \
  --wav-rel-root "training/loli_15s_smoke/wavs" \
  --api "http://127.0.0.1:${PORT}/tts" \
  --no-auto-start \
  --no-stt

echo "=== Prune QC (trim + voice) ==="
mkdir -p "$SMOKE/wavs/v15_pruned" "$SMOKE/wavs/v15_quarantine"
python3 "$ROOT/training/moss-realtime/scripts/legacy/prune_loli15s_teacher_dataset.py" \
  --root "$ROOT" \
  --wav-dir "$SMOKE/wavs/v15" \
  --out-dir "$SMOKE/wavs/v15_pruned" \
  --quarantine-dir "$SMOKE/wavs/v15_quarantine" \
  --corpus "$SMOKE/corpus/texts.jsonl" \
  --qc-dir "$SMOKE/qc" \
  --apply \
  --trim-only \
  --wer-quarantine-threshold 0.75 \
  --end-buffer-ms 750 \
  --voice-qc \
  --ref-wav "$ROOT/data/voices/loli/loli_15s.wav" \
  --teacher-train-raw "$LOLI/train_raw.jsonl" \
  --teacher-pool "$LOLI/wavs/v15_pruned" \
  --min-cos-ref 0.5 \
  --min-cos-teacher 0.5 \
  --workers 2

echo "=== Listen page ==="
python3 "$SCRIPT_DIR/build_smoke_listen_page.py" --smoke-dir "$SMOKE" --out "$LISTEN"

echo ""
echo "Done."
echo "  Report:  $SMOKE/qc/prune_report.html"
echo "  Listen:  $LISTEN/index.html"
echo "  Pass:    $LISTEN/pass/"
echo "  Reject:  $LISTEN/quarantine/"
