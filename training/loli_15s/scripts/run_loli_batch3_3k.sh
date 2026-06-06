#!/usr/bin/env bash
# Build 3k emotion-heavy corpus + 4× GPU teacher gen (descriptive WAV names).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
B3="$ROOT/training/loli_15s_batch3"
LOLI_LEGACY="$ROOT/training/loli_15s/scripts/legacy"
COUNT="${COUNT:-3000}"
LOG="$B3/logs/batch3_3k_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$B3/logs" "$B3/wavs/v15"

exec > >(tee -a "$LOG") 2>&1
echo "=== loli batch3 × $COUNT @ $(date -Is) ==="
echo "Log: $LOG"
echo "WAV names: {id}__{gap}__{style}__{length}.wav  e.g. loli3_st_60000__emotion__excited__short.wav"

export SPEAKER_ROOT="$ROOT"
export MOSS_RT_TRAIN_DIR="$B3"
export LOG_DIR="$B3"
export CORPUS="$B3/corpus/texts.jsonl"
export OUT_DIR="$B3"
export WAV_REL_ROOT="training/loli_15s_batch3/wavs"
export REF_WAV="$ROOT/data/voices/loli/loli_15s.wav"
export TEACHER_TRAIN_RAW="$ROOT/training/loli_15s/train_raw.jsonl"
export TEACHER_POOL="$ROOT/training/loli_15s/wavs/v15_pruned"
# ECAPA at gen is slow on 3k; run voice QC after merge via run_loli_v2_qc.sh
export VOICE_QC="${VOICE_QC:-0}"
# Safer default 2× GPU; 3-way only if MemAvailable is high: GPUS=0,1,2 NUM_SHARDS=3
export GPUS="${GPUS:-0,1}"
export PORTS="${PORTS:-8014,8015}"
export NUM_SHARDS="${NUM_SHARDS:-2}"
export MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-14000}"
export MIN_AVAIL_GB="${MIN_AVAIL_GB:-12}"
export OPENMOSS_AUX_CPU="${OPENMOSS_AUX_CPU:-0}"
export HEALTH_LOG_DIR="$B3/logs/health"

if [[ "${FRESH_WAVS:-1}" == "1" ]]; then
  echo "=== Clear old WAVs (new descriptive filenames) ==="
  rm -f "$B3/wavs/v15"/*.wav "$B3/wavs/v15"/*.bak 2>/dev/null || true
  rm -f "$B3/train_raw.jsonl" "$B3"/train_raw.shard*.jsonl 2>/dev/null || true
fi

echo "=== Sanitize train_raw shards (drop null-byte tails) ==="
python3 - "$B3" <<'PY'
import json
import sys
from pathlib import Path
b3 = Path(sys.argv[1])
for p in b3.glob("train_raw.shard*.jsonl"):
    good = []
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip().replace("\x00", "")
        if not line:
            continue
        try:
            json.loads(line)
            good.append(line)
        except json.JSONDecodeError:
            pass
    p.write_text("\n".join(good) + ("\n" if good else ""))
    print(f"  {p.name}: {len(good)} rows")
PY

if [[ "${SKIP_CORPUS:-0}" != "1" ]]; then
  echo "=== Corpus $COUNT ==="
  COUNT="$COUNT" "$SCRIPT_DIR/build_loli_batch3.sh"
else
  echo "=== Corpus (skip, SKIP_CORPUS=1) ==="
fi

echo "=== Teacher gen ${NUM_SHARDS}× GPU ==="
"$LOLI_LEGACY/run_loli15s_teacher_gen_parallel.sh"

echo "=== Done $(date -Is) ==="
wc -l "$B3/train_raw.jsonl" 2>/dev/null || true
find "$B3/wavs/v15" -name '*.wav' | wc -l
echo "Next: merge_batch3_into_loli15s.sh && run_loli_v2_qc.sh"
