#!/usr/bin/env bash
# Generate WAVs for corpus lines that have no file on disk yet.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
B3="$ROOT/training/loli_15s_batch3"
LOLI_LEGACY="$ROOT/training/loli_15s/scripts/legacy"
MISSING="$B3/corpus/missing.jsonl"
LOG="$B3/logs/mop_up_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$B3/logs" "$B3/wavs/v15"
exec > >(tee -a "$LOG") 2>&1
echo "=== batch3 mop-up @ $(date -Is) ==="
echo "Log: $LOG"

python3 - "$B3" "$MISSING" <<'PY'
import json
import sys
from pathlib import Path

b3 = Path(sys.argv[1])
out = Path(sys.argv[2])
corpus = b3 / "corpus/texts.jsonl"
wav_dir = b3 / "wavs/v15"

def has_wav(cid: str) -> bool:
    if any(wav_dir.glob(f"{cid}.wav")):
        return True
    return any(wav_dir.glob(f"{cid}__*.wav"))

rows = []
for line in corpus.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if not has_wav(row["id"]):
        rows.append(line)

out.write_text("\n".join(rows) + ("\n" if rows else ""))
print(f"Missing corpus: {len(rows)} lines -> {out}")
if not rows:
    sys.exit(0)
PY

if [[ ! -s "$MISSING" ]]; then
  echo "Nothing missing — done."
  exit 0
fi

export SPEAKER_ROOT="$ROOT"
export MOSS_RT_TRAIN_DIR="$B3"
export LOG_DIR="$B3"
export CORPUS="$MISSING"
export OUT_DIR="$B3"
export WAV_REL_ROOT="training/loli_15s_batch3/wavs"
export REF_WAV="$ROOT/data/voices/loli/loli_15s.wav"
export VOICE_QC="${VOICE_QC:-0}"
export GPUS="${GPUS:-0,1}"
export PORTS="${PORTS:-8014,8015}"
export NUM_SHARDS="${NUM_SHARDS:-2}"
export MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-14000}"
export MIN_AVAIL_GB="${MIN_AVAIL_GB:-12}"
export OPENMOSS_AUX_CPU="${OPENMOSS_AUX_CPU:-0}"
export HEALTH_LOG_DIR="$B3/logs/health_mopup"
export FRESH_WAVS=0

echo "=== Teacher gen ${NUM_SHARDS}× GPU (missing only) ==="
"$LOLI_LEGACY/run_loli15s_teacher_gen_parallel.sh"

echo "=== Merge train_raw shards ==="
cat "$B3"/train_raw.shard*.jsonl > "$B3/train_raw.jsonl"

python3 - "$B3" <<'PY'
import json
import sys
from pathlib import Path

b3 = Path(sys.argv[1])
ids = [json.loads(l)["id"] for l in (b3 / "corpus/texts.jsonl").read_text().splitlines() if l.strip()]
wavs = {p.stem.split("__")[0] for p in (b3 / "wavs/v15").glob("*.wav")}
print(f"corpus={len(ids)} wavs={len(wavs)} missing={len(ids)-len(wavs)}")
PY

echo "=== Done $(date -Is) ==="
