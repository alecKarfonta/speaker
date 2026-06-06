#!/usr/bin/env bash
# Quick batch3 teacher-gen status.
set -euo pipefail
ROOT="${SPEAKER_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
B3="$ROOT/training/loli_15s_batch3"
CORPUS="$B3/corpus/texts.jsonl"
TARGET=$(wc -l < "$CORPUS" 2>/dev/null || echo 0)
DISK=$(find "$B3/wavs/v15" -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
SHM=$(find /dev/shm/loli15s_wavs/v15 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
EM=$(find "$B3/wavs/v15" /dev/shm/loli15s_wavs/v15 -name '*__emotion__*.wav' 2>/dev/null | wc -l | tr -d ' ')

echo "batch3 teacher gen"
echo "  corpus target: $TARGET"
echo "  wavs disk:     $DISK"
echo "  wavs shm:      $SHM"
echo "  emotion wavs:  $EM"
echo "  workers:       $(pgrep -cf 'build_realtime_finetune_dataset.py' || echo 0)"
echo "  moss servers:  $(pgrep -cf moss-tts-server || echo 0)"
for s in 0 1 2; do
  f="$B3/teacher_gen.shard${s}.log"
  if [[ -f "$f" ]]; then
    tail -1 "$f" 2>/dev/null | sed "s/^/  shard$s: /"
  fi
done
