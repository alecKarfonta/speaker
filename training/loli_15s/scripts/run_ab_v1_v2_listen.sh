#!/usr/bin/env bash
# A/B listen page: epoch-7 (v1) vs loli15s-v2-merged on the same emotion prompts.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
OUT="$LOLI/eval/listen/v1_vs_v2"
PORT="${MOSS_RT_PORT:-8016}"
GPU="${MOSS_RT_GPU:-2}"
API="http://127.0.0.1:${PORT}"
V1="$LOLI/exports/loli15s-epoch7-merged"
V2="$LOLI/exports/loli15s-v2-merged"
LOG="$LOLI/logs/ab_v1_v2_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUT/v1" "$OUT/v2" "$LOLI/logs"
exec > >(tee -a "$LOG") 2>&1
echo "=== v1 vs v2 A/B @ $(date -Is) ==="

wait_health() {
  local n=0
  while [[ $n -lt 40 ]]; do
    curl -sf --max-time 5 "$API/health" >/dev/null && return 0
    sleep 10
    n=$((n + 1))
  done
  echo "Server not healthy at $API" >&2
  return 1
}

gen_side() {
  local label="$1"
  local subdir="$2"
  echo "=== Generate $label ==="
  MOSS_RT_API="$API" python3 "$LOLI/scripts/generate_eval_samples.py" \
    --preset emotion \
    --out "$OUT/$subdir" \
    --quick-ab
}

stop_server() {
  pkill -f 'app.moss_api' 2>/dev/null || true
  sleep 3
}

for model in "$V1" "$V2"; do
  [[ -f "$model/model.safetensors" ]] || { echo "Missing $model"; exit 1; }
done

stop_server
echo "=== v1 (epoch-7) on GPU $GPU ==="
MOSS_RT_GPU="$GPU" MOSS_RT_PORT="$PORT" MOSS_RT_MODEL_ID="$V1" MOSS_RT_NATIVE_VOICE=true \
  bash "$ROOT/scripts/start-moss-realtime.sh" &
srv=$!
wait_health
gen_side "v1 epoch-7" "v1"
kill "$srv" 2>/dev/null || true
stop_server

echo "=== v2 merged on GPU $GPU ==="
MOSS_RT_GPU="$GPU" MOSS_RT_PORT="$PORT" MOSS_RT_MODEL_ID="$V2" MOSS_RT_NATIVE_VOICE=true \
  bash "$ROOT/scripts/start-moss-realtime.sh" &
srv=$!
wait_health
gen_side "v2" "v2"
kill "$srv" 2>/dev/null || true
stop_server

python3 - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
v1 = json.loads((out / "v1/manifest.json").read_text())
v2 = {r["file"]: r for r in json.loads((out / "v2/manifest.json").read_text())}

rows = []
for r in v1:
    f = r["file"]
    v2r = v2.get(f, {})
    rows.append({**r, "v2_audio_s": v2r.get("audio_s"), "v2_file": f})

html = [
    "<!DOCTYPE html><html><head><meta charset=utf-8>",
    "<title>loli v1 vs v2 — emotion A/B</title>",
    "<style>",
    "body{font-family:system-ui;max-width:72rem;margin:2rem auto;padding:0 1rem}",
    "table{border-collapse:collapse;width:100%}",
    "td,th{border:1px solid #ccc;padding:.5rem;vertical-align:top}",
    "audio{width:100%;min-width:12rem}",
    ".tag{color:#555;font-size:.85rem}",
    "th.v1{background:#f0f4ff} th.v2{background:#f0fff4}",
    "</style></head><body>",
    "<h1>loli v1 vs v2 — emotion A/B</h1>",
    "<p>v1 = <code>loli15s-epoch7-merged</code> (5.1k rows) · "
    "v2 = <code>loli15s-v2-merged</code> (8.9k rows, emotion-heavy)</p>",
    "<table><tr><th>#</th><th>Tag</th><th>Text</th>"
    "<th class=v1>v1 epoch-7</th><th class=v2>v2</th></tr>",
]
for i, r in enumerate(rows, 1):
    html.append(
        f"<tr><td>{i}</td><td class=tag>{r['tag']}</td>"
        f"<td>{r['text']}</td>"
        f"<td><audio controls src='v1/{r['file']}'></audio>"
        f"<br><span class=tag>{r.get('audio_s','?')}s</span></td>"
        f"<td><audio controls src='v2/{r['file']}'></audio>"
        f"<br><span class=tag>{r.get('v2_audio_s','?')}s</span></td></tr>"
    )
html.append("</table></body></html>")
(out / "index.html").write_text("\n".join(html), encoding="utf-8")
print(f"Wrote {out / 'index.html'} ({len(rows)} pairs)")
PY

echo "=== Done: $OUT/index.html ==="
echo "Log: $LOG"
