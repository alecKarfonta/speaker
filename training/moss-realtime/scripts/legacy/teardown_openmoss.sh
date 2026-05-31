#!/usr/bin/env bash
# Stop all local moss-tts-server processes and free openmoss HTTP ports.
#
# Port-only kills (fuser) leave orphan servers running — that caused 5–6 moss
# processes during 4-GPU teacher gen and likely contributed to hard freezes.
#
# Usage:
#   ./scripts/teardown_openmoss.sh
#   PORTS=8014,8015,8016,8017 ./scripts/teardown_openmoss.sh

set -euo pipefail

PORTS="${PORTS:-8014,8015,8016,8017}"
WAIT_SEC="${WAIT_SEC:-30}"

IFS=',' read -r -a PORT_ARR <<< "$PORTS"

echo "Killing listeners on ports: ${PORTS}"
for port in "${PORT_ARR[@]}"; do
  fuser -k "${port}/tcp" 2>/dev/null || true
done
sleep 2

if pgrep -f 'moss-tts-server' >/dev/null 2>&1; then
  echo "Sending SIGTERM to remaining moss-tts-server PIDs..."
  pkill -TERM -f 'moss-tts-server' 2>/dev/null || true
  for _ in $(seq 1 "$WAIT_SEC"); do
    pgrep -f 'moss-tts-server' >/dev/null 2>&1 || break
    sleep 1
  done
fi

if pgrep -f 'moss-tts-server' >/dev/null 2>&1; then
  echo "SIGKILL on stubborn moss-tts-server..."
  pkill -KILL -f 'moss-tts-server' 2>/dev/null || true
  sleep 2
fi

n=0
if pids=$(pgrep -f '[m]oss-tts-server' 2>/dev/null); then
  n=$(printf '%s\n' "$pids" | wc -l)
fi
if [[ "$n" -gt 0 ]]; then
  echo "ERROR: $n moss-tts-server still running:" >&2
  pgrep -a -f 'moss-tts-server' >&2 || true
  exit 1
fi

echo "All moss-tts-server stopped."
