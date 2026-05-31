#!/usr/bin/env bash
# Pause heavy Docker services during teacher gen (reduces node/RCU lockups).
#
# Usage:
#   ./scripts/lighten_host_for_teacher_gen.sh stop
#   ./scripts/lighten_host_for_teacher_gen.sh restore

set -euo pipefail

STATE="${HOME}/.cache/speaker-teacher-gen-docker.state"
# Node-heavy / GPU-adjacent services; add names as needed.
CONTAINERS=(
  airflow-worker-gpu
  airflow-worker-cpu
  airflow-webserver
  airflow-scheduler
  graphrag-neo4j-1
  browser_agent
)

cmd="${1:-stop}"

if [[ "$cmd" == "stop" ]]; then
  : >"$STATE"
  for c in "${CONTAINERS[@]}"; do
    if docker ps -q -f "name=^/${c}$" 2>/dev/null | grep -q .; then
      echo "Stopping $c"
      echo "$c" >>"$STATE"
      docker stop "$c" >/dev/null || true
    fi
  done
  if [[ ! -s "$STATE" ]]; then
    echo "No matching containers running."
  else
    echo "Stopped $(wc -l <"$STATE") containers. State: $STATE"
  fi
elif [[ "$cmd" == "restore" ]]; then
  if [[ ! -f "$STATE" ]]; then
    echo "No state file ($STATE); nothing to restore."
    exit 0
  fi
  while read -r c; do
    [[ -z "$c" ]] && continue
    echo "Starting $c"
    docker start "$c" >/dev/null || true
  done <"$STATE"
  rm -f "$STATE"
  echo "Restore done."
else
  echo "Usage: $0 {stop|restore}" >&2
  exit 1
fi
