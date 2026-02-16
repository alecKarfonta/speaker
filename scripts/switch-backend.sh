#!/bin/bash
# switch-backend.sh — Switch the Speaker frontend to a different TTS backend
#
# Usage:
#   ./scripts/switch-backend.sh glm     # Switch to GLM-TTS (port 8012)
#   ./scripts/switch-backend.sh moss    # Switch to MOSS-TTS (port 8013)
#   ./scripts/switch-backend.sh qwen    # Switch to Qwen3-TTS (port 8016)
#
# This updates the TTS_BACKEND_HOST in docker-compose.yml and restarts the frontend.

set -euo pipefail

COMPOSE_FILE="$(dirname "$0")/../docker-compose.yml"

case "${1:-}" in
    glm)
        BACKEND_HOST="tts-api:8000"
        PROFILE=""
        DESC="GLM-TTS"
        PORT="8012"
        ;;
    moss)
        BACKEND_HOST="moss-tts:8000"
        PROFILE="--profile moss"
        DESC="MOSS-TTS"
        PORT="8013"
        ;;
    qwen)
        BACKEND_HOST="qwen-tts:8000"
        PROFILE="--profile qwen"
        DESC="Qwen3-TTS"
        PORT="8016"
        ;;
    status)
        echo "Current TTS_BACKEND_HOST in docker-compose.yml:"
        grep -A1 "TTS_BACKEND_HOST" "$COMPOSE_FILE" | grep -o '[a-z-]*:8000' || echo "  (not set)"
        echo ""
        echo "Running TTS containers:"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "tts|moss|qwen" || echo "  (none)"
        exit 0
        ;;
    *)
        echo "Usage: $0 {glm|moss|qwen|status}"
        echo ""
        echo "  glm    — GLM-TTS backend (tts-api:8000, port 8012)"
        echo "  moss   — MOSS-TTS backend (moss-tts:8000, port 8013)"
        echo "  qwen   — Qwen3-TTS backend (qwen-tts:8000, port 8016)"
        echo "  status — Show current backend configuration"
        exit 1
        ;;
esac

echo "🔄 Switching TTS backend to ${DESC} (${BACKEND_HOST})"

# Update TTS_BACKEND_HOST in docker-compose.yml
sed -i "s|TTS_BACKEND_HOST=.*|TTS_BACKEND_HOST=${BACKEND_HOST}|" "$COMPOSE_FILE"
echo "   ✅ Updated docker-compose.yml"

# Also update vite.config.ts dev proxy for local dev
VITE_CONFIG="$(dirname "$0")/../frontend/vite.config.ts"
case "${1}" in
    glm)  DEV_PORT="8012" ;;
    moss) DEV_PORT="8013" ;;
    qwen) DEV_PORT="8016" ;;
esac
sed -i "s|http://localhost:80[0-9][0-9]|http://localhost:${DEV_PORT}|g" "$VITE_CONFIG"
echo "   ✅ Updated vite.config.ts dev proxy → localhost:${DEV_PORT}"

# Restart the frontend container
echo "   🔄 Restarting frontend..."
docker compose up -d frontend 2>/dev/null
echo "   ✅ Frontend restarted"

echo ""
echo "✅ Now using ${DESC} backend"
echo "   API: http://localhost:${PORT}"
echo "   Frontend: http://localhost:3012"
echo ""
echo "💡 Make sure the ${DESC} backend is running:"
if [ -n "$PROFILE" ]; then
    echo "   docker compose ${PROFILE} up -d"
else
    echo "   docker compose up -d tts-api"
fi
