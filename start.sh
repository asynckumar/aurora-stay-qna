#!/usr/bin/env bash
# Start (or restart) both containers in a GitHub Codespace.
# Run from the repository root:   bash start.sh
# It is safe to run again, for example after the Codespace has been asleep.
set -euo pipefail
cd "$(dirname "$0")"

# The python:3.13 images send a large first TLS message, and this network path
# can drop it. Inside a container that shows up as download and API timeouts.
# Capping the TCP segment size avoids it. The rule is lost when the Codespace
# restarts, so it is added here every time it is missing.
RULE=(POSTROUTING -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --set-mss 1300)
if ! sudo iptables -t mangle -C "${RULE[@]}" 2>/dev/null; then
  sudo iptables -t mangle -A "${RULE[@]}"
fi

: "${OPENAI_API_KEY:?Add the Codespaces secret OPENAI_API_KEY, then restart the Codespace}"
: "${OPENAI_API_BASE:?Add the Codespaces secret OPENAI_API_BASE, then restart the Codespace}"

docker build -t aurora-backend backend
docker build -t aurora-frontend frontend
docker rm -f backend frontend 2>/dev/null || true
docker run -d --name backend --restart unless-stopped --network host \
  -e OPENAI_API_KEY -e OPENAI_API_BASE \
  -v "$PWD/backend/data_out:/data" aurora-backend
docker run -d --name frontend --restart unless-stopped --network host aurora-frontend

echo "Waiting for the backend to load the model and the index..."
for _ in $(seq 1 60); do
  if curl -s -m 3 localhost:7860/health | grep -q '"status":"ok"'; then break; fi
  sleep 3
done
curl -s -m 5 localhost:7860/health || echo "Backend is not answering yet: check 'docker logs backend'"
echo

if [ -n "${CODESPACE_NAME:-}" ] && command -v gh >/dev/null 2>&1; then
  gh codespace ports visibility 7860:public 8501:public -c "$CODESPACE_NAME" || true
  echo "API: https://${CODESPACE_NAME}-7860.app.github.dev"
  echo "App: https://${CODESPACE_NAME}-8501.app.github.dev"
fi
