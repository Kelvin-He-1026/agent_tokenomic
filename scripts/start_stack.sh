#!/usr/bin/env bash
# Start the logging proxy (8100) and, with --mock, the scripted upstream (8000).
# The harness always talks to the proxy; the proxy talks to whatever ATK_UPSTREAM
# points at, which is vLLM in a real cell.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv/bin}
UPSTREAM=${ATK_UPSTREAM:-http://127.0.0.1:8000}

if [[ "${1:-}" == "--mock" ]]; then
  "$VENV/uvicorn" agent.proxy.mock_upstream:app --port 8000 --log-level warning \
    > /tmp/atk_mock.log 2>&1 &
  echo "mock upstream  pid=$! port=8000"
fi

ATK_UPSTREAM="$UPSTREAM" "$VENV/uvicorn" agent.proxy.server:app --port 8100 --log-level warning \
  > /tmp/atk_proxy.log 2>&1 &
echo "logging proxy  pid=$! port=8100 upstream=$UPSTREAM"

sleep 4
curl -s http://127.0.0.1:8100/control/health && echo
