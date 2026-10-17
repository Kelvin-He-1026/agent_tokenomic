#!/usr/bin/env bash
# Start the logging proxy (8100) and, with --mock, the scripted upstream (8000).
# The harness always talks to the proxy; the proxy talks to whatever ATK_UPSTREAM
# points at, which is vLLM in a real cell.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv/bin}
eval "$(.venv/bin/python -m agent.stacks --shell)" || exit 1
UPSTREAM=${ATK_UPSTREAM:-http://127.0.0.1:$VLLM_PORT}

if [[ "${1:-}" == "--mock" ]]; then
  $PIN "$VENV/uvicorn" agent.proxy.mock_upstream:app --port "$VLLM_PORT" --log-level warning \
    > "/tmp/atk_mock${STACK_NAME:+_$STACK_NAME}.log" 2>&1 &
  echo "mock upstream  pid=$! port=$VLLM_PORT"
fi

ATK_UPSTREAM="$UPSTREAM" $PIN "$VENV/uvicorn" agent.proxy.server:app --port "$PROXY_PORT" --log-level warning \
  > "/tmp/atk_proxy${STACK_NAME:+_$STACK_NAME}.log" 2>&1 &
echo "logging proxy  pid=$! port=$PROXY_PORT upstream=$UPSTREAM"

sleep 4
curl -s "http://127.0.0.1:$PROXY_PORT/control/health" && echo
