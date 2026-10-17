#!/usr/bin/env bash
# Block until vLLM is serving, or report why it never came up.
#
# A fresh serve takes 2-4 minutes: weights load, Mamba kernels JIT-compile, CUDA
# graphs capture. Nothing answers on the port until all of that finishes, so
# polling by hand is guesswork. This waits, then exits 0 when ready.
#
#   ./scripts/wait_ready.sh                 # vLLM on 8000
#   PORT=8100 ./scripts/wait_ready.sh       # the proxy instead
#   ./scripts/serve_vllm.sh ... & ./scripts/wait_ready.sh && <run the benchmark>
set -uo pipefail
eval "$(cd "$(dirname "$0")/.." && .venv/bin/python -m agent.stacks --shell)" || exit 1
PORT=${PORT:-$VLLM_PORT}
TIMEOUT_S=${TIMEOUT_S:-600}
LOG=${LOG:-/tmp/vllm_serve.log}
URL="http://127.0.0.1:${PORT}/v1/models"
[[ "$PORT" == "$PROXY_PORT" ]] && URL="http://127.0.0.1:${PORT}/control/health"

echo "waiting for $URL (up to ${TIMEOUT_S}s)"
start=$(date +%s)
while true; do
  if curl -s --max-time 3 "$URL" > /tmp/.wait_ready_body 2>/dev/null; then
    elapsed=$(( $(date +%s) - start ))
    echo "READY after ${elapsed}s"
    head -c 200 /tmp/.wait_ready_body; echo
    exit 0
  fi

  elapsed=$(( $(date +%s) - start ))
  if (( elapsed >= TIMEOUT_S )); then
    echo "NOT READY after ${elapsed}s"
    [[ -f "$LOG" ]] && { echo "--- last errors in $LOG ---"; grep -iE "error|Traceback" "$LOG" | tail -5; }
    exit 1
  fi

  # If the server process has already died, stop waiting for it.
  if [[ -f "$LOG" ]] && grep -q "Engine core initialization failed" "$LOG" 2>/dev/null; then
    echo "server failed to start after ${elapsed}s"
    grep -iE "ValueError|RuntimeError|Error:" "$LOG" | grep -v "core.py:.*\]   " | tail -3
    exit 1
  fi

  printf '.'
  sleep 5
done
