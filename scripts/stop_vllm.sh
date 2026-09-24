#!/usr/bin/env bash
# Stop the vLLM server and confirm the GPU memory is actually released.
#
# vLLM runs the model in a VLLM::EngineCore child process, and that child holds
# the VRAM. The parent normally tears it down on SIGTERM, but if the parent is
# killed abruptly the child can survive and keep tens of GB reserved, so the
# next serve fails to allocate. This script terminates the parent, waits for the
# GPU to drain, and escalates only if it has to.
#
#   ./scripts/stop_vllm.sh            # port 8000
#   PORT=8001 ./scripts/stop_vllm.sh
set -uo pipefail
PORT=${PORT:-8000}
WAIT_S=${WAIT_S:-30}

gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | paste -sd' '; }

pid=$(ss -lptn "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [[ -z "${pid:-}" ]]; then
  echo "nothing listening on port $PORT"
else
  children=$(pgrep -P "$pid" 2>/dev/null | paste -sd' ' || true)
  echo "stopping vLLM (pid $pid, children: ${children:-none})"
  kill "$pid" 2>/dev/null

  for ((i = 0; i < WAIT_S; i++)); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done

  # Any surviving child still owns the VRAM; take it down explicitly.
  for c in ${children:-}; do
    if kill -0 "$c" 2>/dev/null; then
      echo "  engine child $c still alive, terminating"
      kill "$c" 2>/dev/null
      sleep 3
      kill -0 "$c" 2>/dev/null && { echo "  forcing $c"; kill -9 "$c" 2>/dev/null; }
    fi
  done
fi

# Wait for the driver to reclaim, then report. Leftover compute apps here mean
# something is still holding the GPU and the next serve will fail.
for ((i = 0; i < 15; i++)); do
  [[ -z "$(gpu_pids)" ]] && break
  sleep 1
done

remaining=$(gpu_pids)
echo "GPU memory in use: $(gpu_used)"
if [[ -n "$remaining" ]]; then
  echo "WARNING: still holding the GPU: $remaining"
  echo "  inspect with: nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
  exit 1
fi
echo "GPU released"
