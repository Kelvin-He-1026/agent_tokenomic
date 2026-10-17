#!/usr/bin/env bash
# Stop the logging proxy (8100). The upstream on 8000 is left alone by default:
# during a real cell that is vLLM, and restarting it costs minutes of model
# load. Pass --all to stop the upstream too (use when it is the mock).
#
# Stops by port, never `pkill -f`: that pattern would match the caller's own
# command line and kill the shell running it.
set -uo pipefail
eval "$(cd "$(dirname "$0")/.." && .venv/bin/python -m agent.stacks --shell)" || exit 1
pid=$(ss -lptn "sport = :$PROXY_PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [[ -n "${pid:-}" ]]; then
  kill "$pid" && echo "stopped proxy on $PROXY_PORT (pid $pid)"
else
  echo "nothing on port $PROXY_PORT"
fi

# The upstream on 8000 is the mock during a smoke test and vLLM during a real
# cell. vLLM needs its engine child stopped too, or it keeps the VRAM, so that
# case is delegated to stop_vllm.sh rather than killed by port here.
if [[ "${1:-}" == "--all" ]]; then
  "$(dirname "$0")/stop_vllm.sh"
fi
exit 0
