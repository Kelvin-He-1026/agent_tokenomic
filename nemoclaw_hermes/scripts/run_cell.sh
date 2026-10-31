#!/usr/bin/env bash
# Run one Hermes GAIA cell end to end: serve (if needed), proxy, run, export.
#
#   ./scripts/run_cell.sh --name gaia_hermes --frameworks hermes,langgraph \
#       --repeats 3 --difficulty-mix easy=10,medium=10,hard=10 --run-code hermes-v1
#
# Every argument except --name and --config is passed to agent.harness.cli.
# Results land in output/runs/<name>_<stamp>/ and output/<name>_<stamp>.csv.
#
# An interrupted cell is finished with the same arguments plus
# --resume output/runs/<name>_<stamp>: the missing runs join that folder and
# session, the log is appended to, and the export covers the whole cell.
#
# The harness runs from the parent folder on stack h (agent/stacks.py):
# vLLM on 127.0.0.1:8002, the proxy on 172.18.0.1:8000, the address the
# sandbox's recorded route reaches. Run scripts/setup_sandbox.sh once first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ATK="${ATK:-$(cd "$HERE/.." && pwd)}"
CONFIG="$HERE/configs/gaia_hermes_qwen_tp2.yaml"
NAME="gaia_hermes"
TASKS="GAIA_stack/tasks/phase1-GAIA-text.json"
TEE=(tee)
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --config) CONFIG="$(realpath "$2")"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --resume) TEE=(tee -a); ARGS+=("$1" "$(realpath "$2")"); shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
export ATK_STACK=h NEMOCLAW_DISABLE_TELEMETRY=1 NEMOCLAW_NO_AUTO_AUDIT_LOGS=1
cd "$ATK"
eval "$(.venv/bin/python -m agent.stacks --shell)"

SANDBOX="$(.venv/bin/python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['hermes']['sandbox'])" "$CONFIG")"
PROFILE="$(.venv/bin/python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['hermes']['profile'])" "$CONFIG")"
nemohermes "$SANDBOX" exec --no-stdin -- test -f "/sandbox/.bench/$PROFILE.tar" 2>/dev/null \
  || { echo "no profile snapshot in $SANDBOX; run $HERE/scripts/setup_sandbox.sh first"; exit 1; }

# 1. vLLM: reuse a server already launched from this config, else (re)start it.
# The launch record holds the config path as given, relative to agent_tokenomic.
SERVED_CONFIG="$(.venv/bin/python -c "import json,os,sys; print(os.path.realpath(json.load(open(sys.argv[1]))['config']))" \
  "output/serving_$VLLM_PORT.json" 2>/dev/null || true)"
if curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null && [[ "$SERVED_CONFIG" == "$CONFIG" ]]; then
  echo "vLLM already serving this config on :$VLLM_PORT"
else
  ./scripts/stop_vllm.sh >/dev/null 2>&1 || true
  nohup ./scripts/serve_vllm.sh --config "$CONFIG" > /tmp/vllm_serve_h.log 2>&1 &
  LOG=/tmp/vllm_serve_h.log ./scripts/wait_ready.sh
fi

# 2. A fresh proxy per cell: the runner requires its budget acknowledgment, and
#    restarting clears any binding left by an earlier session.
./scripts/stop_stack.sh >/dev/null 2>&1 || true
./scripts/start_stack.sh

# 3. Run. --stack h points the harness at the proxy's bridge address.
OUT="$HERE/output/runs/$NAME"
.venv/bin/python -u -m agent.harness.cli --stack h --config "$CONFIG" --tasks "$TASKS" \
  --out "$OUT" "${ARGS[@]}" | "${TEE[@]}" "$HERE/output/$NAME.log"
RUN_DIR="$(grep -oP '^wrote \K.*' "$HERE/output/$NAME.log" | tail -1)"

# 4. Export: one row per run, plus one per LLM call and the full texts.
.venv/bin/python scripts/export_csv.py --runs "$RUN_DIR" --tasks "$TASKS" \
  --out "$HERE/output/$NAME.csv" --calls
