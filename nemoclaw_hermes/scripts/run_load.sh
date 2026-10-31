#!/usr/bin/env bash
# Closed-loop concurrency test, end to end: serve (if needed), proxy, one sweep
# per framework (agent_tokenomic/agent/harness/load.py), export.
#
#   ./scripts/run_load.sh --name load_v1 --frameworks hermes,langgraph \
#       --levels 1,4,8,16,32 --run-code load-v1
#
# --cpus-per-socket N (default 0: every CPU; the host has had 48 cores per
# socket since 2026-10-29) pins everything the test runs on the host to N CPUs
# per socket, GPU-local NUMA node first: vLLM (restarted if it runs on other cores), the proxy, the harness, the
# OpenShell gateway, and the sandbox container (docker
# update --cpuset-cpus). The gateway's and the container's previous CPUs are
# restored on exit; vLLM stays pinned until it is restarted.
#
# Arguments other than --name, --config, --tasks, --frameworks, --serve-config
# and --cpus-per-socket are passed to agent.harness.load (e.g. --levels, --min-runs, --exclude,
# --no-stop). Each framework's sweep lands in output/runs/<name>_<framework>_<stamp>/
# with levels.jsonl and load_levels.csv (one row per level), and is exported to
# output/<name>_<framework>_<stamp>.csv, _llm_calls.csv and _stages.csv.
#
# vLLM is reused when it already serves the model and thinking mode the config
# asks for (the harness preflight checks); otherwise it is started from
# --serve-config. Run scripts/setup_sandbox.sh once first.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ATK="${ATK:-$(cd "$HERE/.." && pwd)}"
CONFIG="$HERE/configs/gaia_hermes_qwen_tp2_think_low_load.yaml"
SERVE_CONFIG="$HERE/configs/gaia_hermes_qwen_tp2_think_low.yaml"
NAME="load"
TASKS="$HERE/configs/gaia30_tasks.json"
FRAMEWORKS="hermes,langgraph"
CPUS_PER_SOCKET="${ATK_CPUS_PER_SOCKET:-0}"
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --config) CONFIG="$(realpath "$2")"; shift 2 ;;
    --serve-config) SERVE_CONFIG="$(realpath "$2")"; shift 2 ;;
    --tasks) TASKS="$(realpath "$2")"; shift 2 ;;
    --frameworks) FRAMEWORKS="$2"; shift 2 ;;
    --cpus-per-socket) CPUS_PER_SOCKET="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
export ATK_STACK=h NEMOCLAW_DISABLE_TELEMETRY=1 NEMOCLAW_NO_AUTO_AUDIT_LOGS=1
export ATK_CPUS_PER_SOCKET="$CPUS_PER_SOCKET"
cd "$ATK"
eval "$(.venv/bin/python -m agent.stacks --shell)"
ALL_CPUS="$(cat /sys/devices/system/cpu/online)"
echo "host CPUs for the test: ${CPUSET:-all ($ALL_CPUS)}"

# 1. vLLM: keep a running server unless it serves different settings or CPUs.
read -r SERVED_CONFIG SERVED_CPUSET < <(.venv/bin/python -c "
import json,os,sys; s=json.load(open(sys.argv[1])); print(os.path.realpath(s['config']), s.get('cpuset') or '-')" \
  "output/serving_$VLLM_PORT.json" 2>/dev/null || echo "- -")
if ! curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null \
   || [[ "$SERVED_CONFIG" != "$SERVE_CONFIG" && "$SERVED_CONFIG" != "$CONFIG" ]] \
   || [[ "$SERVED_CPUSET" != "${CPUSET:--}" ]]; then
  ./scripts/stop_vllm.sh >/dev/null 2>&1 || true
  nohup ./scripts/serve_vllm.sh --config "$SERVE_CONFIG" > /tmp/vllm_serve_h.log 2>&1 &
  LOG=/tmp/vllm_serve_h.log ./scripts/wait_ready.sh
fi

# 2. A fresh proxy, so no binding from an earlier session is left ($PIN puts
# it on the test's CPUs).
./scripts/stop_stack.sh >/dev/null 2>&1 || true
./scripts/start_stack.sh

# 3. The OpenShell gateway and the sandbox container on the same CPUs, put
# back as they were on exit.
GW_PIDS="$(pgrep -u "$USER" -f '^/usr/local/bin/openshell-gateway' || true)"
SANDBOX="$(docker ps -q --filter name=openshell- | head -1)"
GW_BEFORE="" SANDBOX_BEFORE=""
restore_cpus() {
  for p in $GW_PIDS; do [[ -n "$GW_BEFORE" ]] && taskset -a -cp "$GW_BEFORE" "$p" >/dev/null 2>&1 || true; done
  [[ -n "$SANDBOX" ]] && docker update --cpuset-cpus "${SANDBOX_BEFORE:-$ALL_CPUS}" "$SANDBOX" >/dev/null 2>&1 || true
}
if [[ -n "$CPUSET" ]]; then
  for p in $GW_PIDS; do
    GW_BEFORE="$(taskset -cp "$p" | awk -F': ' '{print $2}')"
    taskset -a -cp "$CPUSET" "$p" >/dev/null
  done
  if [[ -n "$SANDBOX" ]]; then
    SANDBOX_BEFORE="$(docker inspect -f '{{.HostConfig.CpusetCpus}}' "$SANDBOX")"
    docker update --cpuset-cpus "$CPUSET" "$SANDBOX" >/dev/null
  fi
  trap restore_cpus EXIT
  echo "pinned to $CPUSET: vLLM, proxy, harness, gateway ${GW_PIDS:-none}, sandbox ${SANDBOX:-none}"
fi

# 4. One sweep per framework (agent.harness.load pins itself to $CPUSET; its
# power sampler records that set, then keeps to CPU 0, outside it), then its
# export.
for FW in ${FRAMEWORKS//,/ }; do
  LOGF="$HERE/output/${NAME}_${FW}.log"
  .venv/bin/python -u -m agent.harness.load --stack h --config "$CONFIG" --tasks "$TASKS" \
    --framework "$FW" --out "$HERE/output/runs/${NAME}_${FW}" "${ARGS[@]}" 2>&1 \
    | grep --line-buffered -v 'ExperimentalWarning\|trace-warnings' | tee "$LOGF"
  RUN_DIR="$(grep -oP '^wrote \K.*' "$LOGF" | tail -1)"
  [[ -n "$RUN_DIR" ]] || { echo "no run folder for $FW; see $LOGF"; continue; }
  STEM="$HERE/output/$(basename "$RUN_DIR")"
  .venv/bin/python scripts/export_csv.py --runs "$RUN_DIR" --tasks "$TASKS" --out "$STEM.csv" --calls
  cp "$RUN_DIR/load_levels.csv" "${STEM}_levels.csv"
  # Where each run's time, GPU energy and CPU went, by stage; per-run figures
  # only for runs alone on the machine (the concurrency-1 level).
  .venv/bin/python scripts/attribute_energy.py --runs "$RUN_DIR" --power \
    --out "${STEM}_energy.csv" --stages-out "${STEM}_stages.csv" | tee -a "$LOGF"
done
