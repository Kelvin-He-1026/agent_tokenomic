#!/usr/bin/env bash
# One-glance status of the serving stack. Non-blocking: reports and exits.
# (To wait for startup instead, use wait_ready.sh.)
#
#   ./scripts/status.sh
set -uo pipefail
cd "$(dirname "$0")/.."

ok()   { printf '  \033[32m%-8s\033[0m %s\n' "$1" "$2"; }
bad()  { printf '  \033[31m%-8s\033[0m %s\n' "$1" "$2"; }
info() { printf '  %-8s %s\n' "$1" "$2"; }

echo "vLLM (port 8000)"
models=$(curl -s --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null)
if [[ -n "$models" ]]; then
  name=$(echo "$models" | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
  ok "UP" "serving $name"
  # Settings as the running server was launched, read from its startup banner.
  if [[ -f /tmp/vllm_serve.log ]]; then
    grep -m1 "tool-parser=" /tmp/vllm_serve.log | sed 's/^ */           /'
    grep -m1 "thinking=" /tmp/vllm_serve.log | sed "s/^ */           /"
  fi
elif ss -lptn 'sport = :8000' 2>/dev/null | grep -q LISTEN; then
  bad "STARTING" "port bound but not answering yet (loading, JIT or CUDA graphs)"
elif pgrep -f "vllm serve" >/dev/null 2>&1; then
  bad "STARTING" "process running, port not bound yet; ./scripts/wait_ready.sh to wait"
else
  bad "DOWN" "start: nohup ./scripts/serve_vllm.sh --config configs/pilot_qwen.yaml > /tmp/vllm_serve.log 2>&1 &"
  if grep -q "Engine core initialization failed" /tmp/vllm_serve.log 2>/dev/null; then
    info "" "last start FAILED — see: grep -iE 'Error' /tmp/vllm_serve.log | tail"
  fi
fi

echo "Proxy (port 8100)"
health=$(curl -s --max-time 3 http://127.0.0.1:8100/control/health 2>/dev/null)
if [[ -n "$health" ]]; then
  ok "UP" "$(echo "$health" | .venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); print("upstream", d["upstream"])')"
else
  bad "DOWN" "start: ATK_UPSTREAM=http://127.0.0.1:8000 ./scripts/start_stack.sh"
fi

echo "GPU"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null \
  | while IFS=, read -r idx used total util; do info "gpu$idx" "$used /$total  util$util"; done

echo "Benchmark"
# A sweep restarts vLLM between modes, and during that window no harness is
# running. Checking only for a harness would report "safe to start" while the
# sweep is about to stop the server again, so sweeps are detected separately.
run=$(pgrep -af "agent.harness.cli" 2>/dev/null | grep -v pgrep | head -1)
lock=output/.sweep.lock
sweep_pid=$( [[ -f $lock ]] && .venv/bin/python -c "import json;print(json.load(open('$lock'))['pid'])" 2>/dev/null )
if [[ -n "$sweep_pid" ]] && kill -0 "$sweep_pid" 2>/dev/null; then
  root=$(.venv/bin/python -c "import json;print(json.load(open('$lock'))['out_root'])")
  cfg=$(.venv/bin/python -c "import json;print(json.load(open('$lock'))['config'])")
  bad "SWEEP" "sweep in progress ($cfg) — don't start anything"
  info "" "-> $root"
  for m in $(.venv/bin/python -c "import json;print(' '.join(json.load(open('$lock'))['modes']))"); do
    n=$( [[ -f "$root/$m/run.jsonl" ]] && wc -l < "$root/$m/run.jsonl" || echo 0 )
    info "" "$(printf '%-7s %3s run(s) done' "$m" "$n")"
  done
  info "" "log: /tmp/sweep.log"
elif [[ -n "$run" ]]; then
  out=$(echo "$run" | grep -oP -- '--out \K\S+')
  done_n=$( [[ -f "$out/run.jsonl" ]] && wc -l < "$out/run.jsonl" || echo 0 )
  bad "RUNNING" "writing to $out ($done_n run(s) done) — don't start another"
else
  ok "IDLE" "no harness running; safe to start one"
fi
