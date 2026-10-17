#!/usr/bin/env bash
# One-glance status of the serving stacks. Non-blocking: reports and exits.
# (To wait for startup instead, use wait_ready.sh.)
#
#   ./scripts/status.sh
#
# Shows stack a (GPU0, vLLM 8000, proxy 8100; also the original single stack,
# which uses the same ports) and stack b (GPU1, vLLM 8001, proxy 8101).
set -uo pipefail
cd "$(dirname "$0")/.."

ok()   { printf '  \033[32m%-8s\033[0m %s\n' "$1" "$2"; }
bad()  { printf '  \033[31m%-8s\033[0m %s\n' "$1" "$2"; }
info() { printf '  %-8s %s\n' "$1" "$2"; }
py()   { .venv/bin/python -c "$@" 2>/dev/null; }

show_stack() {
  local label=$1 vport=$2 pport=$3 lock=$4 flag=$5
  echo "== $label =="

  echo "vLLM (port $vport)"
  local models name
  models=$(curl -s --max-time 3 "http://127.0.0.1:$vport/v1/models" 2>/dev/null)
  if [[ -n "$models" ]]; then
    name=$(echo "$models" | py 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
    ok "UP" "serving $name"
    # As launched: serve_vllm.sh records its settings on every launch.
    if [[ -f "output/serving_$vport.json" ]]; then
      py "
import json
s = json.load(open('output/serving_$vport.json'))
print(f\"           stack={s.get('stack') or 'default'} gpus={s.get('gpus') or 'all'} numa={s.get('numa_node') or 'unpinned'} tp={s['tp_size']}\")
print(f\"           tool-parser={s['tool_call_parser']} reasoning-parser={s['reasoning_parser'] or 'none'} max_num_seqs={s['max_num_seqs']}\")
print(f\"           thinking={s['thinking_setting']} effort={s['reasoning_effort'] or 'n/a'} template-kwargs={json.dumps(s['chat_template_kwargs'])}\")"
    fi
  elif ss -lptn "sport = :$vport" 2>/dev/null | grep -q LISTEN; then
    bad "STARTING" "port bound but not answering yet (loading, JIT or CUDA graphs)"
  else
    bad "DOWN" "start: ${flag:+ATK_STACK=$flag }./scripts/serve_vllm.sh --config configs/<cell>.yaml"
  fi

  echo "Proxy (port $pport)"
  local health
  health=$(curl -s --max-time 3 "http://127.0.0.1:$pport/control/health" 2>/dev/null)
  if [[ -n "$health" ]]; then
    ok "UP" "$(echo "$health" | py 'import json,sys; d=json.load(sys.stdin); print("upstream", d["upstream"])')"
  else
    bad "DOWN" "start: ${flag:+ATK_STACK=$flag }./scripts/start_stack.sh"
  fi

  echo "Benchmark"
  # A sweep restarts vLLM between modes, when no harness is running, so a
  # sweep is detected by its lock rather than by a running harness.
  local sweep_pid run
  sweep_pid=$( [[ -f $lock ]] && py "import json;print(json.load(open('$lock'))['pid'])" )
  if [[ $flag == b ]]; then
    run=$(pgrep -af "agent.harness.cli" 2>/dev/null | grep -v -e pgrep -e 'bash -c' | grep -- '--stack b' | head -1)
  else
    run=$(pgrep -af "agent.harness.cli" 2>/dev/null | grep -v -e pgrep -e 'bash -c' | grep -v -- '--stack b' | head -1)
  fi
  if [[ -n "$sweep_pid" ]] && kill -0 "$sweep_pid" 2>/dev/null; then
    local root cfg n
    root=$(py "import json;print(json.load(open('$lock'))['out_root'])")
    cfg=$(py "import json;print(json.load(open('$lock'))['config'])")
    bad "SWEEP" "sweep in progress ($cfg) — don't start anything on this stack"
    info "" "-> $root"
    for m in $(py "import json;print(' '.join(json.load(open('$lock'))['modes']))"); do
      # Parallel recording writes one folder per worker (w0, w1, ...).
      n=$(cat "$root/$m/run.jsonl" "$root/$m"/w*/run.jsonl 2>/dev/null | wc -l)
      info "" "$(printf '%-7s %3s run(s) done' "$m" "$n")"
    done
  elif [[ -n "$run" ]]; then
    local out dir done_n
    out=$(echo "$run" | grep -oP -- '--out \K\S+')
    # --out gets a timestamp suffix unless --no-timestamp was given.
    dir=$(ls -dt "$out" "${out}"_2* 2>/dev/null | head -1)
    done_n=$( [[ -n "$dir" && -f "$dir/run.jsonl" ]] && wc -l < "$dir/run.jsonl" || echo 0 )
    bad "RUNNING" "writing to ${dir:-$out} ($done_n run(s) done) — don't start another"
  else
    ok "IDLE" "no harness running on this stack"
  fi
  echo
}

show_stack "stack a / default (GPU0)" 8000 8100 output/.sweep.lock a
show_stack "stack b (GPU1)" 8001 8101 output/.sweep_b.lock b

echo "GPU"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null \
  | while IFS=, read -r idx used total util; do info "gpu$idx" "$used /$total  util$util"; done
