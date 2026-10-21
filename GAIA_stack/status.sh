#!/usr/bin/env bash
# Where is the GAIA sweep? Read-only: prints the run log, what is running,
# runs finished per stage, and which search engine the recordings came from.
#
#     GAIA_stack/status.sh
cd "$(dirname "$0")/.." || exit 1

echo "== $(TZ=America/New_York date '+%m-%d %H:%M ET')"
LOG=output/run_gaia.log; [[ -f $LOG ]] || LOG=output/run_gaia_35.log
echo "== log ($LOG)"
cat "$LOG" 2>/dev/null || echo "no run log"

echo
echo "== running"
if [[ -f output/run_gaia.pid ]] && kill -0 "$(cat output/run_gaia.pid)" 2>/dev/null; then
  echo "run_gaia.py: running (pid $(cat output/run_gaia.pid))"
else
  echo "run_gaia.py: not running"
fi
echo "harness processes: $(pgrep -fc 'agent\.harness\.cli')   vLLM: $(pgrep -fc 'vllm serve')"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/GPU /'

echo
echo "== runs finished (newest GAIA folders; counted from run.jsonl, as the logs are buffered)"
for d in $(ls -td output/runs/gaia_* output/runs/probe_Qwen3.8-27B_* output/runs/thinking_sweep_Qwen3.8-27B_* 2>/dev/null | head -4); do
  grep -q "GAIA_stack" "$d"/configs/*.yaml 2>/dev/null || continue
  for plan in $(find "$d" -name plan.jsonl | sort); do
    run="$(dirname "$plan")/run.jsonl"
    n=$( [[ -f "$run" ]] && wc -l < "$run" || echo 0 )
    limit=$( [[ -f "$run" ]] && grep -c '"completion_type": *"step_limit"' "$run" || true )
    printf '%-55s %4s / %-4s runs  (%s step_limit)\n' "${plan#output/runs/}" "$n" "$(wc -l < "$plan")" "$limit"
  done
done

echo
echo "== search engine of recorded web searches"
.venv/bin/python - <<'EOF'
import collections, glob, json
fs = [json.load(open(f)) for f in glob.glob("GAIA_stack/fixtures/web_search/*.json")]
print(dict(collections.Counter(f.get("backend", "tavily (before labels)") for f in fs)),
      "| empty:", sum(f["result"].strip() == "No results found." for f in fs),
      "| pages fetched:", len(glob.glob("GAIA_stack/fixtures/fetch_url/*.json")))
EOF
[[ -f output/.tavily_exhausted ]] && echo "Tavily credits used up -> DuckDuckGo since then" || echo "Tavily still in use"

summary=$(ls -t output/runs/gaia_*/summary.md output/runs/thinking_sweep_Qwen3.8-27B_*/summary.md 2>/dev/null | head -1)
[[ -n "$summary" ]] && grep -q GAIA_stack "$(dirname "$summary")"/configs/*.yaml 2>/dev/null \
  && echo && echo "== results: $summary"
exit 0
