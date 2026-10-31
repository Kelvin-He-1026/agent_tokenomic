"""NemoClaw Hermes, run as shipped: one `hermes -z` one-shot per task.

Hermes is an agent product rather than a framework. It brings its own system
prompt, tool set and loop, so none of the parity rows apply except the endpoint,
and that one is what makes it measurable: stack h puts the proxy where the
sandbox's recorded route lands (host.openshell.internal:8000, see
agent/stacks.py), so every Hermes LLM call is recorded like any framework's.

Each run, inside the sandbox (`openshell sandbox exec`, on the gateway
NemoClaw recorded for it):

  1. kills any Hermes process a timed-out run of the same slot left behind, and
     restores the bench profile from the snapshot setup_sandbox.sh took, so no
     run can see an earlier run's sessions, memories or skills;
  2. runs `hermes -p <profile> -t <toolsets> -z <question>` from the
     attachments directory, so a task's file opens by its bare name;
  3. prints the answer, Hermes's own usage file, and the run's timestamps.

Every run writes its run id into the restored profile's X-Run-Id header (the
custom provider's extra_headers, a placeholder setup_sandbox.sh put in the
snapshot), so the proxy files each request under its run whichever run was
bound last. The sandbox's L7 policy permits only the plain /v1 paths, so the
run cannot travel in the URL as it does for in-process frameworks.

Concurrent runs (the load test) each also get a slot: their own copy of the
profile (bench-s3), usage file and kill pattern. A run without a slot restores
and uses the profile itself.

The exec goes straight to OpenShell, not through `nemohermes <sandbox> exec`:
the NemoClaw CLI holds a host-wide lock for as long as the command runs, so
two runs could never be in flight at once, and it adds about 1 s of start-up
per run. The command runs as the same user, in the same directory, with the
same environment, under the same sandbox policy.

Timing: the bench-spans plugin (nemoclaw_hermes/hermes_plugins), installed in
the profile by setup_sandbox.sh, timestamps each tool call inside the sandbox.
Those become tool_call rows (fixture_match "agent"), so t_tool_ms is measured,
not left in T_Orch. The script's own timestamps, with the plugin's load time
and first and last events, name the orchestration around the model calls:

  exec_launch      harness starts the exec -> the script runs in the sandbox
  profile_restore  the profile restore
  hermes_boot      Hermes starts -> its plugins are loaded
  agent_setup      plugins loaded -> the first model request
  hermes_exit      the last model or tool event -> Hermes exits
  exec_return      Hermes exits -> the exec returns to the harness

The tool count stays the proxy's (calls the model asked for), as before.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..adapters.base import AgentResult
from ..config import RunConfig
from ..telemetry import now_ms
from ..tools.registry import ToolRegistry

USAGE_MARKER = "<<<HERMES_USAGE>>>"
SPANS_MARKER = "<<<HERMES_SPANS>>>"

# Stops every Hermes process a run of this slot ($inst) started: the NemoClaw
# `hermes` wrapper and the hermes.real child it spawns, which a kill of the
# wrapper alone leaves running. They are found by the usage-file argument only
# benchmark runs pass, one per slot; the pattern is assembled from $B and
# $inst so this script's own command line, which holds it unexpanded, never
# matches.
_KILL_HERMES = r"""
B=/sandbox/.bench
pkill -f "usage-file $B/usage-$inst.json" 2>/dev/null && sleep 2
pkill -9 -f "usage-file $B/usage-$inst.json" 2>/dev/null
"""

# $1 question  $2 profile  $3 toolsets  $4 attachments dir  $5 "1" for --yolo
# $6 profile instance (the profile, or profile-<slot>)  $7 run id
_RUN_SCRIPT = "inst=$6\n" + _KILL_HERMES + r"""
T0=$(date +%s%3N)
prof=$2
rm -rf "/sandbox/.hermes/profiles/${inst:?}" "/sandbox/.bench/x-${inst:?}" \
  && mkdir -p "/sandbox/.bench/x-$inst" && tar -C "/sandbox/.bench/x-$inst" -xf "$B/$prof.tar" \
  && mv "/sandbox/.bench/x-$inst/$prof" "/sandbox/.hermes/profiles/$inst" \
  && rmdir "/sandbox/.bench/x-$inst" || exit 97
grep -q __ATK_RUN_ID__ "/sandbox/.hermes/profiles/$inst/config.yaml" || exit 96
sed -i "s#__ATK_RUN_ID__#$7#" "/sandbox/.hermes/profiles/$inst/config.yaml" || exit 97
rm -f "$B/usage-$inst.json" "$B/spans-$inst.jsonl"
cd "$4" || exit 98
T1=$(date +%s%3N)
HERMES_BENCH_SPANS="$B/spans-$inst.jsonl" \
  hermes -p "$inst" -t "$3" ${5:+--yolo} --usage-file "$B/usage-$inst.json" -z "$1"
rc=$?
T2=$(date +%s%3N)
printf '\n%s\n' "<<<HERMES_USAGE>>>"
cat "$B/usage-$inst.json" 2>/dev/null
printf '\n%s\n{"ev":"script","t0":%s,"t1":%s,"t2":%s}\n' "<<<HERMES_SPANS>>>" "$T0" "$T1" "$T2"
cat "$B/spans-$inst.jsonl" 2>/dev/null
exit $rc
"""

# NemoClaw's node CLI prints these on stderr for every command.
_STDERR_NOISE = ("ExperimentalWarning", "trace-warnings", "OCSF")


class HermesOneShotAgent:
    name = "hermes"
    # Tools run inside the sandbox, not through ToolRegistry (see runner).
    tools_outside_registry = True

    def __init__(self, cfg: RunConfig, slot: str = ""):
        self.cfg = cfg
        h = cfg.hermes
        missing = [k for k in ("sandbox", "profile", "toolsets", "docs_dir") if not h.get(k)]
        if missing:
            raise ValueError(f"hermes config needs {missing} (see nemoclaw_hermes/configs)")
        self.sandbox = h["sandbox"]
        self.profile = h["profile"]
        self.toolsets = h["toolsets"]
        self.docs_dir = h["docs_dir"]
        self.yolo = bool(h.get("yolo", False))
        self.instance = f"{self.profile}-{slot}" if slot else self.profile
        self.gateway = self._registry().get("gatewayName") or "nemoclaw"

    def _registry(self) -> dict:
        try:
            return json.loads((Path.home() / ".nemoclaw/sandboxes.json").read_text())["sandboxes"][self.sandbox]
        except Exception:
            return {}

    def _exec(self, *command: str) -> list[str]:
        return ["openshell", "sandbox", "exec", "-g", self.gateway, "-n", self.sandbox, "--no-tty", "--",
                *command]

    def framework_version(self) -> str:
        try:
            s = self._registry()
            image = s.get("imageTag", "").rsplit("@sha256:", 1)[-1][:12]
            return (f"hermes=={s.get('agentVersion')},nemoclaw=={s.get('nemoclawVersion')},"
                    f"openshell=={s.get('openshellVersion')},image=sha256:{image}")
        except Exception:
            return "unknown"

    def run(self, question: str, registry: ToolRegistry) -> AgentResult:
        cmd = self._exec(
            "sh", "-c", _RUN_SCRIPT, "sh",
            question, self.profile, self.toolsets, self.docs_dir, "1" if self.yolo else "",
            self.instance, registry.run_id,
        )
        env = dict(os.environ, NEMOCLAW_DISABLE_TELEMETRY="1", NEMOCLAW_NO_AUTO_AUDIT_LOGS="1")
        # The proxy enforces the task deadline on model calls; this only stops a
        # run whose tools hang after that. The next run's restore kills it.
        timeout = self.cfg.limits.task_timeout_s + 60
        launched = now_ms()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                                  stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            # Killing the exec leaves Hermes running in the sandbox, its last
            # model call still streaming through the proxy into the next run's
            # binding. Stop it now, which also closes that call.
            try:
                subprocess.run(self._exec("sh", "-c", "inst=$1\n" + _KILL_HERMES, "sh", self.instance),
                               capture_output=True, timeout=60, env=env, stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                pass  # the next run's script kills it before starting
            return AgentResult(status="timeout", error=f"hermes exec exceeded {timeout:.0f}s")
        except FileNotFoundError as exc:
            return AgentResult(status="framework_error", error=f"openshell missing: {exc}")

        returned = now_ms()
        answer, _, rest = proc.stdout.partition(USAGE_MARKER)
        usage_text, _, spans_text = rest.partition(SPANS_MARKER)
        try:
            usage = json.loads(usage_text) if usage_text.strip() else {}
        except json.JSONDecodeError:
            usage = {}
        timing = _timing(spans_text, launched, returned)
        stderr = "\n".join(line for line in proc.stderr.splitlines()
                           if line.strip() and not any(n in line for n in _STDERR_NOISE))
        steps = usage.get("api_calls")

        if proc.returncode in (96, 97, 98):
            what = {96: "profile snapshot has no run-id header", 97: "profile restore",
                    98: "attachments dir"}[proc.returncode]
            return AgentResult(status="framework_error", error=f"sandbox setup failed: {what}; "
                               "rerun nemoclaw_hermes/scripts/setup_sandbox.sh")
        if proc.returncode != 0 or usage.get("failed"):
            # On failure Hermes exits 0 and prints its error as the answer, e.g.
            # "API call failed after 3 retries: Connection error." It is the
            # error, not an answer to score.
            detail = stderr[-300:] if proc.returncode else answer.strip()[:300]
            return AgentResult(status="framework_error",
                               error=f"hermes exit {proc.returncode}: {detail}",
                               self_reported_steps=steps, raw={"usage": usage, **timing})
        # A run that stopped without completing hit Hermes's own agent.max_turns.
        status = "completed" if usage.get("completed", True) else "step_limit"
        return AgentResult(final_answer=answer.strip(), status=status,
                           self_reported_steps=steps, raw={"usage": usage, **timing})


def _timing(text: str, launched: float, returned: float) -> dict:
    """Named orchestration phases and tool calls, from the script's
    timestamps and the bench-spans plugin's events (see the module doc)."""
    events = []
    for line in text.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    script = next((e for e in events if e.get("ev") == "script"), None)
    if script is None:
        return {}
    t0, t1, t2 = (float(script[k]) for k in ("t0", "t1", "t2"))
    marks = [e for e in events if e.get("ev") in ("api_start", "api_end", "tool_start", "tool_end")]
    loaded = next((e["t"] for e in events if e.get("ev") == "plugin_loaded"), None)
    phases = [("exec_launch", launched, t0), ("profile_restore", t0, t1), ("exec_return", t2, returned)]
    if loaded is not None:
        phases.append(("hermes_boot", t1, loaded))
        if marks:
            phases += [("agent_setup", loaded, marks[0]["t"]), ("hermes_exit", marks[-1]["t"], t2)]

    # Pair each tool_end with its tool_start: by call id, else the oldest open
    # call of the same tool. A call still open at exit ran until Hermes exited.
    tools, open_calls = [], []
    for e in events:
        if e.get("ev") == "tool_start":
            open_calls.append(e)
        elif e.get("ev") == "tool_end":
            start = next((s for s in open_calls if e.get("id") and s.get("id") == e["id"]), None) \
                or next((s for s in open_calls if s.get("tool") == e.get("tool")), None)
            if start is None:
                continue
            open_calls.remove(start)
            tools.append({**start, "latency_ms": e["t"] - start["t"], "status": e.get("status", "ok"),
                          "error_type": e.get("error_type", ""), "result_bytes": e.get("result_bytes", 0)})
    tools += [{**s, "latency_ms": t2 - s["t"], "status": "error", "error_type": "unfinished"} for s in open_calls]
    for t in tools:
        t["start_ts"] = t.pop("t")
    return {"phases": phases, "tool_calls": sorted(tools, key=lambda t: t["start_ts"])}
