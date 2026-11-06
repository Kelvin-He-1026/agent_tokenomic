"""External Harbor adapter: NemoClaw Hermes, run as shipped, on Terminal-Bench tasks.

Hermes runs where the GAIA arm runs it: inside the NemoClaw sandbox
(gaia-hermes), as one `hermes -z` per trial, from a fresh copy of the bench
profile snapshot, its model calls going through the sandbox's recorded route
to the logging proxy (which binds them to the trial with the X-Run-Id header
and enforces the budget). The difference is where its tools act: the profile
copy switches the terminal backend to ssh, aimed at a per-trial relay
(terminal_bench/ssh_relay.py) that runs every command in this trial's task
container. Hermes's file tools work through the same backend, so reads,
writes and patches land in the task container, which Harbor's verifier then
checks. The sandbox keeps its own policy, user and filesystem; it gains one
egress rule (configs/policies/terminal-bench-relay.yaml) and no Docker access.

    sandbox: hermes -> ssh (ControlMaster) -> socat -> OpenShell proxy
             -> 172.18.0.1:2222 relay sshd -> docker exec -i <task container>

Per trial, under the trial's agent/ log directory:
    hermes_stdout.txt, hermes_stderr.txt  what the exec printed
    usage.json, spans.jsonl               Hermes's usage file, bench-spans events
    proxy/llm_call.jsonl, proxy/blobs/    the proxy's record of every model call
    relay/relay_calls.jsonl, relay/cmds/  every command the task container ran
    accounting.json                       completion type, budget and timings
"""
from __future__ import annotations

import asyncio
import base64
import getpass
import json
import os
import time
import uuid
from importlib.metadata import version
from pathlib import Path

import httpx
from harbor.agents.base import BaseAgent
from harbor.constants import MAIN_SERVICE_NAME
from harbor.environments.docker.docker import DockerEnvironment, _sanitize_docker_compose_project_name
from harbor.models.agent.context import AgentContext

from agent.config import RunConfig
from agent.hermes.oneshot_agent import SPANS_MARKER, USAGE_MARKER, _STDERR_NOISE, _timing
from terminal_bench.ssh_relay import SSHRelay

ROOT = Path(__file__).resolve().parents[1]

# Kills what an earlier trial of this instance left running in the sandbox
# (found by the usage-file argument only these runs pass) and its SSH masters.
_KILL = r"""
B=/sandbox/.bench
pkill -f "usage-file $B/usage-$inst.json" 2>/dev/null && sleep 2
pkill -9 -f "usage-file $B/usage-$inst.json" 2>/dev/null
for s in /tmp/hermes-ssh/*.sock; do [ -S "$s" ] && ssh -o ControlPath="$s" -O exit relay >/dev/null 2>&1; done
rm -rf /tmp/hermes-ssh /sandbox/.bench/tb
"""

#  $1 instance  $2 profile  $3 ssh alias  $4 task cwd  $5 run id  $6 client key (base64)
#  $7 known_hosts line  $8 toolsets  $9 max_turns  $10 sandbox workdir  $11 relay port
#  $12 relay ssh user  $13 "1" for --yolo  $14 instruction
_RUN = "inst=$1\n" + _KILL + r"""
prof=$2 alias=$3 cwd=$4 run_id=$5 toolsets=$8 turns=$9 workdir=${10} port=${11} user=${12} yolo=${13}
instruction=${14}
T=$B/tb P=/sandbox/.hermes/profiles/$inst
T0=$(date +%s%3N)
rm -rf "/sandbox/.hermes/profiles/${inst:?}" "$B/x-${inst:?}" \
  && mkdir -p "$B/x-$inst" && tar -C "$B/x-$inst" -xf "$B/$prof.tar" \
  && mv "$B/x-$inst/$prof" "$P" && rmdir "$B/x-$inst" || exit 97
grep -q __ATK_RUN_ID__ "$P/config.yaml" || exit 96
sed -i "s#__ATK_RUN_ID__#$run_id#" "$P/config.yaml" || exit 97
# The only profile changes: terminal backend -> the relay, and the turn cap.
/opt/hermes/.venv/bin/python - "$P/config.yaml" "$alias" "$user" "$cwd" "$turns" <<'PY' || exit 94
import sys, yaml
path, alias, user, cwd, turns = sys.argv[1:]
with open(path) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('terminal', {}).update(backend='ssh', ssh_host=alias, ssh_user=user, cwd=cwd)
cfg.setdefault('agent', {})['max_turns'] = int(turns)
with open(path, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
# ssh reaches the relay through the sandbox's egress proxy (CONNECT via socat).
proxy=${HTTPS_PROXY:-${https_proxy:-}}; proxy=${proxy#*://}; proxy=${proxy%%/*}
phost=${proxy%:*} pport=${proxy##*:}
[ -n "$phost" ] && [ -n "$pport" ] && [ "$phost" != "$proxy" ] || exit 95
(umask 077 && mkdir -p "$T" /sandbox/.ssh \
  && printf '%s' "$6" | base64 -d > "$T/id_ed25519" \
  && printf '%s\n' "$7" > "$T/known_hosts" \
  && cat > /sandbox/.ssh/config <<EOF
# Written by terminal_bench/nemoclaw_agent.py for the current trial only.
Host $alias
  HostName host.openshell.internal
  Port $port
  User $user
  IdentityFile $T/id_ed25519
  IdentitiesOnly yes
  HostKeyAlias $alias
  UserKnownHostsFile $T/known_hosts
  ProxyCommand /usr/bin/socat - PROXY:$phost:%h:%p,proxyport=$pport
  ServerAliveInterval 30
EOF
) || exit 95
rm -f "$B/usage-$inst.json" "$B/spans-$inst.jsonl"
cd "$workdir" || exit 98
T1=$(date +%s%3N)
HERMES_BENCH_SPANS="$B/spans-$inst.jsonl" \
  hermes -p "$inst" -t "$toolsets" ${yolo:+--yolo} --usage-file "$B/usage-$inst.json" -z "$instruction"
rc=$?
T2=$(date +%s%3N)
rm -rf "$T" /sandbox/.ssh/config
printf '\n%s\n' "<<<HERMES_USAGE>>>"
cat "$B/usage-$inst.json" 2>/dev/null
printf '\n%s\n{"ev":"script","t0":%s,"t1":%s,"t2":%s}\n' "<<<HERMES_SPANS>>>" "$T0" "$T1" "$T2"
cat "$B/spans-$inst.jsonl" 2>/dev/null
exit $rc
"""

_SETUP_EXIT = {94: 'profile edit', 95: 'ssh client setup', 96: 'profile snapshot has no run-id header',
               97: 'profile restore', 98: 'sandbox workdir'}
# The proxy's stop reasons, as completion types.
_PROXY_STOP = {'call_cap': 'step_limit', 'output_cap': 'truncated', 'timeout': 'timeout',
               'usage_missing': 'accounting_error', 'output_cap_violation': 'accounting_error',
               'concurrency_violation': 'error'}


class NemoClawHermesAgent(BaseAgent):
    def __init__(self, *args, cell_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell_config = Path(cell_config or ROOT / 'terminal_bench/configs/nemoclaw_hermes.yaml')
        self.cfg = RunConfig.from_yaml(self.cell_config)
        h = self.cfg.hermes
        missing = [k for k in ('sandbox', 'profile', 'instance', 'toolsets', 'max_turns', 'workdir',
                               'relay_host', 'relay_port') if not h.get(k)]
        if missing:
            raise ValueError(f'hermes block needs {missing} (see terminal_bench/configs/nemoclaw_hermes.yaml)')
        if int(h['max_turns']) >= self.cfg.limits.max_llm_calls:
            raise ValueError('hermes.max_turns must be below limits.max_llm_calls')
        if self.model_name and self.model_name.removeprefix('openai/') != self.cfg.model.served_name:
            raise ValueError('Harbor model must match the cell served_name')
        if self.mcp_servers or self.skills_dir or self.load_trajectory:
            raise ValueError('This adapter does not implement MCP, skills, or session resume')
        self.h = h

    @staticmethod
    def name():
        return 'nemoclaw-hermes'

    def _registry(self) -> dict:
        try:
            return json.loads((Path.home() / '.nemoclaw/sandboxes.json').read_text())['sandboxes'][self.h['sandbox']]
        except Exception:
            return {}

    def version(self):
        s = self._registry()
        return (f"hermes=={s.get('agentVersion')};nemoclaw=={s.get('nemoclawVersion')};"
                f"openshell=={s.get('openshellVersion')};adapter=1")

    def _exec(self, *command: str) -> list[str]:
        gateway = self._registry().get('gatewayName') or 'nemoclaw'
        return ['openshell', 'sandbox', 'exec', '-g', gateway, '-n', self.h['sandbox'], '--no-tty', '--', *command]

    async def setup(self, environment):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if not isinstance(environment, DockerEnvironment):
            raise ValueError('The NemoClaw relay supports Harbor docker environments only')

    async def _task_container(self, environment) -> tuple[str, str, str, dict]:
        """The main service's container and what environment.exec would add: workdir, user, env."""
        project = _sanitize_docker_compose_project_name(environment.session_id)
        proc = await asyncio.create_subprocess_exec(
            'docker', 'ps', '-q', '--no-trunc', '--filter', f'label=com.docker.compose.project={project}',
            '--filter', f'label=com.docker.compose.service={MAIN_SERVICE_NAME}',
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        ids = out.decode().split()
        if len(ids) != 1:
            raise RuntimeError(f'expected one main container for {project}, found {ids} {err.decode()[-200:]}')
        workdir = environment.task_env_config.workdir or ''
        if not workdir:
            proc = await asyncio.create_subprocess_exec(
                'docker', 'inspect', '-f', '{{.Config.WorkingDir}}', ids[0], stdout=asyncio.subprocess.PIPE)
            workdir = (await proc.communicate())[0].decode().strip()
        user = environment._resolve_user(None)
        return ids[0], workdir, '' if user is None else str(user), environment._merge_env(None) or {}

    async def _proxy(self, path: str, body: dict) -> dict:
        url = self.cfg.base_url.rsplit('/v1', 1)[0] + path
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            response = await client.post(url, json=body)
        if response.status_code != 200:
            raise RuntimeError(f'proxy {path} -> {response.status_code}: {response.text[:300]}')
        return response.json()

    async def _kill_in_sandbox(self):
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._exec('sh', '-c', 'inst=$1\n' + _KILL, 'sh', self.h['instance']),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL, env=self._env())
            await asyncio.wait_for(proc.wait(), 60)
        except Exception:
            pass  # the next trial's script kills it before starting

    @staticmethod
    def _env():
        return dict(os.environ, NEMOCLAW_DISABLE_TELEMETRY='1', NEMOCLAW_NO_AUTO_AUDIT_LOGS='1')

    async def run(self, instruction, environment, context: AgentContext):
        started = time.perf_counter()
        cfg, h = self.cfg, self.h
        container, cwd, user, task_env = await self._task_container(environment)
        run_id = f'tb-{uuid.uuid4().hex[:16]}'
        alias = f'tbrelay-{run_id[3:11]}'
        relay = SSHRelay(container, self.logs_dir / 'relay', workdir=cwd, user=user, env=task_env,
                         host=h['relay_host'], port=int(h['relay_port']))
        proxy_dir = self.logs_dir / 'proxy'
        proxy_dir.mkdir(parents=True, exist_ok=True)
        status, error, rc, out, err, snapshot = 'error', '', None, '', '', {}
        proc, launched, returned = None, 0.0, 0.0
        try:
            relay.start()
            key = base64.b64encode(relay.client_key.read_bytes()).decode()
            await self._proxy('/control/run', {
                'run_id': run_id, 'out_dir': str(proxy_dir), 'session': f'terminal-bench-{self.logs_dir.parents[1].name}',
                'limits': {'max_llm_calls': cfg.limits.max_llm_calls,
                           'max_tokens_per_call': cfg.sampling.max_tokens,
                           'max_total_output_tokens': cfg.limits.max_total_output_tokens,
                           'task_timeout_s': cfg.limits.task_timeout_s}})
            launched = time.time() * 1000
            proc = await asyncio.create_subprocess_exec(
                *self._exec('sh', '-c', _RUN, 'sh', h['instance'], h['profile'], alias, cwd or '/', run_id, key,
                            relay.known_hosts_line(alias), h['toolsets'], str(h['max_turns']), h['workdir'],
                            str(h['relay_port']), getpass.getuser(), '1' if h.get('yolo') else '', instruction),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=self._env())
            out_b, err_b = await proc.communicate()
            out, err, rc = out_b.decode(errors='replace'), err_b.decode(errors='replace'), proc.returncode
            returned = time.time() * 1000
        except asyncio.CancelledError:
            status, error = 'timeout', 'cancelled'
            if proc and proc.returncode is None:
                proc.kill()
            await self._kill_in_sandbox()
            raise
        except Exception as exc:
            status, error = 'error', f'{type(exc).__name__}: {exc}'[:500]
            raise
        finally:
            try:
                snapshot = await self._proxy('/control/end', {'run_id': run_id, 'reason': status if error else 'finished'})
            except Exception as exc:
                snapshot = {'error': f'{type(exc).__name__}: {exc}'[:300]}
            relay.stop()
            answer, _, rest = out.partition(USAGE_MARKER)
            usage_text, _, spans_text = rest.partition(SPANS_MARKER)
            try:
                usage = json.loads(usage_text) if usage_text.strip() else {}
            except json.JSONDecodeError:
                usage = {}
            stderr = '\n'.join(l for l in err.splitlines() if l.strip() and not any(n in l for n in _STDERR_NOISE))
            if not error:
                status, error = self._completion(rc, usage, snapshot, answer, stderr)
            calls = _rows(proxy_dir / 'llm_call.jsonl')
            usage_complete = bool(calls) and not any(c.get('usage_missing') for c in calls)
            context.n_input_tokens = sum(c.get('prompt_tokens') or 0 for c in calls) if usage_complete else None
            context.n_output_tokens = sum(c.get('completion_tokens') or 0 for c in calls) if usage_complete else None
            context.n_cache_tokens = sum(c.get('cached_tokens') or 0 for c in calls) if usage_complete else None
            relay_calls = _rows(self.logs_dir / 'relay/relay_calls.jsonl')
            timing = _timing(spans_text, launched, returned) if rc is not None else {}
            metadata = {
                'runtime': 'nemoclaw-hermes', 'completion_type': status, 'error_type': error,
                'natural_completion': status == 'natural_completion', 'forced_final_answer': False,
                'run_id': run_id, 'hermes_exit_code': rc, 'hermes_api_calls': usage.get('api_calls'),
                'n_llm_calls': len(calls), 'n_relay_commands': len(relay_calls),
                'n_tool_calls': len(timing.get('tool_calls', [])),
                'blocked_llm_requests': snapshot.get('blocked_requests'),
                'proxy_termination_reason': snapshot.get('termination_reason'),
                'usage_complete': usage_complete, 'budget_enforced': True,
                'max_llm_calls': cfg.limits.max_llm_calls, 'max_tokens_per_call': cfg.sampling.max_tokens,
                'max_total_output_tokens': cfg.limits.max_total_output_tokens, 'hermes_max_turns': h['max_turns'],
                'call_cap_reached': snapshot.get('termination_reason') == 'call_cap',
                'total_output_cap_reached': snapshot.get('termination_reason') == 'output_cap',
                'per_call_output_cap_reached': any(c.get('finish_reason') == 'length' for c in calls),
                'agent_wall_ms': (time.perf_counter() - started) * 1000,
                'llm_request_ms': sum(c.get('latency_ms') or 0 for c in calls),
                'terminal_wall_ms': sum((r['t1'] - r['t0']) for r in relay_calls),
                'thinking_setting': cfg.model.thinking_setting, 'reasoning_effort': cfg.model.reasoning_effort,
                'phases': timing.get('phases', []),
                'timing_note': 'llm_request_ms includes server queue/CPU/GPU; terminal_wall_ms is relay time per '
                               'command (ssh mux + docker exec + the command), including Hermes file-sync commands.',
            }
            context.metadata = metadata
            for name, text in [('hermes_stdout.txt', out), ('hermes_stderr.txt', err),
                               ('final_answer.txt', answer.strip()), ('usage.json', json.dumps(usage, indent=2)),
                               ('spans.jsonl', spans_text.strip()), ('accounting.json', json.dumps(metadata, indent=2)),
                               ('tool_calls.json', json.dumps(timing.get('tool_calls', []), indent=2))]:
                (self.logs_dir / name).write_text(text)

    @staticmethod
    def _completion(rc, usage, snapshot, answer, stderr) -> tuple[str, str]:
        reason = snapshot.get('termination_reason')
        if reason in _PROXY_STOP:
            return _PROXY_STOP[reason], f'proxy stopped the run: {reason}'
        if rc in _SETUP_EXIT:
            return 'error', f'sandbox setup failed: {_SETUP_EXIT[rc]}'
        if rc != 0 or usage.get('failed'):
            # On failure Hermes exits 0 and prints its error as the answer.
            detail = stderr[-300:] if rc else answer.strip()[:300]
            return 'error', f'hermes exit {rc}: {detail}'
        if not usage.get('completed', True):
            return 'step_limit', 'hermes agent.max_turns'
        return 'natural_completion', ''


def _rows(path: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except (OSError, json.JSONDecodeError):
        return []
