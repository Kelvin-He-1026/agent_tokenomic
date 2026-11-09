"""Build a pinned Harbor job. Preview by default; --execute starts containers."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import ipaddress
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TB = ROOT / 'terminal_bench'


def configure_local_docker():
    # Preserve an explicitly selected Docker context/config. Our generated
    # config contains only CLI plugin paths, no credentials or daemon changes.
    if (TB / '.docker/config.json').exists():
        os.environ.setdefault('DOCKER_CONFIG',str(TB / '.docker'))


def local_endpoint(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Use an HTTP(S) local model endpoint without embedded credentials')
    if parsed.query or parsed.fragment or parsed.path.rstrip('/') != '/v1':
        raise ValueError('Model endpoint must end in /v1 without query or fragment')
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 80, type=socket.SOCK_STREAM)
    if not addresses or not all(ipaddress.ip_address(a[4][0]).is_private for a in addresses):
        raise ValueError('This study entry point accepts private/local inference endpoints only')
    return url.rstrip('/')


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=['oracle', 'react', 'nemoclaw-hermes', 'hermes-native'], default='oracle')
    parser.add_argument('--suite', choices=['smoke', 'pilot'], default='smoke')
    parser.add_argument('--task', action='append', help='Exact task name; repeat to replace the named suite')
    parser.add_argument('--attempts', type=int, help='Default: oracle=5, agents=3')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--task-cpus', type=float, help='Task container CPU quota; not host/vLLM CPU allocation')
    parser.add_argument('--memory-mb', type=int)
    parser.add_argument('--agent-timeout', type=float, help='Research override; default from cell config for ReAct, task default for oracle/native Hermes')
    parser.add_argument('--cell-config', type=Path, help='Default: configs/nemoclaw_hermes.yaml for nemoclaw-hermes, else configs/qwen_tp2.yaml')
    parser.add_argument('--base-url', help='Host-reachable local Qwen URL; overrides cell config for ReAct')
    parser.add_argument('--container-base-url', help='For native Hermes: explicitly supply a task-container-reachable model endpoint')
    parser.add_argument('--hermes-version', default='v2026.9.24', help='Verified upstream Hermes tag, not a NemoClaw image version')
    parser.add_argument('--local-tasks', type=Path, help='Optional local Harbor dataset, e.g. the offline smoke task')
    parser.add_argument('--job-name')
    parser.add_argument('--execute', action='store_true', help='Download tasks, create containers and run this job')
    return parser


def build_job(args):
    from agent.config import RunConfig
    from harbor.models.job.config import JobConfig

    suites = json.loads((TB / 'configs/suites.json').read_text())
    tasks = args.task or suites[args.suite]
    if not tasks or len(tasks) != len(set(tasks)) or any(any(c in t for c in '*?[]/') for t in tasks):
        raise ValueError('Supply distinct, exact task names')
    attempts = args.attempts if args.attempts is not None else (5 if args.agent == 'oracle' else 3)
    if attempts < 1 or args.concurrency < 1:
        raise ValueError('Attempts and concurrency must be positive')
    if args.task_cpus is not None and args.task_cpus <= 0:
        raise ValueError('CPU quota must be positive')
    if args.memory_mb is not None and args.memory_mb <= 0:
        raise ValueError('Memory must be positive')
    if args.agent_timeout is not None and args.agent_timeout <= 0:
        raise ValueError('Timeout must be positive')
    name = args.job_name or f"{args.agent}_{args.suite}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    if Path(name).name != name or name in {'.','..'}:
        raise ValueError('job-name must be a single path component')
    environment = {'type':'docker','delete':True,'cpu_enforcement_policy':'limit',
                   'memory_enforcement_policy':'limit','override_gpus':0}
    if args.task_cpus is not None:
        environment['override_cpus'] = args.task_cpus
    if args.memory_mb is not None:
        environment['override_memory_mb'] = args.memory_mb
    if args.agent == 'nemoclaw-hermes' and args.concurrency != 1:
        raise ValueError('nemoclaw-hermes runs one trial at a time: one sandbox profile instance and one relay port')
    cell_config = args.cell_config or TB / ('configs/nemoclaw_hermes.yaml' if args.agent == 'nemoclaw-hermes'
                                            else 'configs/qwen_tp2.yaml')
    cfg = RunConfig.from_yaml(cell_config)
    endpoint = local_endpoint(args.base_url or cfg.base_url)
    cell = cfg.to_dict()
    cell['base_url'] = endpoint
    if cfg.hermes:
        # to_dict() leaves provenance fields out; the adapter needs this block.
        cell['hermes'] = dict(cfg.hermes)
    if cell.get('api_key') != 'EMPTY':
        raise ValueError('Keep api_key: EMPTY in committed config; use TB_API_KEY environment variable')
    if args.agent_timeout is not None:
        cell['limits']['task_timeout_s'] = args.agent_timeout
    # A generated, non-secret snapshot is imported by every ReAct trial.
    generated_cell = TB / 'output/plans' / f'{name}.cell.json'
    if args.agent == 'react':
        agent = {'import_path':'terminal_bench.agents:ProjectReActAgent',
                 'model_name':cfg.model.served_name,
                 'override_timeout_sec':cell['limits']['task_timeout_s'],
                 'kwargs':{'cell_config':str(generated_cell)}}
    elif args.agent == 'nemoclaw-hermes':
        agent = {'import_path':'terminal_bench.nemoclaw_agent:NemoClawHermesAgent',
                 'model_name':cfg.model.served_name,
                 'override_timeout_sec':cell['limits']['task_timeout_s'],
                 'kwargs':{'cell_config':str(generated_cell)}}
    elif args.agent == 'hermes-native':
        if not args.container_base_url:
            raise ValueError('Native Hermes needs --container-base-url reachable INSIDE its task container; loopback host URL is not enough')
        container_endpoint = args.container_base_url.rstrip('/')
        parsed = urlsplit(container_endpoint)
        if parsed.hostname in {'127.0.0.1','localhost','::1'}:
            raise ValueError('Native Hermes cannot use host loopback inside Docker')
        # Resolve/validate on execution host when possible; a Docker-only alias
        # may legitimately be unresolvable here. IP addresses are always checked.
        if parsed.scheme not in {'http','https'} or parsed.username or parsed.password or parsed.path != '/v1' or parsed.query or parsed.fragment:
            raise ValueError('Invalid container endpoint')
        try:
            addr = ipaddress.ip_address(parsed.hostname or '')
        except ValueError:
            if parsed.hostname not in {'host.docker.internal','host.openshell.internal'}:
                raise ValueError('Use a private bridge IP or explicit Docker host alias')
        else:
            if not addr.is_private:
                raise ValueError('Native Hermes endpoint must be private/local')
        agent = {'name':'hermes','model_name':'openai/' + cfg.model.served_name,
                 'base_url':container_endpoint,'model_api_format':'openai_chat_completions',
                 'env':{'OPENAI_API_KEY':'EMPTY'},
                 'kwargs':{'version':args.hermes_version,'toolsets':'terminal,file',
                           'disable_web_search':True,'max_turns':cfg.limits.max_llm_calls}}
        if args.agent_timeout is not None:
            agent['override_timeout_sec'] = args.agent_timeout
    else:
        agent = {'name':'oracle'}
    lock_path = TB / 'configs/dataset.lock.json'
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else None
    if lock and not args.local_tasks and any(t not in lock['tasks'] for t in tasks):
        raise ValueError('A selected task is not in the pinned CPU-only dataset')
    dataset = {'name':suites['dataset'],
               **({'ref':lock['content_ref']} if lock else {'version':suites['version']}),
               'task_names':['terminal-bench/' + t for t in tasks],
               'download_dir':str(TB / 'data/tasks')}
    if args.local_tasks:
        dataset = {'path':str(args.local_tasks.resolve()),'task_names':tasks}
    raw = {'job_name':name,'jobs_dir':str(TB / 'output/jobs'),'n_attempts':attempts,
           'n_concurrent_trials':args.concurrency,'retry':{'max_retries':0},
           'agents':[agent],'environment':environment,'datasets':[dataset]}
    job = JobConfig.model_validate(raw)
    plan = {'job_name':name,'dataset':suites['dataset'] if not args.local_tasks else str(args.local_tasks.resolve()),
            'dataset_version':suites['version'] if not args.local_tasks else 'local-smoke',
            'dataset_content_ref':lock['content_ref'] if lock and not args.local_tasks else None,
            'task_names':tasks,'n_attempts':attempts,'expected_trials':len(tasks)*attempts,
            'agent':args.agent,'n_concurrent_trials':args.concurrency,
            'task_cpus':args.task_cpus,'memory_mb':args.memory_mb,
            'study_protocol':'local-resource-study; not official leaderboard parity',
            'budget_enforced':args.agent in ('react', 'nemoclaw-hermes'),
            'native_hermes_note':'Harbor installed Hermes; NOT NemoClaw/OpenShell. max_turns is not a proxy-enforced call/token budget.' if args.agent == 'hermes-native' else '',
            'harbor_version':version('harbor'),
            'cell_sha256':hashlib.sha256(json.dumps(cell,sort_keys=True).encode()).hexdigest(),
            'job_config':job.model_dump(mode='json')}
    return job, plan, cell, generated_cell


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        job, plan, cell, cell_path = build_job(args)
        if args.agent == 'react' and cell['limits']['max_total_output_tokens'] <= 0:
            raise ValueError('ReAct requires an explicit positive total output token cap')
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc))
    if version('harbor') != '0.25.0':
        raise SystemExit('Use terminal_bench/.venv and the pinned requirements')
    cell_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path = cell_path.with_name(plan['job_name'] + '.json')
    if plan_path.exists() or cell_path.exists() or (TB / 'output/jobs' / plan['job_name']).exists():
        raise SystemExit('Job name already exists; choose a new name to preserve prior data')
    cell_path.write_text(json.dumps(cell, indent=2))
    plan_path.write_text(json.dumps(plan, indent=2))
    print(json.dumps({k:v for k,v in plan.items() if k != 'job_config'},indent=2))
    print('Plan:', plan_path)
    if not args.execute:
        print('Preview only. No task downloads, containers or model requests. Add --execute with a NEW job name to run.')
        return 0
    configure_local_docker()
    for command in [['docker','info'],['docker','compose','version'],['docker','buildx','version']]:
        subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
    if args.agent != 'oracle':
        import httpx
        with httpx.Client(timeout=10,trust_env=False) as client:
            response = client.get(cell['base_url'] + '/models')
            response.raise_for_status()
            ids = {r['id'] for r in response.json().get('data',[])}
        if cell['model']['served_name'] not in ids:
            raise SystemExit('Expected Qwen served_name not found at local endpoint')
    if args.agent == 'nemoclaw-hermes':
        sandbox = cell['hermes']['sandbox']
        policy = subprocess.run(['openshell','policy','get',sandbox,'--full'],capture_output=True,text=True,
                                timeout=60,stdin=subprocess.DEVNULL)
        if 'terminal-bench-relay' not in policy.stdout:
            raise SystemExit(f'Sandbox {sandbox} lacks the relay egress rule: nemohermes {sandbox} policy add '
                             '--from-file terminal_bench/configs/policies/terminal-bench-relay.yaml --yes')
        # Every trial binds the logging proxy under this job's session. A run left open by a
        # killed job, or another job's session lock (held 10 min after its last bind), would
        # refuse every trial with 409, so check first and release the lock when done.
        control = cell['base_url'].rsplit('/v1',1)[0] + '/control'
        with httpx.Client(timeout=10,trust_env=False) as client:
            health = client.get(control + '/health').json()
        if health.get('active_runs') or health.get('inflight'):
            raise SystemExit(f'Proxy has an open run ({health.get("run_id")}, out_dir {health.get("out_dir")}). '
                             f'End it first: curl -X POST {control}/end -d \'{{"run_id":"<id>"}}\'')
    from harbor.job import Job

    async def execute():
        instance = await Job.create(job)
        plan['resolved_job_config'] = instance.config.model_dump(mode='json')
        plan_path.write_text(json.dumps(plan,indent=2))
        return await instance.run()

    try:
        asyncio.run(execute())
    finally:
        if args.agent == 'nemoclaw-hermes':
            try:
                with httpx.Client(timeout=10,trust_env=False) as client:
                    client.post(control + '/release', json={'session': f'terminal-bench-{plan["job_name"]}'})
            except Exception as exc:
                print(f'Could not release the proxy session: {exc}', file=sys.stderr)
    print('Raw job:', TB / 'output/jobs' / plan['job_name'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
