"""Read-only checks; never installs, starts, stops, or repairs services."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell-config',type=Path,default=ROOT/'terminal_bench/configs/qwen_tp2.yaml')
    parser.add_argument('--skip-model',action='store_true',help='Oracle setup does not need a model server')
    args = parser.parse_args()
    checks = {'python':sys.version.split()[0]}
    ok = True
    from terminal_bench.run import configure_local_docker
    configure_local_docker()
    for package,expected in [('harbor','0.25.0'),('langgraph','1.2.14'),('langchain-openai','1.7.1'),('langchain-core','1.6.10')]:
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = 'missing'
        checks[package] = {'actual':actual,'expected':expected,'ok':actual==expected}
        ok &= actual == expected
    if shutil.which('docker'):
        result = subprocess.run(['docker','info','--format','{{json .ServerVersion}}'],capture_output=True,text=True,timeout=15)
        checks['docker'] = {'ok':result.returncode==0,'server_version':result.stdout.strip(),
                            'error':result.stderr.strip()[:400]}
        ok &= result.returncode == 0
        for plugin in ['compose','buildx']:
            result = subprocess.run(['docker',plugin,'version'],capture_output=True,text=True,timeout=15)
            checks['docker_'+plugin] = {'ok':result.returncode==0,'version':result.stdout.strip(),
                                        'error':result.stderr.strip()[:400]}
            ok &= result.returncode == 0
    else:
        checks['docker'] = {'ok':False,'error':'docker not found'}
        ok = False
    from agent.config import RunConfig
    cfg = RunConfig.from_yaml(args.cell_config)
    if not args.skip_model:
        import httpx
        try:
            with httpx.Client(timeout=10,trust_env=False) as client:
                response = client.get(cfg.base_url.rstrip('/')+'/models')
                response.raise_for_status()
                ids = [d['id'] for d in response.json().get('data',[])]
            checks['model'] = {'ok':cfg.model.served_name in ids,'endpoint':cfg.base_url,'model_ids':ids}
        except Exception as exc:
            checks['model'] = {'ok':False,'endpoint':cfg.base_url,'error':type(exc).__name__}
        ok &= checks['model']['ok']
    checks['notes'] = ['Readiness is not a passed oracle/model trial.',
                       'Task CPU quota does not pin physical cores or limit the host-side agent/vLLM CPU pool.',
                       'No containers, downloads or model inference were started by this check.']
    print(json.dumps({'ready':ok,'checks':checks},indent=2))
    return 0 if ok else 1


if __name__=='__main__':
    raise SystemExit(main())
