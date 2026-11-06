"""Install checksum-pinned Docker CLI plugins inside this project only."""
from __future__ import annotations

import hashlib
import json
import platform
import urllib.request
from pathlib import Path

TB = Path(__file__).resolve().parents[1]
PLUGINS = [
    ('docker-compose','https://github.com/docker/compose/releases/download/v5.7.0/docker-compose-linux-x86_64',
     '7758048ad38e580f22559603312874fec8bb638d021a9e64182385fc17cc12a2'),
    ('docker-buildx','https://github.com/docker/buildx/releases/download/v0.38.2/buildx-v0.38.2.linux-amd64',
     '7992ec09413b80ff57ba4e081f56fc833d7bac1d5e71143c6c809e7e6065f6d6'),
]


def main():
    if platform.system() != 'Linux' or platform.machine() not in {'x86_64','amd64'}:
        raise SystemExit('This pinned helper supports Linux x86_64 only; install official plugins for your platform')
    config = TB / '.docker'
    folder = config / 'cli-plugins'
    folder.mkdir(parents=True,exist_ok=True)
    for name,url,expected in PLUGINS:
        path=folder/name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==expected:
            print(name,'already verified');continue
        temporary=path.with_suffix('.download')
        digest=hashlib.sha256()
        with urllib.request.urlopen(url,timeout=60) as source, temporary.open('wb') as target:
            while chunk:=source.read(1024*1024):
                target.write(chunk);digest.update(chunk)
        if digest.hexdigest()!=expected:
            temporary.unlink()
            raise SystemExit(f'{name}: checksum mismatch')
        temporary.chmod(0o755);temporary.replace(path)
        print(name,'installed and SHA-256 verified')
    path=config/'config.json'
    desired={'cliPluginsExtraDirs':[str(folder)]}
    if path.exists() and json.loads(path.read_text())!=desired:
        raise SystemExit('Preserving an existing custom Docker configuration; merge cliPluginsExtraDirs manually')
    path.write_text(json.dumps(desired,indent=2)+'\n')
    print('Project-only Docker config:',config)


if __name__=='__main__':
    main()
