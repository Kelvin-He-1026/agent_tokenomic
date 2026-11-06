#!/usr/bin/env bash
set -euo pipefail
TB_DIR="$(cd "$(dirname "$0")/.." && pwd)"
python3 -c 'import sys; assert sys.version_info >= (3, 12), "Python >=3.12 required"'
if [[ ! -x "$TB_DIR/.venv/bin/python" ]]; then
  python3 -m venv "$TB_DIR/.venv"
fi
"$TB_DIR/.venv/bin/python" -m pip install --disable-pip-version-check \
  --index-url https://pypi.org/simple -r "$TB_DIR/requirements.lock.txt"
"$TB_DIR/.venv/bin/python" -m pip check
