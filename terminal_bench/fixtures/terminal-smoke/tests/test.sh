#!/usr/bin/env bash
set -euo pipefail
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
python3 - <<'PY'
from pathlib import Path
assert Path('/app/result.txt').read_text() == '338350\n'
PY
echo 1 > /logs/verifier/reward.txt
