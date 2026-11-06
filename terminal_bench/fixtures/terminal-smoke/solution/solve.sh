#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
Path('/app/result.txt').write_text(str(sum(i*i for i in range(1,101)))+'\n')
PY
