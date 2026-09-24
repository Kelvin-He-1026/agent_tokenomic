"""Re-derive how each run ended, for run directories written before the fix.

Applies the runner's own _classify_ending to stored runs, so old and new data
are classified by exactly the same rule. The original run.jsonl is kept as
run.jsonl.bak; nothing else is modified.

    python scripts/reclassify_runs.py output/runs/thinking_sweep_*/{off,low,medium,xhigh}
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.harness.runner import Runner  # noqa: E402
from agent.telemetry import RunRecord  # noqa: E402


def main() -> int:
    for d in map(Path, sys.argv[1:]):
        runs_path = d / "run.jsonl"
        if not runs_path.exists():
            continue
        calls: dict[str, list[dict]] = {}
        for line in (d / "llm_call.jsonl").read_text().splitlines():
            c = json.loads(line)
            calls.setdefault(c["run_id"], []).append(c)

        out, changed = [], []
        for line in runs_path.read_text().splitlines():
            row = json.loads(line)
            rec = RunRecord(**{k: v for k, v in row.items() if k in RunRecord.__dataclass_fields__})
            before = rec.status
            Runner._classify_ending(rec, calls.get(rec.run_id, []))
            if rec.status != before:
                row["status"], row["error"] = rec.status, rec.error
                changed.append(f"{row['task_id']}: {before} -> {rec.status}")
            out.append(json.dumps(row, separators=(",", ":")))

        if changed:
            shutil.copy(runs_path, runs_path.with_suffix(".jsonl.bak"))
            runs_path.write_text("\n".join(out) + "\n")
        print(f"{d.name:8s} {len(changed)} reclassified" + "".join(f"\n    {c}" for c in changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
