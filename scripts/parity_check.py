"""Pilot gate 2: prove the frameworks send the same thing.

Runs one task through each framework and diffs the first request body the proxy
logged. Anything that differs is either fixed in the adapter or written into the
residuals table with its token cost measured. This is what turns "same
orchestration" from an assertion into a check.

    python scripts/parity_check.py --config configs/pilot_mock.yaml \
        --tasks dataset/tasks/pilot.json --frameworks langgraph,crewai
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.adapters import get_adapter  # noqa: E402
from agent.config import RunConfig  # noqa: E402
from agent.harness.runner import Runner, Task  # noqa: E402

# Fields that legitimately differ and are not parity failures.
IGNORED = {"messages", "stream_options", "user"}


def first_request(out_dir: Path, run_id: str) -> dict:
    p = out_dir / "blobs" / run_id / "001_request.json"
    if not p.exists():
        raise SystemExit(f"no logged request for {run_id}; is the proxy running?")
    return json.loads(p.read_text())


def compare(payloads: dict[str, dict]) -> int:
    names = list(payloads)
    reference = names[0]
    failures = 0

    keys = sorted({k for p in payloads.values() for k in p} - IGNORED)
    print(f"\n{'field':24s} " + " ".join(f"{n:22s}" for n in names))
    print("-" * (25 + 23 * len(names)))
    for key in keys:
        values = {n: payloads[n].get(key) for n in names}
        same = all(_norm(v) == _norm(values[reference]) for v in values.values())
        flag = "  " if same else "XX"
        shown = " ".join(f"{_short(values[n]):22s}" for n in names)
        print(f"{flag} {key:22s} {shown}")
        if not same:
            failures += 1

    # The system prompt is inside messages, so it is compared on its own.
    systems = {n: _system(payloads[n]) for n in names}
    same_system = len(set(systems.values())) == 1
    print(f"{'  ' if same_system else 'XX'} system_prompt          "
          + " ".join(f"{s[:22]:22s}" for s in (f"len={len(v)}" for v in systems.values())))
    if not same_system:
        failures += 1
        ref = systems[reference]
        for n in names[1:]:
            if systems[n] != ref:
                print(f"\n--- {reference} vs {n} system prompt ---")
                print(f"{reference}: {ref[:400]!r}")
                print(f"{n}: {systems[n][:400]!r}")

    print(f"\n{'PARITY OK' if failures == 0 else f'PARITY FAILED: {failures} field(s) differ'}")
    return failures


def _norm(value):
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value


def _short(value) -> str:
    if isinstance(value, list):
        if value and isinstance(value[0], dict) and "function" in value[0]:
            return f"[{len(value)} tools]"
        return f"[{len(value)} items]"
    return str(value)[:22]


def _system(payload: dict) -> str:
    for m in payload.get("messages", []):
        if m.get("role") == "system":
            return str(m.get("content", ""))
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--frameworks", default="langgraph,crewai,openai_agents")
    ap.add_argument("--out", default="output/runs/parity")
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    task = Task.load_all(args.tasks)[0]
    out_dir = Path(args.out)

    payloads: dict[str, dict] = {}
    for framework in [f.strip() for f in args.frameworks.split(",") if f.strip()]:
        cell = replace(cfg, framework=framework)
        runner = Runner(cell, get_adapter(framework, cell), out_dir)
        record = runner.run_task(task, repeat_idx=0, run_order_idx=0)
        print(f"{framework:16s} status={record.status:14s} "
              f"llm={record.n_llm_calls} tools={record.n_tool_calls} "
              f"config_id={runner.config_id}")
        if record.status == "framework_error":
            print(f"  error: {record.error}")
            continue
        payloads[framework] = first_request(out_dir, record.run_id)

    if len(payloads) < 2:
        print("\nneed at least two working frameworks to compare")
        return 1
    return 1 if compare(payloads) else 0


if __name__ == "__main__":
    raise SystemExit(main())
