"""Record web_search and fred_releases responses once, then replay them forever.

Live APIs drift in content and latency, which would land in a framework
comparison as noise. This records a fixture per (tool, arguments) pair; runs
afterwards read only from disk.

Needs TAVILY_API_KEY and FRED_API_KEY. The FRED window is computed from the
config's frozen as_of_date, not from today, so re-recording later reproduces
the same calendar window.

    python scripts/record_fixtures.py --config configs/pilot_qwen.yaml \
        --queries dataset/fixture_queries.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import RunConfig  # noqa: E402
from agent.env import load_env, redact, report  # noqa: E402
from agent.tools import impl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--queries",
        help="JSON list of {tool, args}; defaults to both FRED windows only",
    )
    ap.add_argument("--force", action="store_true", help="re-record existing fixtures")
    args = ap.parse_args()

    loaded = load_env()
    print(f"loaded {len(loaded)} entries from .env")
    report(loaded)
    print()

    cfg = replace(RunConfig.from_yaml(args.config), tool_mode="record")
    store = impl.FixtureStore(cfg.resolve(cfg.fixtures_dir))

    if args.queries:
        wanted = json.loads(Path(args.queries).read_text())
    else:
        wanted = [
            {"tool": "fred_releases", "args": {"window": "this_week"}},
            {"tool": "fred_releases", "args": {"window": "next_week"}},
        ]

    recorded = skipped = failed = 0
    for item in wanted:
        tool, tool_args = item["tool"], item["args"]
        fid = impl.fixture_id(tool, tool_args)
        if not args.force and store.path(tool, fid).exists():
            print(f"skip    {tool:14s} {json.dumps(tool_args)} ({fid})")
            skipped += 1
            continue
        try:
            result, hit = impl.execute(tool, tool_args, cfg, store)
            saved = hit.fixture_id if hit else "(not saved)"
            print(f"record  {tool:14s} {json.dumps(tool_args)} -> {saved} ({len(result)} chars)")
            recorded += 1
        except impl.ToolFailure as exc:
            print(redact(f"FAIL    {tool:14s} {json.dumps(tool_args)}: {exc}"))
            failed += 1

    print(f"\n{recorded} recorded, {skipped} skipped, {failed} failed -> {store.root}")
    print("as_of_date =", cfg.as_of_date, "(FRED windows are relative to this, not today)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
