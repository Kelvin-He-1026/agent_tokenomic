"""GAIA sweep, unattended: Qwen TP=2, thinking off / low / xhigh, LangGraph +
OpenAI Agents SDK, 1 repeat, tasks from GAIA_stack/tasks/phase1-GAIA-text.json
picked per level by --mix (file order, so a smaller mix is a subset of a larger).

Two methods:

  live-record (default)  one sweep, one run at a time, power recorded, tools in
              record mode: a search or page already recorded (same or reworded
              query) is replayed, anything new is fetched live and saved. Every
              run gets real tool results. Tool time is live, but LLM time and
              energy are measured as in replay, and tools are ~1% of GPU energy.
  record-then-replay     a parallel recording pass per mode, then a clean replay
              sweep. Fine for short Phase 1 runs; on GAIA's 8-30 step runs the
              replay drifts away from what was recorded (2026-10-21: ~40% of
              searches matched only approximately, ~35% of page fetches had no
              recording), so it is kept only for comparison.

Then export, energy by phase, energy-method check and summary.md (no judge:
GAIA answers are scored exactly against the reference).

    nohup .venv/bin/python -u GAIA_stack/run_gaia.py --mix easy=10,medium=10,hard=10 > /dev/null 2>&1 &

Progress: GAIA_stack/status.sh. Log: output/run_gaia.log; sweep logs
output/sweep_gaia_*.log; results output/runs/gaia_<method>_<mix>_<stamp>/.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PY = str(ROOT / ".venv/bin/python")
CONFIG = "GAIA_stack/configs/gaia_qwen_tp2.yaml"
TASKS = "GAIA_stack/tasks/phase1-GAIA-text.json"
MODES = ["off", "low", "xhigh"]
FRAMEWORKS = "langgraph,openai_agents"
LOG = ROOT / "output/run_gaia.log"
PID = ROOT / "output/run_gaia.pid"


def log(msg: str) -> None:
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%m-%d %H:%M:%S")
    with LOG.open("a") as fh:
        fh.write(f"{stamp} {msg}\n")


def tavily_left() -> int | None:
    from agent.env import load_env

    load_env()
    try:
        r = requests.get("https://api.tavily.com/usage", timeout=15,
                         headers={"Authorization": f"Bearer {os.environ.get('TAVILY_API_KEY', '')}"})
        acct = r.json()["account"]
        left = (acct.get("plan_limit") or 0) - (acct.get("plan_usage") or 0)
        if acct.get("paygo_limit"):
            left += acct["paygo_limit"] - (acct.get("paygo_usage") or 0)
        return left
    except Exception as exc:  # noqa: BLE001 - logged, and treated as unknown
        log(f"could not read Tavily usage: {exc}")
        return None


def run(cmd: list[str], log_name: str) -> int:
    with (ROOT / "output" / log_name).open("w") as fh:
        return subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="easy=10,medium=10,hard=10", help="tasks per level")
    ap.add_argument("--method", choices=["live-record", "record-then-replay"], default="live-record")
    ap.add_argument("--credits-per-mode", type=int, default=500,
                    help="record-then-replay with plain tavily only: credits one mode may need")
    args = ap.parse_args()
    import yaml

    from agent.telemetry import file_stamp

    PID.write_text(str(os.getpid()))
    backend = yaml.safe_load((ROOT / CONFIG).read_text()).get("web_search_backend", "tavily")
    size = sum(int(p.split("=")[1]) for p in args.mix.split(","))
    tag = f"{args.method.replace('-', '_')}_{size}"
    root = ROOT / "output/runs" / f"gaia_{tag}_{file_stamp()}"
    common = ["--base-config", CONFIG, "--tasks", TASKS, "--frameworks", FRAMEWORKS,
              "--difficulty-mix", args.mix, "--repeats", "1"]
    exhausted = (ROOT / "output/.tavily_exhausted").exists()
    log(f"GAIA sweep ({args.method}): Qwen TP=2, modes {MODES}, {args.mix}, {FRAMEWORKS}, "
        f"web_search via {backend}{' (Tavily used up: DuckDuckGo)' if exhausted else ''} -> {root}")

    if args.method == "live-record":
        log("sweep: one run at a time, power recorded, new searches and pages fetched live")
        rc = run([PY, "-u", "scripts/sweep_thinking.py", *common, "--modes", ",".join(MODES),
                  "--tool-mode", "record", "--out-root", str(root.relative_to(ROOT)),
                  "--run-code", f"gaia-{tag}"], f"sweep_gaia_{tag}.log")
    else:
        for mode in MODES:
            left = tavily_left() if backend.startswith("tavily") else None
            if backend == "tavily" and left is not None and left < args.credits_per_mode:
                log(f"STOP before recording {mode}: {left} Tavily credits left, "
                    f"need ~{args.credits_per_mode}")
                return 1
            log(f"recording {mode}" + (f" ({left} Tavily credits left)" if left is not None else ""))
            rc = run([PY, "-u", "scripts/sweep_thinking.py", *common, "--modes", mode,
                      "--tool-mode", "record", "--workers", "6", "--run-code", f"gaia-{tag}-record"],
                     f"sweep_gaia_{tag}_record_{mode}.log")
            if rc:
                log(f"STOP: recording {mode} failed (exit {rc})")
                return 1
        log("clean sweep (replay, one run at a time, power recorded)")
        rc = run([PY, "-u", "scripts/sweep_thinking.py", *common, "--modes", ",".join(MODES),
                  "--out-root", str(root.relative_to(ROOT)), "--run-code", f"gaia-{tag}"],
                 f"sweep_gaia_{tag}.log")
    log(f"sweep exited with code {rc}")
    subprocess.run(["./scripts/stop_stack.sh", "--all"], cwd=ROOT, capture_output=True)

    modes = [str(root / m) for m in MODES if (root / m / "run.jsonl").exists()]
    if not modes:
        log("no runs found; stopping")
        return 1
    steps = [
        ("export", [PY, "scripts/export_csv.py", "--runs", *modes, "--tasks", TASKS,
                    "--out", str(root / "results.csv"), "--no-timestamp", "--calls"]),
        ("energy by phase", [PY, "scripts/attribute_energy.py", "--runs", *modes,
                             "--idle-w", "88", "--out", str(root / "energy.csv")]),
        ("energy method check", [PY, "scripts/verify_energy.py", "--runs", *modes,
                                 "--out", str(root / "energy_verification.json")]),
        ("summary", [PY, "scripts/summarize_results.py", "--results", str(root / "results.csv"),
                     "--energy", str(root / "energy.csv"), "--out", str(root / "summary.md")]),
    ]
    for name, cmd in steps:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        last = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()][-1:] or [""]
        log(f"{name}: {'ok' if p.returncode == 0 else 'FAILED'} {last[0][:160]}")
    log(f"done: {root}/summary.md")
    PID.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
