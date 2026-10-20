"""GAIA sweep: Qwen TP=2, thinking off / low / xhigh, 10 L1 / 15 L2 / 10 L3 tasks
from GAIA_stack/tasks/phase1-GAIA-text.json, LangGraph + OpenAI Agents SDK, 1 repeat, unattended.

  1. record   one probe sweep per mode (parallel workers, timings not clean) to
              fill web_search / fetch_url fixtures. The GAIA config searches
              with Tavily until its credits run out, then DuckDuckGo (free).
              With plain tavily, each mode first checks that the remaining
              credits cover --credits-per-mode,
              since a search that fails for lack of credit is not recorded and
              would turn into a tool failure in the clean sweep.
  2. sweep    clean replay sweep of all modes, one run at a time, power recorded
  3. results  export, energy by phase, energy-method check, summary.md
              (no judge: GAIA answers are scored exactly against the reference)

    nohup .venv/bin/python -u GAIA_stack/run_gaia_35.py > /dev/null 2>&1 &

Logs go to output/ (run_gaia_35.log, sweep_gaia35*.log); results to output/runs/.
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
MIX = "easy=10,medium=15,hard=10"
FRAMEWORKS = "langgraph,openai_agents"
COMMON = ["--base-config", CONFIG, "--tasks", TASKS, "--frameworks", FRAMEWORKS,
          "--difficulty-mix", MIX, "--repeats", "1"]
LOG = ROOT / "output/run_gaia_35.log"
PID = ROOT / "output/run_gaia_35.pid"


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


def newest(pattern: str, after: float) -> Path | None:
    dirs = [d for d in (ROOT / "output/runs").glob(pattern) if d.stat().st_mtime >= after]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--credits-per-mode", type=int, default=500,
                    help="Tavily backend only: credits one mode's recording may need")
    ap.add_argument("--skip-record", action="store_true")
    args = ap.parse_args()
    PID.write_text(str(os.getpid()))
    import yaml

    backend = yaml.safe_load((ROOT / CONFIG).read_text()).get("web_search_backend", "tavily")
    log(f"GAIA sweep: Qwen TP=2, modes {MODES}, {MIX}, {FRAMEWORKS}, web_search via {backend}")

    if not args.skip_record:
        for mode in MODES:
            left = tavily_left() if backend.startswith("tavily") else None
            if backend == "tavily" and left is not None and left < args.credits_per_mode:
                log(f"STOP before recording {mode}: {left} Tavily credits left, "
                    f"need ~{args.credits_per_mode}. Add credits, then rerun with the modes left")
                return 1
            log(f"recording {mode}" + (f" ({left} Tavily credits left)" if left is not None else ""))
            rc = run([PY, "-u", "scripts/sweep_thinking.py", *COMMON, "--modes", mode,
                      "--tool-mode", "record", "--workers", "6", "--run-code", "gaia35-record"],
                     f"sweep_gaia35_record_{mode}.log")
            if rc:
                log(f"STOP: recording {mode} failed (exit {rc}); see output/sweep_gaia35_record_{mode}.log")
                return 1
            after = tavily_left() if backend.startswith("tavily") else None
            if backend == "tavily" and after is not None and after <= 0:
                log(f"STOP after recording {mode}: Tavily credits ran out, so some searches "
                    "may be missing; add credits and re-record this mode")
                return 1
            log(f"recorded {mode}" + (f": {(left or 0) - after} Tavily credits used, {after} left"
                                      if after is not None else ""))

    started = time.time()
    log("clean sweep (replay, one run at a time, power recorded)")
    rc = run([PY, "-u", "scripts/sweep_thinking.py", *COMMON, "--modes", ",".join(MODES),
              "--run-code", "gaia35-qwen-tp2"], "sweep_gaia35.log")
    log(f"sweep exited with code {rc}")
    subprocess.run(["./scripts/stop_stack.sh", "--all"], cwd=ROOT, capture_output=True)

    root = newest("thinking_sweep_Qwen3.8-27B_*", started)
    if root is None:
        log("no sweep folder found; stopping")
        return 1
    modes = [str(root / m) for m in MODES if (root / m / "run.jsonl").exists()]
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
