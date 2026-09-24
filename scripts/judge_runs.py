"""Score stored runs with DeepEval, judged by Claude through Claude Code.

Offline by design: runs are never re-executed, the judge sees only what the
proxy and tool layer recorded, and results go to each run directory's
eval.jsonl, the telemetry table reserved for them. It is resumable: a (run,
metric) pair already scored is skipped, so an interrupted pass, or one halted
by subscription usage limits, continues where it stopped.

    # pilot: the time-sensitive web tasks the deterministic scorer cannot judge
    python scripts/judge_runs.py --runs "output/runs/thinking_sweep_*_20260919_*/*" \\
        --select judge_only --metrics faithfulness,task_completion

    python scripts/judge_runs.py --runs ... --select all --dry-run   # what would run

Metrics:
  faithfulness          is the answer supported by the tool outputs?  (skipped: no tool output)
  task_completion       did the agent accomplish the task?
  argument_correctness  were the tool arguments sensible?             (skipped: no tool calls)
  tool_correctness      were the right tools called?                  (compared to acceptable sets)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.eval.cases import build_case  # noqa: E402
from agent.eval.claude_code_judge import ClaudeCodeJudge, JudgeError  # noqa: E402
from agent.scoring import score as deterministic_score  # noqa: E402
from agent.telemetry import EvalRecord, TelemetryWriter  # noqa: E402

METRICS = ("faithfulness", "task_completion", "argument_correctness", "tool_correctness")


def make_metric(name: str, judge: ClaudeCodeJudge):
    from deepeval.metrics import (ArgumentCorrectnessMetric, FaithfulnessMetric,
                                  TaskCompletionMetric, ToolCorrectnessMetric)
    common = {"model": judge, "async_mode": False, "include_reason": True}
    return {
        "faithfulness": lambda: FaithfulnessMetric(**common),
        "task_completion": lambda: TaskCompletionMetric(**common),
        "argument_correctness": lambda: ArgumentCorrectnessMetric(**common),
        "tool_correctness": lambda: ToolCorrectnessMetric(**common),
    }[name]()


def applicable(metric: str, case) -> bool:
    if metric == "faithfulness":
        return bool(case.retrieval_context)
    if metric == "argument_correctness":
        return bool(case.tools_called)
    return True


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (globs allowed)")
    ap.add_argument("--tasks", default="dataset/tasks/phase1.json")
    ap.add_argument("--metrics", default="faithfulness,task_completion")
    ap.add_argument("--select", default="judge_only",
                    help="judge_only | all | comma-separated task types")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel judge calls; keep modest under subscription usage limits")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N items")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = set(metrics) - set(METRICS)
    if bad:
        raise SystemExit(f"unknown metrics {sorted(bad)}; choose from {METRICS}")
    tasks = {t["task_id"]: t for t in json.loads(Path(args.tasks).read_text())}

    def selected(task: dict) -> bool:
        if args.select == "all":
            return True
        if args.select == "judge_only":
            return task["answer_check"]["mode"] == "judge_only"
        return task.get("task_type") in args.select.split(",")

    # Work list: every (run, metric) not yet scored.
    dirs = [Path(p) for pat in args.runs for p in sorted(glob.glob(pat)) if (Path(p) / "run.jsonl").exists()]
    work, writers = [], {}
    for d in dirs:
        done = {(e["run_id"], e["metric_name"]) for e in read_jsonl(d / "eval.jsonl")}
        tools_by_run: dict[str, list] = {}
        for t in read_jsonl(d / "tool_call.jsonl"):
            tools_by_run.setdefault(t["run_id"], []).append(t)
        for run in read_jsonl(d / "run.jsonl"):
            task = tasks.get(run["task_id"])
            if task is None or not selected(task):
                continue
            case = build_case(d, run, tools_by_run.get(run["run_id"], []), task)
            for m in metrics:
                if (run["run_id"], m) not in done and applicable(m, case):
                    work.append((d, run, task, case, m))
        writers[d] = TelemetryWriter(d)
    if args.limit:
        work = work[: args.limit]

    print(f"{len(dirs)} run dir(s); {len(work)} (run, metric) item(s) to score "
          f"with {args.model} via Claude Code, {args.workers} worker(s)")
    if args.dry_run or not work:
        by_metric: dict[str, int] = {}
        for *_, m in work:
            by_metric[m] = by_metric.get(m, 0) + 1
        print("  by metric:", by_metric)
        return 0

    call_log = Path("output") / f"judge_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    def score_one(item):
        d, run, task, case, metric_name = item
        judge = ClaudeCodeJudge(model=args.model, effort=args.effort, call_log=call_log)
        metric = make_metric(metric_name, judge)
        metric.measure(case)
        det = deterministic_score(run.get("final_answer", ""), task["answer_check"], run["status"])
        rec = EvalRecord(
            run_id=run["run_id"], metric_name=metric_name,
            score=metric.score, threshold=metric.threshold, passed=metric.is_successful(),
            judge_model=judge.get_model_name(), judge_version=judge.version,
            rationale=(metric.reason or "")[:1500],
            deterministic_check_passed=det.success,
            judge_calls=judge.stats["calls"],
            judge_input_tokens=judge.stats["input_tokens"],
            judge_output_tokens=judge.stats["output_tokens"],
            judge_seconds=round(judge.stats["seconds"], 2),
        )
        writers[d].write(rec)
        return item, rec

    started, ok, failed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(score_one, w) for w in work]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                (d, run, task, _, m), rec = fut.result()
                ok += 1
                cell = d.parent.name.replace("thinking_sweep_", "")[:26] + "/" + d.name
                print(f"[{i}/{len(work)}] {cell:34s} {task['task_id'][-18:]:18s} {m:21s} "
                      f"score={rec.score:.2f} {'pass' if rec.passed else 'FAIL'}  "
                      f"({rec.judge_calls} calls, {rec.judge_seconds:.0f}s)", flush=True)
            except (JudgeError, Exception) as exc:  # left unscored; a rerun retries it
                failed += 1
                print(f"[{i}/{len(work)}] error: {type(exc).__name__}: {str(exc)[:150]}", flush=True)

    print(f"\n{ok} scored, {failed} failed (rerun to retry) in {(time.time() - started) / 60:.1f} min")
    print(f"judge call log: {call_log}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
