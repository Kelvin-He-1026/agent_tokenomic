"""Score stored runs with DeepEval, judged by Claude through Claude Code.

Offline by design: runs are never re-executed, the judge sees only what the
proxy and tool layer recorded, and results go to each run directory's
eval.jsonl, the telemetry table reserved for them. It is resumable: a (run,
metric, judge, answer fingerprint) already scored is skipped, so an interrupted pass, or one halted
by subscription usage limits, continues where it stopped.

    # pilot: the time-sensitive web tasks the deterministic scorer cannot judge
    python scripts/judge_runs.py --runs "output/runs/thinking_sweep_*_20260919_*/*" \\
        --select judge_only --metrics semantic_correctness,faithfulness

    python scripts/judge_runs.py --runs ... --select all --dry-run   # what would run

Metrics:
  semantic_correctness  content correctness, independent of formatting and termination
  faithfulness          is the answer supported by the tool outputs?  (skipped: no tool output)
  task_completion       did the agent accomplish the task?
  argument_correctness  were the tool arguments sensible?             (skipped: no tool calls)
  tool_correctness      were the right tools called?                  (compared to acceptable sets)
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.eval.cases import build_case  # noqa: E402
from agent.eval.claude_code_judge import ClaudeCodeJudge, JudgeError, JUDGE_SYSTEM_HASH  # noqa: E402
from agent.scoring import score as deterministic_score  # noqa: E402
from agent.telemetry import EvalRecord, TelemetryWriter, file_stamp  # noqa: E402
from agent.evaluation import (EVALUATION_VERSION, SEMANTIC_METRIC, SEMANTIC_METRIC_VERSION,
                              answer_fingerprint, semantic_threshold, valid_semantic_judgment)  # noqa: E402

METRICS = (SEMANTIC_METRIC, "faithfulness", "task_completion", "argument_correctness", "tool_correctness")


def make_metric(name: str, judge: ClaudeCodeJudge, task: dict | None = None):
    if name == SEMANTIC_METRIC:
        from deepeval.metrics import GEval
        from deepeval.test_case import SingleTurnParams
        task = task or {}
        return GEval(
            name="Semantic correctness",
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                               SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.CONTEXT],
            criteria="Assess all requested conclusions against expected_output and the supplied reference context. "
                     "Ignore JSON syntax, numeric serialization, markdown, brevity and output formatting. "
                     "Do not infer correctness from tool use or execution status. "
                     "Reject contradicted conclusions, invented missing evidence, or omitted required subanswers. "
                     "Task rubric: " + json.dumps(task.get("grading_rubric", [])),
            evaluation_steps=[
                "Identify required factual conclusions, computations and evidence limitations in the expected output.",
                "Compare each required conclusion with the actual output; allow equivalent meanings and notation.",
                "Ignore presentation and formatting errors. Penalize factual errors, unsupported claims and missing conclusions.",
                "Give full credit only when all substantive requirements are satisfied. "
                "Apply these task-specific requirements: " + json.dumps(task.get("grading_rubric", []))
            ],
            model=judge, async_mode=False,
            threshold=semantic_threshold(task),
        )
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
        # An empty answer makes no claims, so there is nothing to be faithful to.
        return bool(case.retrieval_context) and bool(case.actual_output.strip())
    if metric == "argument_correctness":
        return bool(case.tools_called)
    return True


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def metrics_for_run(metrics: list[str], task: dict, run: dict, used: list[str], diagnostics_only: bool = False) -> list[str]:
    chosen = [m for m in metrics if not (diagnostics_only and m == SEMANTIC_METRIC)]
    if (not diagnostics_only and SEMANTIC_METRIC not in chosen
            and deterministic_score(run.get("final_answer", ""), task["answer_check"], run["status"], used).mode == "judge_only"):
        chosen.insert(0, SEMANTIC_METRIC)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (globs allowed)")
    ap.add_argument("--tasks", default="dataset/tasks/phase1.json")
    ap.add_argument("--metrics", default="semantic_correctness,faithfulness")
    ap.add_argument("--select", default="judge_only",
                    help="judge_only | all | comma-separated task types")
    ap.add_argument("--model", default="claude-opus-5-5",
                    help="pinned in dataset/policies/phase1_v2.json (judge.model); change both together")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallel judge calls; keep modest under subscription usage limits")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N items")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--diagnostics-only", action="store_true",
                    help="explicitly omit semantic correctness; otherwise judge-only runs always include it")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = set(metrics) - set(METRICS)
    if bad:
        raise SystemExit(f"unknown metrics {sorted(bad)}; choose from {METRICS}")
    tasks = {t["task_id"]: t for t in json.loads(Path(args.tasks).read_text())}
    judge_id = hashlib.sha256(json.dumps({"model": args.model, "effort": args.effort,
                                        "version": EVALUATION_VERSION, "provider": "claude_code",
                                        "system_prompt_hash": JUDGE_SYSTEM_HASH}, sort_keys=True).encode()).hexdigest()[:16]

    def selected(task: dict, run: dict, tools_used: list[str]) -> bool:
        if args.select == "all":
            return True
        if args.select == "judge_only":
            # Decided per run, not per task: a task with judge_if_tools is
            # judged only for the runs that called an online tool.
            verdict = deterministic_score(
                run.get("final_answer", ""), task["answer_check"], run["status"], tools_used
            )
            return verdict.mode == "judge_only"
        return task.get("task_type") in args.select.split(",")

    # Work list: every (run, metric) not yet scored.
    dirs = [Path(p) for pat in args.runs for p in sorted(glob.glob(pat)) if (Path(p) / "run.jsonl").exists()]
    work, writers = [], {}
    for d in dirs:
        previous = read_jsonl(d / "eval.jsonl")
        done = {(e["run_id"], e["metric_name"], e.get("judge_id"), e.get("answer_fingerprint"))
                for e in previous if e.get("passed") is not None}
        tools_by_run: dict[str, list] = {}
        for t in read_jsonl(d / "tool_call.jsonl"):
            tools_by_run.setdefault(t["run_id"], []).append(t)
        for run in read_jsonl(d / "run.jsonl"):
            task = tasks.get(run["task_id"])
            used = [t["tool_name"] for t in tools_by_run.get(run["run_id"], [])]
            if task is None or not selected(task, run, used):
                continue
            if run.get("task_fingerprint") and run["task_fingerprint"] != answer_fingerprint(task, ""):
                print(f"skip {run['run_id']}: task definition changed; rerun the agent with this dataset version")
                continue
            case = build_case(d, run, tools_by_run.get(run["run_id"], []), task)
            for m in metrics_for_run(metrics, task, run, used, args.diagnostics_only):
                key = (run["run_id"], m, judge_id, answer_fingerprint(task, case.actual_output))
                already_done = key in done
                if m == SEMANTIC_METRIC:
                    already_done = any(e.get("run_id") == run["run_id"] and e.get("judge_id") == judge_id
                                       and valid_semantic_judgment(e, task, case.actual_output) for e in previous)
                if not already_done and applicable(m, case):
                    work.append((d, run, task, case, m))
        writers[d] = TelemetryWriter(d)
    if args.limit:
        work = work[: args.limit]

    print(f"{len(dirs)} run dir(s); {len(work)} (run, metric) item(s) to score "
          f"with {args.model} via Claude Code, {args.workers} worker(s); judge_id={judge_id}")
    if args.dry_run or not work:
        by_metric: dict[str, int] = {}
        for *_, m in work:
            by_metric[m] = by_metric.get(m, 0) + 1
        print("  by metric:", by_metric)
        return 0

    call_log = Path("output") / f"judge_calls_{file_stamp()}.jsonl"

    def score_one(item):
        d, run, task, case, metric_name = item
        if not case.actual_output.strip():
            # DeepEval refuses an empty answer. A run that produced no answer
            # did not complete its task, so record that without a judge call.
            rec = EvalRecord(
                run_id=run["run_id"], metric_name=metric_name,
                score=0.0, threshold=semantic_threshold(task) if metric_name == SEMANTIC_METRIC else 0.5, passed=False,
                judge_model="none: empty final answer", judge_version="",
                rationale=f"No final answer (run status: {run['status']}); scored 0 without a judge call.",
                deterministic_check_passed=None,
                judge_id=judge_id, evaluation_version=EVALUATION_VERSION,
                answer_fingerprint=answer_fingerprint(task, case.actual_output),
                metric_version=SEMANTIC_METRIC_VERSION if metric_name == SEMANTIC_METRIC else "",
                judge_calls=0, judge_input_tokens=0, judge_output_tokens=0, judge_seconds=0.0,
            )
            writers[d].write(rec)
            return item, rec
        judge = ClaudeCodeJudge(model=args.model, effort=args.effort, call_log=call_log)
        metric = make_metric(metric_name, judge, task)
        metric.measure(case)
        if metric_name == SEMANTIC_METRIC:
            import math
            if not isinstance(metric.score, (int, float)) or not math.isfinite(metric.score) or not 0 <= metric.score <= 1:
                raise JudgeError(f"invalid normalized semantic score {metric.score!r}; expected [0, 1]")
        det = deterministic_score(run.get("final_answer", ""), task["answer_check"], run["status"],
                                  [t.name for t in (case.tools_called or [])])
        rec = EvalRecord(
            run_id=run["run_id"], metric_name=metric_name,
            score=metric.score, threshold=metric.threshold, passed=metric.is_successful(),
            judge_model=judge.get_model_name(), judge_version=judge.version,
            rationale=(metric.reason or "")[:1500],
            deterministic_check_passed=det.success,
            judge_id=judge_id, evaluation_version=EVALUATION_VERSION,
            answer_fingerprint=answer_fingerprint(task, case.actual_output),
            metric_version=SEMANTIC_METRIC_VERSION if metric_name == SEMANTIC_METRIC else "",
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
