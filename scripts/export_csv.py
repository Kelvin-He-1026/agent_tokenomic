"""Flatten run telemetry into one analysis-ready CSV.

One row per agent run: the analysis unit for every Phase 1 comparison. The four
JSONL tables are joined on run_id, config fields are denormalized onto each row,
and per-call tables are aggregated.

Nothing is filtered or corrected here. Warm-up runs, framework errors and
approximate fixture matches all appear, flagged by their own columns, so the
analysis decides what to exclude rather than the exporter deciding silently.

    python scripts/export_csv.py --runs output/runs/first_react --out output/results.csv
    python scripts/export_csv.py --runs "output/runs/*" --out output/all.csv --calls
"""

from __future__ import annotations

import argparse
import csv
import sys
import glob
import json
import statistics
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT_DIR = Path(__file__).resolve().parent.parent

from agent import prompt_breakdown, transcripts
from agent.accounting import cap_hits
from agent.evaluation import OUTCOME_POLICY_VERSION

COLUMNS = [
    # identity
    "run_id", "config_id", "run_dir", "run_dir_path", "run_code", "framework", "framework_version",
    "session", "task_fingerprint", "task_definition_matches",
    "architecture", "model_repo", "precision", "tp_size",
    "task_id", "repeat_idx", "run_order_idx", "is_first_in_dir",
    # outcome
    "status", "error", "final_answer",
    # Unified correctness and independent presentation/execution dimensions.
    "task_type", "check_mode", "success", "success_detail", "format_ok",
    "difficulty", "semantic_correct", "execution_completed", "strict_success", "format_success",
    "completion_type", "natural_completion", "outcome_policy_version",
    # one label per run: success | wrong_answer | format_failure | tool_failure
    # | step_limit | output_truncation | timeout | framework_error | pending_judge
    "failure_category",
    "score_source", "score_state", "evaluation_version",
    # LLM judge (DeepEval, scripts/judge_runs.py); empty where not yet scored
    "judge_faithfulness", "judge_faithfulness_pass",
    "judge_task_completion", "judge_task_completion_pass",
    "judge_argument_correctness", "judge_argument_correctness_pass",
    "judge_tool_correctness", "judge_tool_correctness_pass",
    "judge_model", "judge_calls", "judge_input_tokens", "judge_output_tokens",
    "judge_id", "judge_semantic_correctness", "judge_semantic_correctness_pass",
    "judge_semantic_correctness_threshold", "judge_semantic_correctness_version",
    # trajectory
    "n_llm_calls", "n_tool_calls", "n_failed_tool_calls", "n_malformed_tool_calls",
    "n_tool_switches", "tools_used", "hit_step_limit",
    "framework_status", "termination_reason", "forced_final_answer", "call_cap_reached",
    "output_cap_reached", "blocked_llm_requests", "budget_enforced", "accounting_policy",
    "per_call_output_cap_reached", "total_output_cap_reached", "output_cap_type", "last_finish_reason",
    "usage_complete",
    # tokens
    # output_tokens = reasoning_tokens + answer_tokens. Thinking tokens are
    # generated and paid for like any other output, so they stay in the total.
    "input_tokens", "output_tokens", "reasoning_tokens", "answer_tokens",
    "reasoning_share_pct", "reasoning_chars", "reasoning_tokens_missing",
    "cached_input_tokens", "total_tokens", "prefix_cache_hit_rate", "tokens_per_llm_call",
    # input_tokens by source, summed over the run's calls (agent/prompt_breakdown.py)
    "system_prompt_tokens", "conversation_history_tokens", "tool_response_tokens",
    "current_user_prompt_tokens", "other_prompt_tokens",
    # latency
    "e2e_ms", "t_llm_ms", "t_tool_ms", "t_orch_ms", "t_queue_ms",
    "llm_share_pct", "orch_share_pct",
    "ttft_first_ms", "ttft_mean_ms", "tpot_mean_ms", "llm_latency_max_ms",
    # resources
    "proc_cpu_ms", "peak_rss_mb", "energy_gpu_j", "energy_cpu_j",
    # energy_gpu_j = sum of the per-GPU values, each the driver's cumulative
    # energy counter (NVML total energy) at run end minus at run start
    "energy_gpu0_j", "energy_gpu1_j", "energy_total_j", "energy_method",
    # fixtures
    "fixture_exact", "fixture_normalized", "fixture_nearest", "fixture_live",
    "has_approx_fixture",
    # provenance
    "system_prompt_hash", "tool_schema_hash", "tool_parser", "reasoning_parser",
    "thinking_setting", "reasoning_effort", "seed", "temperature", "max_llm_calls",
    "max_tokens", "max_total_output_tokens", "task_timeout_s",
    "start_ts", "end_ts", "harness_version", "schema_version",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def mean(values: list[float]) -> float | str:
    return round(statistics.mean(values), 2) if values else ""


def pct(part: float, whole: float) -> float | str:
    return round(part / whole * 100, 2) if whole else ""


TASKS: dict[str, dict[str, Any]] = {}
PRIMARY_JUDGE_ID = ""


def load_tasks(path: str) -> None:
    """Task annotations, so each run can be scored against its answer check."""
    p = Path(path)
    if p.exists():
        TASKS.clear()
        TASKS.update({t["task_id"]: t for t in json.loads(p.read_text())})


def flatten_dir(run_dir: Path) -> list[dict[str, Any]]:
    runs = read_jsonl(run_dir / "run.jsonl")
    if not runs:
        return []

    configs = {c["config_id"]: c for c in read_jsonl(run_dir / "config.jsonl")}
    calls_by_run: dict[str, list[dict]] = {}
    for c in read_jsonl(run_dir / "llm_call.jsonl"):
        calls_by_run.setdefault(c["run_id"], []).append(c)
    tools_by_run: dict[str, list[dict]] = {}
    for t in read_jsonl(run_dir / "tool_call.jsonl"):
        tools_by_run.setdefault(t["run_id"], []).append(t)
    evals_by_run: dict[str, list[dict]] = {}
    for e in read_jsonl(run_dir / "eval.jsonl"):
        evals_by_run.setdefault(e["run_id"], []).append(e)

    # Warm-up: the first run executed in a directory pays import and JIT costs
    # that belong to nothing being measured. Flagged, not dropped.
    first_run_id = min(runs, key=lambda r: r["run_order_idx"])["run_id"]

    rows = []
    for r in runs:
        cfg = configs.get(r["config_id"], {})
        calls = sorted(calls_by_run.get(r["run_id"], []), key=lambda c: c["call_idx"])
        tools = sorted(tools_by_run.get(r["run_id"], []), key=lambda t: t["tool_idx"])
        r = {**r, **cap_hits(r, calls, cfg)}

        reasoning_chars = sum(c.get("reasoning_chars", 0) or 0 for c in calls)
        reasoning_tokens = r["reasoning_tokens"]
        ttfts = [c["ttft_ms"] for c in calls if c.get("ttft_ms") is not None]
        tpots = [c["tpot_ms"] for c in calls if c.get("tpot_ms") is not None]
        latencies = [c.get("latency_ms", 0.0) for c in calls]
        matches = [t.get("fixture_match", "") for t in tools]
        total_tokens = r["input_tokens"] + r["output_tokens"]

        rows.append({
            "run_id": r["run_id"],
            "config_id": r["config_id"],
            "run_dir": run_dir.name,
            "run_dir_path": str(run_dir.resolve()),
            "run_code": r.get("run_code", ""),
            "session": r.get("session", ""),
            "task_fingerprint": r.get("task_fingerprint", ""),
            "framework": cfg.get("framework", ""),
            "framework_version": cfg.get("framework_version", ""),
            "architecture": cfg.get("architecture", ""),
            "model_repo": cfg.get("model_repo", ""),
            "precision": cfg.get("precision", ""),
            "tp_size": cfg.get("tp_size", ""),
            "task_id": r["task_id"],
            "repeat_idx": r["repeat_idx"],
            "run_order_idx": r["run_order_idx"],
            "is_first_in_dir": r["run_id"] == first_run_id,

            "status": r["status"],
            "error": r.get("error", "")[:200],
            "final_answer": " ".join(r.get("final_answer", "").split())[:500],
            **_success(r, [t["tool_name"] for t in tools], evals_by_run.get(r["run_id"], [])),
            **_judge(evals_by_run.get(r["run_id"], []), TASKS.get(r["task_id"]), r.get("final_answer", "")),
            "difficulty": TASKS.get(r["task_id"], {}).get("difficulty", ""),
            "completion_type": r["completion_type"],
            "outcome_policy_version": OUTCOME_POLICY_VERSION,

            "n_llm_calls": r["n_llm_calls"],
            "n_tool_calls": r["n_tool_calls"],
            "n_failed_tool_calls": r["n_failed_tool_calls"],
            "n_malformed_tool_calls": r["n_malformed_tool_calls"],
            "n_tool_switches": r["n_tool_switches"],
            "tools_used": "|".join(t["tool_name"] for t in tools),
            "hit_step_limit": r["status"] == "step_limit",
            "framework_status": r.get("framework_status", r["status"]),
            "termination_reason": r.get("termination_reason", "legacy_unknown"),
            "forced_final_answer": r.get("forced_final_answer", False),
            "call_cap_reached": r.get("call_cap_reached", False),
            "output_cap_reached": r.get("output_cap_reached", False),
            "per_call_output_cap_reached": r["per_call_output_cap_reached"],
            "total_output_cap_reached": r["total_output_cap_reached"],
            "output_cap_type": r["output_cap_type"],
            "last_finish_reason": r["last_finish_reason"],
            "blocked_llm_requests": r.get("blocked_llm_requests", 0),
            "budget_enforced": r.get("budget_enforced", False),
            "usage_complete": r.get("usage_complete", ""),
            "accounting_policy": r.get("accounting_policy", "legacy_unknown"),

            "input_tokens": r["input_tokens"],
            **run_breakdown(run_dir, calls),
            "output_tokens": r["output_tokens"],
            "reasoning_tokens": reasoning_tokens,
            "answer_tokens": r["output_tokens"] - reasoning_tokens,
            "reasoning_share_pct": pct(reasoning_tokens, r["output_tokens"]),
            "reasoning_chars": reasoning_chars,
            # Reasoning text present but no token count: the server or parser did
            # not report it, so thinking tokens are unknown rather than zero.
            "reasoning_tokens_missing": reasoning_chars > 0 and reasoning_tokens == 0,
            "cached_input_tokens": r["cached_input_tokens"],
            "total_tokens": total_tokens,
            "prefix_cache_hit_rate": r.get("prefix_cache_hit_rate", ""),
            "tokens_per_llm_call": round(total_tokens / r["n_llm_calls"], 1) if r["n_llm_calls"] else "",

            "e2e_ms": round(r["e2e_ms"], 1),
            "t_llm_ms": round(r["t_llm_ms"], 1),
            "t_tool_ms": round(r["t_tool_ms"], 1),
            "t_orch_ms": round(r["t_orch_ms"], 1),
            "t_queue_ms": round(r["t_queue_ms"], 1),
            "llm_share_pct": pct(r["t_llm_ms"], r["e2e_ms"]),
            "orch_share_pct": pct(r["t_orch_ms"], r["e2e_ms"]),
            "ttft_first_ms": round(ttfts[0], 1) if ttfts else "",
            "ttft_mean_ms": mean(ttfts),
            "tpot_mean_ms": mean(tpots),
            "llm_latency_max_ms": round(max(latencies), 1) if latencies else "",

            "proc_cpu_ms": round(r["proc_cpu_ms"], 1),
            "peak_rss_mb": round(r["peak_rss_mb"], 1),
            **run_energy(r),

            "fixture_exact": matches.count("exact"),
            "fixture_normalized": matches.count("normalized"),
            "fixture_nearest": matches.count("nearest"),
            "fixture_live": matches.count("live"),
            "has_approx_fixture": "nearest" in matches,

            "system_prompt_hash": cfg.get("system_prompt_hash", ""),
            "tool_schema_hash": cfg.get("tool_schema_hash", ""),
            "tool_parser": cfg.get("tool_parser", ""),
            "reasoning_parser": cfg.get("reasoning_parser", ""),
            "thinking_setting": cfg.get("thinking_setting", ""),
            "reasoning_effort": cfg.get("reasoning_effort", ""),
            "seed": cfg.get("seed", ""),
            "temperature": cfg.get("temperature", ""),
            "max_llm_calls": cfg.get("max_llm_calls", ""),
            "max_tokens": cfg.get("max_tokens", ""),
            "max_total_output_tokens": cfg.get("max_total_output_tokens", ""),
            "task_timeout_s": cfg.get("timeout_s", ""),
            "start_ts": r["start_ts"],
            "end_ts": r["end_ts"],
            "harness_version": r.get("harness_version", ""),
            "schema_version": r.get("schema_version", ""),
        })
        rows[-1]["failure_category"] = failure_category(rows[-1], r["n_failed_tool_calls"])
    return rows


def _judge(evals: list[dict[str, Any]], task: dict | None = None, answer: str = "") -> dict[str, Any]:
    """Latest judge verdict per metric, and the total cost of judging this run."""
    out: dict[str, Any] = {}
    if task is not None:
        from agent.evaluation import SEMANTIC_METRIC, valid_semantic_judgment
        evals = [e for e in evals if e.get("metric_name") != SEMANTIC_METRIC or valid_semantic_judgment(e, task, answer)]
    identities = {e.get("judge_id", e.get("judge_model", "")) for e in evals}
    if PRIMARY_JUDGE_ID:
        evals = [e for e in evals if e.get("judge_id") == PRIMARY_JUDGE_ID]
    elif len(identities) > 1:
        return {"judge_model": "multiple: select --judge-id"}
    latest: dict[str, dict] = {}
    for e in evals:
        latest[e["metric_name"]] = e  # rows are appended, so the last one wins
    for name, e in latest.items():
        out[f"judge_{name}"] = e["score"]
        out[f"judge_{name}_pass"] = e["passed"]
        if name == "semantic_correctness":
            out["judge_semantic_correctness_threshold"] = e.get("threshold", "")
            out["judge_semantic_correctness_version"] = e.get("metric_version", "")
    if latest:
        out["judge_model"] = next(iter(latest.values())).get("judge_model", "")
        out["judge_id"] = next(iter(latest.values())).get("judge_id", "")
        out["judge_calls"] = sum(e.get("judge_calls", 0) for e in latest.values())
        out["judge_input_tokens"] = sum(e.get("judge_input_tokens", 0) for e in latest.values())
        out["judge_output_tokens"] = sum(e.get("judge_output_tokens", 0) for e in latest.values())
    return out


def _success(r: dict[str, Any], tools_used: list[str], evals: list[dict] = ()) -> dict[str, Any]:
    from agent.evaluation import evaluate
    from agent.evaluation import answer_fingerprint

    task = TASKS.get(r["task_id"])
    if task is None:
        return {"task_type": "", "check_mode": "", "success": "", "success_detail": "task not found"}
    fingerprint = r.get("task_fingerprint")
    matches = fingerprint == answer_fingerprint(task, "") if fingerprint else ""
    if matches is False:
        return {"task_type": task.get("task_type", ""), "check_mode": task["answer_check"]["mode"],
                "success": "", "success_detail": "stored task definition differs; rerun with this task version",
                "score_state": "task_mismatch", "task_definition_matches": False}
    return {**evaluate(task, r.get("final_answer", ""), r["status"], tools_used, evals, PRIMARY_JUDGE_ID,
                       completion_type=r.get("completion_type")),
            "task_definition_matches": matches}


def quality_report(dirs: list[Path], rows: list[dict]) -> dict:
    """Account for planned runs even when an interrupted harness wrote no result."""
    cells = []
    for directory in dirs:
        plans = read_jsonl(directory / "plan.jsonl")
        local = [r for r in rows if r.get("run_dir_path", r["run_dir"]) == str(directory.resolve())
                 or (not r.get("run_dir_path") and r["run_dir"] == directory.name)]
        groups = {(r["config_id"], r.get("session", "")) for r in local + plans}
        for config_id, session in sorted(groups):
            actual = [r for r in local if (r["config_id"], r.get("session", "")) == (config_id, session)]
            planned = [r for r in plans if (r["config_id"], r.get("session", "")) == (config_id, session)]
            def key(r):
                return r["task_id"], r["repeat_idx"], r["run_order_idx"]
            expected, observed = {key(r) for r in planned}, {key(r) for r in actual}
            pending = sum(r.get("score_state") in ("pending_judge", "ambiguous_judge", "task_mismatch") for r in actual)
            wins = sum(r.get("success") is True for r in actual)
            duplicate_count = len(actual) - len({(r["task_id"], r["repeat_idx"]) for r in actual})
            dataset_complete = {r["task_id"] for r in actual} == set(TASKS)
            plan_complete = (bool(planned) and expected == observed and not duplicate_count
                             and len(planned) == len({(r["task_id"], r["repeat_idx"]) for r in planned}))
            confirmed_rate = wins / len(planned) if plan_complete else None
            cells.append({"run_dir": str(directory.resolve()), "config_id": config_id, "session": session,
                          "expected_dataset_tasks": len(TASKS), "observed_tasks": len({r['task_id'] for r in actual}),
                          "planned_runs": len(planned) if planned else None, "observed_runs": len(actual),
                          "missing_runs": [list(k) for k in sorted(expected - observed)],
                          "unexpected_runs": [list(k) for k in sorted(observed - expected)] if planned else [],
                          "duplicate_runs": duplicate_count, "pending_judgments_or_task_mismatch": pending,
                          "semantic_scored": sum(r.get("semantic_correct", "") in (True, False) for r in actual),
                          "unscorable_format": sum(r.get("score_state") == "unscorable_format" for r in actual),
                          "format_failures": sum(r.get("format_ok") is False for r in actual),
                          "confirmed_successes": wins, "confirmed_success_rate": confirmed_rate,
                          "plan_complete": plan_complete,
                          "selected_quality_complete": plan_complete and pending == 0,
                          "provisional": not plan_complete or pending > 0,
                          "strict_successes": sum(r.get("strict_success") is True for r in actual),
                          "strict_success_rate": sum(r.get("strict_success") is True for r in actual) / len(planned) if plan_complete else None,
                          "completion_types": {name: sum(r.get("completion_type") == name for r in actual)
                                               for name in ("natural_completion", "forced_final", "step_limit", "truncated", "timeout", "error", "unknown_completion")},
                          "per_call_output_cap_hits": sum(r.get("per_call_output_cap_reached") is True for r in actual),
                          "total_output_cap_hits": sum(r.get("total_output_cap_reached") is True for r in actual),
                          "quality_complete": dataset_complete and plan_complete and pending == 0,
                          "resource_accounting_complete": bool(actual) and all(r.get("budget_enforced") is True
                              and r.get("usage_complete") is True for r in actual),
                          "total_agent_tokens": sum(r["total_tokens"] for r in actual),
                          "total_agent_seconds": sum(r["e2e_ms"] for r in actual) / 1000})
    return {"dataset_tasks": len(TASKS), "primary_judge_id": PRIMARY_JUDGE_ID,
            "outcome_policy_version": OUTCOME_POLICY_VERSION, "cells": cells}


def select_primary_judge(evals: list[dict], requested: str = "") -> str:
    from agent.evaluation import EVALUATION_VERSION, SEMANTIC_METRIC, SEMANTIC_METRIC_VERSION
    if requested:
        return requested
    identities = {e["judge_id"] for e in evals if e.get("metric_name") == SEMANTIC_METRIC
                  and e.get("evaluation_version") == EVALUATION_VERSION
                  and e.get("metric_version") == SEMANTIC_METRIC_VERSION
                  and e.get("judge_id") and e.get("passed") is not None}
    if len(identities) > 1:
        raise ValueError("multiple semantic judges in this export; select --judge-id from " + ", ".join(sorted(identities)))
    return next(iter(identities), "")


def cell_key(r: dict) -> str:
    """Never average observations from different resource policies together."""
    return (f"{(r['model_repo'] or '?').split('/')[-1]} {r['framework']} "
            f"{r['thinking_setting']}/{r.get('reasoning_effort') or '-'} "
            f"calls={r.get('max_llm_calls', '?')} output={r.get('max_tokens', '?')} "
            f"total={r.get('max_total_output_tokens', '?')} timeout={r.get('task_timeout_s', '?')} "
            f"{r.get('accounting_policy', '?')}/{r.get('budget_enforced', False)} cfg={r.get('config_id', '?')}")


# Power traces from scripts/power_sampler.py, and every exported run's window.
# Set in main(); used by run_energy().
ENERGY_TRACE = None
RUN_WINDOWS: list[tuple[float, float]] = []


def run_energy(r: dict[str, Any]) -> dict[str, Any]:
    """GPU and CPU energy (J) for one run, from a power trace covering it.

    Blank when no trace covers the run, or when another exported run overlapped
    it (parallel recording): energy over a shared window cannot be split per
    run. CPU energy needs readable RAPL counters during the run.
    """
    out = {"energy_gpu_j": r.get("energy_gpu_j") or "", "energy_cpu_j": r.get("energy_cpu_j") or "",
           "energy_gpu0_j": "", "energy_gpu1_j": "", "energy_total_j": "", "energy_method": ""}
    a, b = r.get("start_ts"), r.get("end_ts")
    if ENERGY_TRACE is None or not a or not b or not ENERGY_TRACE.covers(a, b):
        return out
    if sum(1 for x, y in RUN_WINDOWS if x < b and y > a) > 1:
        out["energy_method"] = "overlapping runs: not attributable"
        return out
    each = ENERGY_TRACE.gpu_energy_each(a, b)
    out["energy_gpu_j"] = round(sum(each), 2)
    for i, e in enumerate(each[:2]):
        out[f"energy_gpu{i}_j"] = round(e, 2)
    cpu = ENERGY_TRACE.cpu_energy(a, b)
    out["energy_cpu_j"] = "" if cpu is None else round(cpu, 2)
    out["energy_total_j"] = "" if cpu is None else round(sum(each) + cpu, 2)
    out["energy_method"] = ("gpu: NVML energy counter, all GPUs; "
                            + ("cpu: RAPL package counters, both sockets" if cpu is not None
                               else "cpu: not measured (RAPL unreadable)"))
    return out


def failure_category(row: dict[str, Any], tools_failed: int) -> str:
    """One label per run, in a fixed order of precedence: how the run ended
    first, then whether the answer was right, then why not."""
    status, ctype = row.get("status"), row.get("completion_type")
    if status == "step_limit" or ctype == "step_limit":
        return "step_limit"
    if status == "truncated" or ctype == "truncated":
        return "output_truncation"
    if status == "timeout" or ctype == "timeout":
        return "timeout"
    if status not in ("completed", None, ""):
        return "framework_error"
    if row.get("score_state") in ("pending_judge", "ambiguous_judge"):
        return "pending_judge"
    if row.get("success") is True:
        return "success"
    if row.get("score_state") == "unscorable_format" or (
            row.get("semantic_correct") in ("", None) and row.get("format_ok") is False):
        return "format_failure"
    if tools_failed:
        return "tool_failure"
    return "wrong_answer"


def load_energy(dirs: list[Path]) -> None:
    """Find power traces for these runs: each run folder's power.jsonl (written
    by the harness), plus any in output/power/ (a sampler started by hand)."""
    global ENERGY_TRACE, RUN_WINDOWS
    RUN_WINDOWS = [(r["start_ts"], r["end_ts"]) for d in dirs for r in read_jsonl(d / "run.jsonl")
                   if r.get("start_ts") and r.get("end_ts")]
    files = [d / "power.jsonl" for d in dirs if (d / "power.jsonl").exists()]
    files += [Path(p) for p in sorted(glob.glob(str(ROOT_DIR / "output" / "power" / "*.jsonl")))]
    if not files or not RUN_WINDOWS:
        return
    lo, hi = min(a for a, _ in RUN_WINDOWS), max(b for _, b in RUN_WINDOWS)
    from attribute_energy import Trace  # scripts/ is on sys.path when run as a script

    def overlaps(f: Path) -> bool:
        with f.open() as fh:
            first = [json.loads(l) for _, l in zip(range(2), fh)]
        start = next((x["t"] for x in first if "t" in x), None)
        return start is not None and start <= hi and f.stat().st_mtime * 1000 >= lo

    files = [f for f in files if overlaps(f)]
    if files:
        ENERGY_TRACE = Trace(files)


# Excel stops at 32,767 characters per cell; the JSONL keeps the full text.
CSV_TEXT_LIMIT = 32000


def export_texts(dirs: list[Path], rows: list[dict[str, Any]], out_csv: Path, out_jsonl: Path) -> int:
    """Per run: the prompt sent, the reasoning, the answer, and the four token
    sources, as a CSV for reading and a JSONL with the full call-by-call trace."""
    by_id = {r["run_id"]: r for r in rows}
    keep = ["run_id", "run_dir", "task_id", "framework", "model_repo", "thinking_setting",
            "reasoning_effort", "status", "success", "check_mode", *prompt_breakdown.COLUMNS]
    text_cols = ["question", "system_prompt", "user_prompt", "reasoning", "final_answer"]
    n = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fh, open(out_jsonl, "w", encoding="utf-8") as jh:
        w = csv.DictWriter(fh, fieldnames=keep + ["reasoning_chars", "truncated_in_csv"] + text_cols)
        w.writeheader()
        for d in dirs:
            calls_by_run: dict[str, list[dict]] = {}
            for c in read_jsonl(d / "llm_call.jsonl"):
                calls_by_run.setdefault(c["run_id"], []).append(c)
            for run in read_jsonl(d / "run.jsonl"):
                base = by_id.get(run["run_id"], {})
                t = transcripts.run_transcript(d, run, calls_by_run.get(run["run_id"], []))
                question = (TASKS.get(run["task_id"]) or {}).get("question", "")
                record = {k: base.get(k, "") for k in keep}
                record.update({"question": question, **{k: t[k] for k in text_cols if k in t}})
                jh.write(json.dumps({**record, "calls": t["calls"]}, ensure_ascii=False) + "\n")
                row = dict(record)
                cut = [k for k in text_cols if len(row[k] or "") > CSV_TEXT_LIMIT]
                for k in cut:
                    row[k] = row[k][:CSV_TEXT_LIMIT] + " [...truncated; full text in the JSONL]"
                row["reasoning_chars"] = len(t["reasoning"])
                row["truncated_in_csv"] = "|".join(cut)
                w.writerow(row)
                n += 1
    return n


_BREAKDOWN: dict[tuple[str, str, int], dict[str, Any]] = {}


def call_breakdown(run_dir: Path, call: dict[str, Any]) -> dict[str, Any]:
    """Prompt tokens by source for one call, counted once per export."""
    key = (str(run_dir), call["run_id"], call["call_idx"])
    if key not in _BREAKDOWN:
        model = next((c.get("model_repo", "") for c in read_jsonl(run_dir / "config.jsonl")), "")
        _BREAKDOWN[key] = prompt_breakdown.for_call(call, run_dir, prompt_breakdown.counter_for(model))
    return _BREAKDOWN[key]


def run_breakdown(run_dir: Path, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """The per-call breakdown summed over a run; blank if any call is uncounted."""
    parts = [call_breakdown(run_dir, c) for c in calls]
    out: dict[str, Any] = {}
    for col in prompt_breakdown.COLUMNS:
        vals = [p[col] for p in parts]
        out[col] = sum(vals) if vals and all(v != "" for v in vals) else ""
    return out


def export_calls(dirs: list[Path], out: Path) -> int:
    """Long-form per-LLM-call CSV, for looking at trajectories step by step."""
    cols = ["run_dir", "run_dir_path", "run_id", "call_idx", "request_ts", "ttft_ms", "latency_ms",
            "prompt_tokens", "completion_tokens", "reasoning_tokens", "answer_tokens",
            "reasoning_chars", "answer_chars", "cached_tokens",
            "tpot_ms", "finish_reason", "n_tool_calls_requested", "tool_names",
            "n_messages", "prompt_chars", "contains_reasoning_in_history"]
    cols += ["requested_output_cap", "effective_output_cap", "http_status", "usage_missing", "forced_final_answer", "error"]
    # Where each call's prompt tokens come from (agent/prompt_breakdown.py).
    cols += list(prompt_breakdown.COLUMNS)
    rows = []
    for d in dirs:
        for c in read_jsonl(d / "llm_call.jsonl"):
            row = {k: c.get(k, "") for k in cols}
            row.update(call_breakdown(d, c))
            row["run_dir"] = d.name
            row["run_dir_path"] = str(d.resolve())
            row["tool_names"] = "|".join(c.get("tool_names") or [])
            row["answer_tokens"] = (c.get("completion_tokens") or 0) - (c.get("reasoning_tokens") or 0)
            rows.append(row)
    rows.sort(key=lambda r: (r["run_dir"], r["run_id"], r["call_idx"]))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def summarize(rows: list[dict[str, Any]]) -> None:
    warm = [r for r in rows if not r["is_first_in_dir"] and r["status"] == "completed"]
    print(f"\n{len(rows)} run(s); {len(warm)} completed and not first-in-dir")

    # Grouped by everything that defines a cell, so a sweep over thinking modes
    # (or models) is never averaged together into one row.
    by_fw: dict[str, list[dict]] = {}
    for r in warm:
        key = cell_key(r)
        by_fw.setdefault(key, []).append(r)

    print(f"\n{'cell':45s} {'n':>3s} {'e2e_s':>7s} {'llm%':>6s} {'orch%':>6s} "
          f"{'steps':>6s} {'tools':>6s} {'in_tok':>7s} {'out_tok':>7s} {'think':>6s} "
          f"{'tok_med':>8s} {'tok_max':>8s} {'tasks':>6s}")
    print("-" * 133)
    for fw, rs in sorted(by_fw.items()):
        print(f"{fw:45s} {len(rs):3d} "
              f"{statistics.mean(r['e2e_ms'] for r in rs) / 1000:7.1f} "
              f"{statistics.mean(r['llm_share_pct'] for r in rs):6.1f} "
              f"{statistics.mean(r['orch_share_pct'] for r in rs):6.2f} "
              f"{statistics.mean(r['n_llm_calls'] for r in rs):6.1f} "
              f"{statistics.mean(r['n_tool_calls'] for r in rs):6.1f} "
              f"{statistics.mean(r['input_tokens'] for r in rs):7.0f} "
              f"{statistics.mean(r['output_tokens'] for r in rs):7.0f} "
              f"{statistics.mean(r['reasoning_tokens'] for r in rs):6.0f} "
              f"{statistics.median(r['total_tokens'] for r in rs):8.0f} "
              f"{max(r['total_tokens'] for r in rs):8.0f} "
              f"{len({r['task_id'] for r in rs}):6d}")

    # Cost per successful task: EVERY run's tokens (failures cost too) divided
    # by confirmed successful runs. Pending/failed runs still incur cost.
    # Warm-up runs are included here: warm-up inflates timing (imports, JIT),
    # not correctness or token counts, and with one repeat per task, dropping
    # it would drop that task from every cell.
    print(f"\n{'cell':45s} {'all':>5s} {'pending':>7s} {'confirmed':>9s} {'tok/success':>12s} {'s/success':>10s}")
    print("-" * 86)
    cells: dict[str, list[dict]] = {}
    for r in rows:
        cells.setdefault(cell_key(r), []).append(r)
    for key, rs in sorted(cells.items()):
        scored = [r for r in rs if r["success"] != ""]
        wins = sum(r["success"] is True for r in scored)
        tok = sum(r["total_tokens"] for r in rs)
        sec = sum(r["e2e_ms"] for r in rs) / 1000
        pending = sum(r.get("score_state") in ("pending_judge", "ambiguous_judge") for r in rs)
        print(f"{key:45s} {len(rs):5d} {pending:7d} {wins / len(rs):9.0%} "
              f"{(tok / wins) if wins else float('nan'):12,.0f} {(sec / wins) if wins else float('nan'):10.1f}")
        unknown = sum(r.get("score_state") == "unscorable_format" for r in rs)
        print(f"  task coverage={len({r['task_id'] for r in rs})}/{len(TASKS)} loaded tasks")
        print(f"  semantic coverage={sum(r.get('semantic_correct') in (True, False) for r in rs)}/{len(rs)}; "
              f"unscorable_format={unknown}; format failures={sum(r.get('format_ok') is False for r in rs)}")
        print(f"  strict successes={sum(r.get('strict_success') is True for r in rs)}/{len(rs)}; "
              + ", ".join(f"{kind}={sum(r.get('completion_type') == kind for r in rs)}"
                          for kind in ("natural_completion", "forced_final", "step_limit", "truncated")))
    print("(All run costs, including failed and pending runs, are charged to confirmed successes.\n"
          " Rates and costs remain provisional while judgments are pending. Warm-up runs are included.)")

    print("\nMeans are shown for scale, but agent costs are heavy-tailed: one looping task")
    print("can dominate a mean. Compare tok_med (typical) and tok_max (worst) before")
    print("concluding. 'tasks' is the independent sample size; at temperature 0 repeats")
    print("of a task are often identical, so n overstates the evidence.")

    budget = [r for r in rows if r["max_llm_calls"] and str(r["n_llm_calls"]) == str(r["max_llm_calls"])]
    if budget:
        print(f"NOTE: {len(budget)} run(s) used their entire LLM-call budget "
              f"({', '.join(sorted({r['task_id'] for r in budget}))}); a lower budget "
              "would have ended them as step_limit")

    excluded = [r for r in rows if r["is_first_in_dir"] or r["status"] != "completed"]
    if excluded:
        print(f"\nnot in the completed-run timing table ({len(excluded)}; retained in quality/cost totals): "
              + ", ".join(f"{r['task_id']}({'warm-up' if r['is_first_in_dir'] else r['status']})"
                          for r in excluded[:6]))
    missing = [r for r in rows if r["reasoning_tokens_missing"]]
    if missing:
        print(f"WARNING: {len(missing)} run(s) produced reasoning with no token count; "
              "thinking tokens are unknown for those, not zero")
    approx = [r for r in rows if r["has_approx_fixture"]]
    if approx:
        print(f"WARNING: {len(approx)} run(s) used an approximate fixture match")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dir(s); globs allowed")
    ap.add_argument("--out", required=True,
                    help="CSV path; a _YYYYmmdd_HHMMSS suffix is added unless --no-timestamp")
    ap.add_argument("--no-timestamp", action="store_true")
    ap.add_argument("--calls", action="store_true", help="also write a per-LLM-call CSV")
    ap.add_argument("--no-texts", action="store_true",
                    help="skip <out>_texts.csv / <out>_transcripts.jsonl (prompt, reasoning and "
                         "answer of every run)")
    ap.add_argument("--tasks", default="dataset/tasks/phase1.json",
                    help="task annotations to score answers against")
    ap.add_argument("--judge-id", default="", help="select primary judge when multiple judges are stored")
    args = ap.parse_args()
    global PRIMARY_JUDGE_ID
    PRIMARY_JUDGE_ID = args.judge_id

    load_tasks(args.tasks)
    dirs: list[Path] = []
    for pattern in args.runs:
        for p in sorted(glob.glob(pattern)):
            if (Path(p) / "run.jsonl").exists() or (Path(p) / "plan.jsonl").exists():
                dirs.append(Path(p))
    if not dirs:
        raise SystemExit(f"no run directories with run.jsonl or plan.jsonl matched {args.runs}")
    try:
        PRIMARY_JUDGE_ID = select_primary_judge([e for d in dirs for e in read_jsonl(d / "eval.jsonl")], args.judge_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rows: list[dict[str, Any]] = []
    load_energy(dirs)
    for d in dirs:
        found = flatten_dir(d)
        rows += found
        print(f"  {d}: {len(found)} run(s)")

    rows.sort(key=lambda r: (r["run_dir_path"], r["run_order_idx"]))
    out = Path(args.out)
    if not args.no_timestamp:
        # Exports never overwrite each other; results_20260918_193012.csv etc.
        from agent.telemetry import file_stamp
        out = out.with_name(f"{out.stem}_{file_stamp()}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows x {len(COLUMNS)} columns)")
    report_path = out.with_suffix(".quality.json")
    report_path.write_text(json.dumps(quality_report(dirs, rows), indent=2) + "\n")
    print(f"wrote {report_path} (planned-run coverage and quality completeness)")

    if args.calls:
        calls_out = out.with_name(out.stem + "_llm_calls.csv")
        n = export_calls(dirs, calls_out)
        print(f"wrote {calls_out}  ({n} rows)")

    if not args.no_texts:
        texts_out = out.with_name(out.stem + "_texts.csv")
        trace_out = out.with_name(out.stem + "_transcripts.jsonl")
        n = export_texts(dirs, rows, texts_out, trace_out)
        print(f"wrote {texts_out} and {trace_out.name}  ({n} runs: prompt, reasoning, answer)")

    summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
