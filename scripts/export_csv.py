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

COLUMNS = [
    # identity
    "run_id", "config_id", "run_dir", "run_code", "framework", "framework_version",
    "architecture", "model_repo", "precision", "tp_size",
    "task_id", "repeat_idx", "run_order_idx", "is_first_in_dir",
    # outcome
    "status", "error", "final_answer",
    # deterministic success (agent/scoring.py): "" when the task is judge_only
    "task_type", "check_mode", "success", "success_detail",
    # LLM judge (DeepEval, scripts/judge_runs.py); empty where not yet scored
    "judge_faithfulness", "judge_faithfulness_pass",
    "judge_task_completion", "judge_task_completion_pass",
    "judge_argument_correctness", "judge_argument_correctness_pass",
    "judge_tool_correctness", "judge_tool_correctness_pass",
    "judge_model", "judge_calls", "judge_input_tokens", "judge_output_tokens",
    # trajectory
    "n_llm_calls", "n_tool_calls", "n_failed_tool_calls", "n_malformed_tool_calls",
    "n_tool_switches", "tools_used", "hit_step_limit",
    # tokens
    # output_tokens = reasoning_tokens + answer_tokens. Thinking tokens are
    # generated and paid for like any other output, so they stay in the total.
    "input_tokens", "output_tokens", "reasoning_tokens", "answer_tokens",
    "reasoning_share_pct", "reasoning_chars", "reasoning_tokens_missing",
    "cached_input_tokens", "total_tokens", "prefix_cache_hit_rate", "tokens_per_llm_call",
    # latency
    "e2e_ms", "t_llm_ms", "t_tool_ms", "t_orch_ms", "t_queue_ms",
    "llm_share_pct", "orch_share_pct",
    "ttft_first_ms", "ttft_mean_ms", "tpot_mean_ms", "llm_latency_max_ms",
    # resources
    "proc_cpu_ms", "peak_rss_mb", "energy_gpu_j", "energy_cpu_j",
    # fixtures
    "fixture_exact", "fixture_normalized", "fixture_nearest", "fixture_live",
    "has_approx_fixture",
    # provenance
    "system_prompt_hash", "tool_schema_hash", "tool_parser", "reasoning_parser",
    "thinking_setting", "reasoning_effort", "seed", "temperature", "max_llm_calls",
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


def load_tasks(path: str) -> None:
    """Task annotations, so each run can be scored against its answer check."""
    p = Path(path)
    if p.exists():
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
            "run_code": r.get("run_code", ""),
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
            **_success(r),
            **_judge(evals_by_run.get(r["run_id"], [])),

            "n_llm_calls": r["n_llm_calls"],
            "n_tool_calls": r["n_tool_calls"],
            "n_failed_tool_calls": r["n_failed_tool_calls"],
            "n_malformed_tool_calls": r["n_malformed_tool_calls"],
            "n_tool_switches": r["n_tool_switches"],
            "tools_used": "|".join(t["tool_name"] for t in tools),
            "hit_step_limit": r["status"] == "step_limit",

            "input_tokens": r["input_tokens"],
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
            "energy_gpu_j": r.get("energy_gpu_j") or "",
            "energy_cpu_j": r.get("energy_cpu_j") or "",

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
            "start_ts": r["start_ts"],
            "end_ts": r["end_ts"],
            "harness_version": r.get("harness_version", ""),
            "schema_version": r.get("schema_version", ""),
        })
    return rows


def _judge(evals: list[dict[str, Any]]) -> dict[str, Any]:
    """Latest judge verdict per metric, and the total cost of judging this run."""
    out: dict[str, Any] = {}
    latest: dict[str, dict] = {}
    for e in evals:
        latest[e["metric_name"]] = e  # rows are appended, so the last one wins
    for name, e in latest.items():
        out[f"judge_{name}"] = e["score"]
        out[f"judge_{name}_pass"] = e["passed"]
    if latest:
        out["judge_model"] = next(iter(latest.values())).get("judge_model", "")
        out["judge_calls"] = sum(e.get("judge_calls", 0) for e in latest.values())
        out["judge_input_tokens"] = sum(e.get("judge_input_tokens", 0) for e in latest.values())
        out["judge_output_tokens"] = sum(e.get("judge_output_tokens", 0) for e in latest.values())
    return out


def _success(r: dict[str, Any]) -> dict[str, Any]:
    from agent.scoring import score

    task = TASKS.get(r["task_id"])
    if task is None:
        return {"task_type": "", "check_mode": "", "success": "", "success_detail": "task not found"}
    s = score(r.get("final_answer", ""), task["answer_check"], r["status"])
    return {
        "task_type": task.get("task_type", ""),
        "check_mode": s.mode,
        "success": "" if s.success is None else s.success,
        "success_detail": s.detail[:160],
    }


def export_calls(dirs: list[Path], out: Path) -> int:
    """Long-form per-LLM-call CSV, for looking at trajectories step by step."""
    cols = ["run_dir", "run_id", "call_idx", "request_ts", "ttft_ms", "latency_ms",
            "prompt_tokens", "completion_tokens", "reasoning_tokens", "answer_tokens",
            "reasoning_chars", "answer_chars", "cached_tokens",
            "tpot_ms", "finish_reason", "n_tool_calls_requested", "tool_names",
            "n_messages", "prompt_chars", "contains_reasoning_in_history"]
    rows = []
    for d in dirs:
        for c in read_jsonl(d / "llm_call.jsonl"):
            row = {k: c.get(k, "") for k in cols}
            row["run_dir"] = d.name
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
    if not warm:
        return

    # Grouped by everything that defines a cell, so a sweep over thinking modes
    # (or models) is never averaged together into one row.
    by_fw: dict[str, list[dict]] = {}
    for r in warm:
        think = r["thinking_setting"] or "?"
        if r.get("reasoning_effort"):
            think += f"/{r['reasoning_effort']}"
        model = (r["model_repo"] or "?").split("/")[-1]
        key = f"{model[:22]} {r['framework'] or '?'} {think}"
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
    # by the number of successful runs. Only deterministically scored tasks.
    # Warm-up runs are included here: warm-up inflates timing (imports, JIT),
    # not correctness or token counts, and with one repeat per task, dropping
    # it would drop that task from every cell.
    print(f"\n{'cell':45s} {'scored':>7s} {'success':>8s} {'tok/success':>12s} {'s/success':>10s}")
    print("-" * 86)
    cells: dict[str, list[dict]] = {}
    for r in rows:
        think = r["thinking_setting"] or "?"
        if r.get("reasoning_effort"):
            think += f"/{r['reasoning_effort']}"
        cells.setdefault(f"{(r['model_repo'] or '?').split('/')[-1][:22]} {r['framework']} {think}", []).append(r)
    for key, rs in sorted(cells.items()):
        scored = [r for r in rs if r["success"] != ""]
        wins = sum(r["success"] is True for r in scored)
        tok = sum(r["total_tokens"] for r in scored)
        sec = sum(r["e2e_ms"] for r in scored) / 1000
        print(f"{key:45s} {len(scored):7d} {wins / len(scored) if scored else 0:8.0%} "
              f"{(tok / wins) if wins else float('nan'):12,.0f} {(sec / wins) if wins else float('nan'):10.1f}")
    print("(tok/success and s/success charge failed runs to the successes, as the metric requires;\n"
          " warm-up runs are included, adding ~1 s per cell to s/success)")

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
        print(f"\nnot in the table above ({len(excluded)}): "
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
    ap.add_argument("--tasks", default="dataset/tasks/phase1.json",
                    help="task annotations to score answers against")
    args = ap.parse_args()

    load_tasks(args.tasks)
    dirs: list[Path] = []
    for pattern in args.runs:
        for p in sorted(glob.glob(pattern)):
            if (Path(p) / "run.jsonl").exists():
                dirs.append(Path(p))
    if not dirs:
        raise SystemExit(f"no run directories with run.jsonl matched {args.runs}")

    rows: list[dict[str, Any]] = []
    for d in dirs:
        found = flatten_dir(d)
        rows += found
        print(f"  {d}: {len(found)} run(s)")

    rows.sort(key=lambda r: (r["run_dir"], r["run_order_idx"]))
    out = Path(args.out)
    if not args.no_timestamp:
        # Exports never overwrite each other; results_20260918_193012.csv etc.
        from datetime import datetime
        out = out.with_name(f"{out.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows x {len(COLUMNS)} columns)")

    if args.calls:
        calls_out = out.with_name(out.stem + "_llm_calls.csv")
        n = export_calls(dirs, calls_out)
        print(f"wrote {calls_out}  ({n} rows)")

    summarize(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
