"""One readable summary of an exported results.csv: outcomes, failures, tokens,
prompt composition, latency and energy, per thinking mode and framework.

    python scripts/summarize_results.py --results output/runs/<sweep>/results.csv \\
        --energy output/runs/<sweep>/energy.csv --out output/runs/<sweep>/summary.md

Rates use every run in the cell as the denominator (failures, truncations and
pending judgments included), as dataset/policies/phase1_v2.json requires.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FAILURES = ["success", "wrong_answer", "format_failure", "tool_failure", "step_limit",
            "output_truncation", "timeout", "framework_error", "pending_judge"]
PROMPT = ["system_prompt_tokens", "conversation_history_tokens", "tool_response_tokens",
          "current_user_prompt_tokens", "other_prompt_tokens"]


def is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower() == "true"


def table(df: pd.DataFrame) -> str:
    return df.to_markdown()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--energy", default="", help="per-run phase split from attribute_energy.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.results)
    d["mode"] = d["reasoning_effort"].fillna("").replace("", pd.NA).fillna(
        d["thinking_setting"].fillna("off"))
    keys = ["mode", "framework"]
    g = d.groupby(keys)
    out = [f"# Results summary\n\nSource: `{args.results}` ({len(d)} runs)\n"]

    out.append("## Outcomes (share of all runs in the cell)\n")
    out.append("`semantic_correct`: the answer is right, whatever its format or how the run ended. "
               "`success`: run completed and semantically correct. `format_ok`: the reply followed "
               "the requested format. `strict_success`: success, correct format, and a natural "
               "finish (no forced final answer, step limit or truncation).\n")
    o = pd.DataFrame({
        "runs": g.size(),
        "semantic_correct": g["semantic_correct"].apply(lambda s: is_true(s).mean()),
        "success": g["success"].apply(lambda s: is_true(s).mean()),
        "format_ok": g["format_ok"].apply(lambda s: is_true(s).sum() / max(1, s.notna().sum())),
        "strict_success": g["strict_success"].apply(lambda s: is_true(s).mean()),
    }).round(3)
    out.append(table(o) + "\n")

    out.append("## Failure categories (run counts)\n")
    fc = pd.crosstab([d["mode"], d["framework"]], d["failure_category"])
    out.append(table(fc[[c for c in FAILURES if c in fc.columns]]) + "\n")

    out.append("## Tokens per run (median)\n")
    t = g[["n_llm_calls", "n_tool_calls", "input_tokens", "output_tokens", "reasoning_tokens"]].median()
    out.append(table(t.round(0)) + "\n")

    out.append("## Prompt composition (share of all input tokens)\n")
    out.append("`other_prompt_tokens` = chat-template markup plus the tool definitions: the part of "
               "`prompt_tokens` that is not message text.\n")
    pc = g[PROMPT].sum()
    out.append(table(pc.div(pc.sum(axis=1), axis=0).round(3)) + "\n")

    out.append("## Latency per run (median seconds)\n")
    lat = (g[["e2e_ms", "t_llm_ms", "t_tool_ms", "t_orch_ms"]].median() / 1000).round(2)
    out.append(table(lat) + "\n")

    out.append("## Energy\n")
    e = pd.DataFrame({
        "gpu_kj_per_run": g["energy_gpu_j"].median() / 1000,
        "cpu_kj_per_run": g["energy_cpu_j"].median() / 1000,
        "total_kj_per_run": g["energy_total_j"].median() / 1000,
        "total_kj_per_success": g.apply(lambda x: x["energy_total_j"].sum() / max(1, is_true(x["success"]).sum()) / 1000),
        "gpu_kj_per_success": g.apply(lambda x: x["energy_gpu_j"].sum() / max(1, is_true(x["success"]).sum()) / 1000),
    }).round(2)
    out.append("Gross energy per run (both GPUs incl. idle draw; CPU = both sockets' RAPL "
               "package counters). Per-success figures charge failed runs to the successes.\n")
    out.append(table(e) + "\n")
    method = d["energy_method"].dropna().unique()
    if len(method):
        out.append(f"Method: {method[0]}\n")

    if args.energy and Path(args.energy).exists():
        en = pd.read_csv(args.energy)
        en = en[en["overlap_runs"] == 0].merge(d[["run_id", "mode"]], on="run_id")
        cols = [c for c in ("gpu_active_llm_j", "gpu_active_tool_j", "gpu_active_orch_j") if c in en.columns]
        if cols:
            out.append("### GPU energy above idle, by phase (share)\n")
            ph = en.groupby(["mode", "framework"])[cols].sum()
            out.append(table(ph.div(ph.sum(axis=1), axis=0).round(3)) + "\n")
        util = [c for c in ("gpu_util_llm", "gpu_util_orch", "cpu_busy_llm", "cpu_busy_tool", "cpu_busy_orch")
                if c in en.columns]
        if util:
            out.append("### Which part is busy, by phase (mean GPU utilisation %, CPU busy fraction)\n")
            out.append(table(en.groupby("mode")[util].mean().round(2)) + "\n")

    Path(args.out).write_text("\n".join(out))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
