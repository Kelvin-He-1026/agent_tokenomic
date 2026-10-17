"""One interpretation of resource limits and termination for live/stored runs."""
from __future__ import annotations

from typing import Any


def completion_type(status: str, reason: str = "", forced: bool = False,
                    last_finish_reason: str = "") -> str:
    if status == "completed":
        if forced or reason == "forced_final_answer":
            return "forced_final"
        if reason == "natural_stop" or (not reason and last_finish_reason == "stop"):
            return "natural_completion"
        return "unknown_completion"
    return {"step_limit": "step_limit", "truncated": "truncated", "timeout": "timeout"}.get(status, "error")


def cap_hits(run: dict, calls: list[dict], config: dict | None = None) -> dict[str, Any]:
    """Reconstruct cap hits from actual finish reasons and token usage.

    ``output_cap_reached`` is the union, never an alias for the run-total cap.
    A normal stop exactly on a limit remains a completion; reaching a limit
    and being terminated by it are separate facts.
    """
    cfg = config or {}
    per_call = run.get("max_tokens_per_call") or cfg.get("max_tokens") or 0
    call_limit = run.get("max_llm_calls") or cfg.get("max_llm_calls") or 0
    total_limit = (run.get("max_total_output_tokens") or cfg.get("max_total_output_tokens")
                   or per_call * call_limit)
    ordered = sorted(calls, key=lambda c: c.get("call_idx", 0))
    used = sum(c.get("completion_tokens", 0) or 0 for c in ordered) if ordered else run.get("output_tokens", 0)
    per_hit = False
    total_hit = bool(total_limit and used >= total_limit)
    unknown_hit = False
    for call in ordered:
        count = call.get("completion_tokens", 0) or 0
        effective = call.get("effective_output_cap") or call.get("requested_output_cap") or per_call
        if per_call and count >= per_call:
            per_hit = True
        if call.get("finish_reason") == "length":
            if per_call and effective == per_call:
                per_hit = True
            elif per_call and effective and effective < per_call:
                # The proxy shrinks only for the remaining run-total budget.
                total_hit = True
            else:
                unknown_hit = True
    reason = run.get("termination_reason", "")
    if reason == "per_call_output_cap":
        per_hit = True
    if reason == "output_cap":
        total_hit = True
    any_hit = per_hit or total_hit or unknown_hit or run.get("status") == "truncated"
    cap_type = "both" if per_hit and total_hit else "per_call" if per_hit else "total" if total_hit else "unknown" if any_hit else "none"
    last_reason = ordered[-1].get("finish_reason", "") if ordered else ""
    forced = bool(run.get("forced_final_answer") or reason == "forced_final_answer"
                  or any(c.get("forced_final_answer") for c in ordered))
    return {
        "call_cap_reached": bool(call_limit and (len(ordered) if ordered else run.get("n_llm_calls", 0)) >= call_limit),
        "per_call_output_cap_reached": per_hit,
        "total_output_cap_reached": total_hit,
        "output_cap_reached": bool(any_hit),
        "output_cap_type": cap_type,
        "last_finish_reason": last_reason,
        "forced_final_answer": forced,
        "completion_type": completion_type(run.get("status", ""), reason, forced, last_reason),
    }
