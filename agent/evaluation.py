"""Framework-neutral quality results; no LLM imports or inference here."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .scoring import score

EVALUATION_VERSION = "phase1-v2-p0-1"
SEMANTIC_METRIC = "semantic_correctness"
SEMANTIC_METRIC_VERSION = "semantic-correctness-v1"
OUTCOME_POLICY_VERSION = "phase1-v2-p0-2"


def semantic_threshold(task: dict) -> float:
    value = task.get("evaluation", {}).get("semantic_threshold", 0.8)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("semantic_threshold must be finite and in [0, 1]")
    return float(value)


def valid_semantic_judgment(entry: dict, task: dict, answer: str) -> bool:
    value = entry.get("score")
    return (entry.get("metric_name") == SEMANTIC_METRIC
            and entry.get("evaluation_version") == EVALUATION_VERSION
            and entry.get("metric_version") == SEMANTIC_METRIC_VERSION
            and entry.get("answer_fingerprint") == answer_fingerprint(task, answer)
            and isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1
            and entry.get("threshold") == semantic_threshold(task)
            and isinstance(entry.get("passed"), bool)
            and entry["passed"] == (value >= semantic_threshold(task)))


def answer_fingerprint(task: dict, answer: str) -> str:
    payload = {k: task.get(k) for k in (
        "question", "reference_answer", "reference_context", "grading_rubric", "answer_check", "evaluation"
    )}
    payload["actual_output"] = answer
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def evaluate(task: dict, answer: str, status: str, tools_used: list[str],
             evals: list[dict] = (), judge_id: str = "", *, completion_type: str | None = None) -> dict[str, Any]:
    """Unify deterministic and judged correctness without a format gate.

    Missing judgments remain pending. Unparseable deterministic answers are
    explicit unscorable failures in the full denominator, with unknown content
    correctness. Only the dedicated semantic metric can fill a judge-only slot;
    faithfulness and task-completion diagnostics are never substituted.
    """
    verdict = score(answer, task["answer_check"], status, tools_used)
    semantic, source, state = verdict.semantic_correct, "deterministic", "scored"
    detail = verdict.detail
    if verdict.mode == "judge_only":
        source = "semantic_judge"
        if not answer.strip():
            semantic, source = False, "empty_answer"
        else:
            candidates = [e for e in evals if valid_semantic_judgment(e, task, answer)
                          and (not judge_id or e.get("judge_id") == judge_id)]
            identities = {e.get("judge_id", "") for e in candidates}
            if len(identities) > 1:
                state, detail = "ambiguous_judge", "select --judge-id for the primary semantic score"
            elif candidates:
                latest = candidates[-1]
                semantic = latest["score"] >= semantic_threshold(task)
                detail = latest.get("rationale", "")
            else:
                state, detail = "pending_judge", "semantic correctness judgment missing or stale"
    elif semantic is None:
        state = "unscorable_format"

    completed = status == "completed"
    success = semantic if completed else False
    if state == "unscorable_format":
        success = False
    format_success = False if success is False or verdict.format_ok is False else success
    natural = completed and completion_type in (None, "natural_completion")
    strict = format_success if natural else False
    return {
        "task_type": task.get("task_type", ""), "check_mode": verdict.mode,
        "success": "" if success is None else success,
        "success_detail": detail[:300], "score_source": source, "score_state": state,
        "semantic_correct": "" if semantic is None else semantic,
        "format_ok": "" if verdict.format_ok is None else verdict.format_ok,
        "execution_completed": completed,
        "strict_success": "" if strict is None else strict,
        "format_success": "" if format_success is None else format_success,
        "natural_completion": natural,
        "evaluation_version": EVALUATION_VERSION,
        "outcome_policy_version": OUTCOME_POLICY_VERSION,
    }
