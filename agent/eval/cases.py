"""Turn a stored run into a DeepEval test case, from framework-neutral telemetry.

Everything comes from the proxy's saved request bodies and the shared tool
layer's rows, never from a framework's own trace. The same builder therefore
serves LangGraph, CrewAI and the Agents SDK alike, and the judge sees exactly
what the model saw.

    input             the task question
    actual_output     the run's final answer
    expected_output   the task's reference answer
    tools_called      each tool call: name, arguments, and the output returned
    expected_tools    the acceptable tool set closest to what was called
    retrieval_context every tool output, in order: what the answer must be
                      faithful to
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepeval.test_case import LLMTestCase, ToolCall


def tool_outputs(run_dir: Path, run_id: str) -> list[str]:
    """Tool results as the model received them, from the last request body.

    Each request carries the whole conversation so far, so the final one holds
    every tool result the run produced, in order.
    """
    requests = sorted((run_dir / "blobs" / run_id).glob("*_request.json"))
    if not requests:
        return []
    messages = json.loads(requests[-1].read_text()).get("messages", [])
    return [str(m.get("content", "")) for m in messages if m.get("role") == "tool"]


def closest_tool_set(called: list[str], acceptable: list[list[str]]) -> list[str]:
    """The acceptable set that best matches the calls actually made.

    A task can have several valid routes (e.g. retrieve, or retrieve then web).
    Scoring against the nearest one avoids penalising a legitimate alternative.
    """
    if not acceptable:
        return []
    names = set(called)
    return max(acceptable, key=lambda s: (len(names & set(s)), -len(set(s) ^ names)))


def build_case(run_dir: Path, run: dict[str, Any], tool_rows: list[dict[str, Any]],
               task: dict[str, Any]) -> LLMTestCase:
    outputs = tool_outputs(run_dir, run["run_id"])
    rows = sorted(tool_rows, key=lambda t: t["tool_idx"])
    called = [
        ToolCall(name=r["tool_name"], input_parameters=json.loads(r["args_json"]), output=o)
        for r, o in zip(rows, outputs + [""] * (len(rows) - len(outputs)))
    ]
    expected = closest_tool_set([c.name for c in called], task.get("acceptable_tool_sets", []))
    return LLMTestCase(
        input=task["question"],
        actual_output=run.get("final_answer") or "",
        expected_output=task.get("reference_answer") or None,
        retrieval_context=outputs or None,
        tools_called=called or None,
        expected_tools=[ToolCall(name=n) for n in expected] or None,
        name=f"{run['run_id']}:{task['task_id']}",
    )
