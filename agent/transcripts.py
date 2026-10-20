"""Each run's actual text: the prompt sent, the model's reasoning, its answer.

Rebuilt from the requests and responses the proxy saved (blobs/<run_id>/), so
it is exactly what went over the wire, whatever the framework.

Per run:
  system_prompt    the system message of the first request
  user_prompt      the user message of the first request, as the framework sent
                   it (CrewAI's "Current Task: ..." wrapper included)
  reasoning        the model's thinking text from every call, in order
                   (empty with thinking off)
  final_answer     the run's final answer, in full
  calls            per call: the messages new since the previous request (the
                   tool results the framework added, any follow-up prompt), the
                   model's visible text, its reasoning and its tool calls
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .prompt_breakdown import message_text


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def run_transcript(run_dir: Path, run: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    blobs = run_dir / "blobs" / run["run_id"]
    out_calls, reasoning, prev = [], [], 0
    system_prompt = user_prompt = ""
    for c in sorted(calls, key=lambda c: c["call_idx"]):
        req = _load(blobs / f"{c['call_idx']:03d}_request.json")
        resp = _load(blobs / f"{c['call_idx']:03d}_response.json")
        messages = req.get("messages") or []
        if c["call_idx"] == 1 or not system_prompt:
            system_prompt = next((message_text(m) for m in messages if m.get("role") == "system"), "")
            user_prompt = next((message_text(m) for m in messages if m.get("role") == "user"), "")
        msg = ((resp.get("choices") or [{}])[0] or {}).get("message") or {}
        think = msg.get("reasoning") or msg.get("reasoning_content") or ""
        if think:
            reasoning.append(f"[call {c['call_idx']}]\n{think}")
        out_calls.append({
            "call_idx": c["call_idx"],
            "new_messages": [{"role": m.get("role"), "text": message_text(m)} for m in messages[prev:]],
            "content": msg.get("content") or "",
            "reasoning": think,
            "tool_calls": [{"name": (t.get("function") or {}).get("name", ""),
                            "arguments": (t.get("function") or {}).get("arguments", "")}
                           for t in msg.get("tool_calls") or []],
            "finish_reason": c.get("finish_reason", ""),
        })
        prev = len(messages)
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "reasoning": "\n\n".join(reasoning),
        "final_answer": run.get("final_answer", ""),
        "calls": out_calls,
    }
