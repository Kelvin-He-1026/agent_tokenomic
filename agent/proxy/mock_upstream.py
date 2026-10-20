"""A scripted OpenAI-compatible server, so the harness runs with no GPU.

It answers /v1/chat/completions with a fixed trajectory per task: call a tool,
read the observation, then answer. That makes every part of the pipeline
testable — adapters, proxy, tool layer, telemetry — before vLLM exists, and it
is also the basis for the optional replay arm that measures orchestration
overhead directly.

    uvicorn agent.proxy.mock_upstream:app --port 8000
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="agent-tokenomic mock upstream")

# Which tool the mock calls for a question, by keyword. Deliberately crude: the
# point is a deterministic trajectory, not a good agent.
KEYWORD_TOOLS = [
    (("release", "cpi", "gdp", "fomc", "macro", "economic"), "fred_releases", {"window": "this_week"}),
    (("news", "latest", "current", "today", "price", "stock"), "web_search", None),
]
DEFAULT_TOOL = "retrieve"

LATENCY_MS = 40.0  # simulated per-call generation time


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": "mock-model", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    messages = payload.get("messages", [])
    stream = bool(payload.get("stream"))

    has_observation = any(m.get("role") == "tool" for m in messages)
    question = _last_user_text(messages)

    if has_observation:
        message = {
            "role": "assistant",
            "content": _answer_from(messages, question),
            "tool_calls": None,
        }
        finish_reason = "stop"
    else:
        offered = [t.get("function", {}).get("name") for t in payload.get("tools") or []]
        name, args = _pick_tool(question, offered)
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        }
        finish_reason = "tool_calls"

    usage = {
        "prompt_tokens": max(1, len(json.dumps(messages)) // 4),
        "completion_tokens": max(1, len(json.dumps(message)) // 4),
        "total_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    if stream:
        return StreamingResponse(
            _sse(message, finish_reason, usage), media_type="text/event-stream"
        )

    time.sleep(LATENCY_MS / 1000.0)
    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": payload.get("model", "mock-model"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }
    )


def _sse(message: dict[str, Any], finish_reason: str, usage: dict[str, Any]):
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    def frame(delta: dict[str, Any], fr: str | None = None) -> bytes:
        body = {
            "id": cid,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": fr}],
        }
        return f"data: {json.dumps(body)}\n\n".encode()

    time.sleep(LATENCY_MS / 1000.0)  # simulated TTFT
    if message.get("tool_calls"):
        tc = message["tool_calls"][0]
        yield frame(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                ]
            }
        )
    else:
        for piece in _chunks(message["content"]):
            yield frame({"content": piece})
    yield frame({}, finish_reason)
    yield f"data: {json.dumps({'id': cid, 'choices': [], 'usage': usage})}\n\n".encode()
    yield b"data: [DONE]\n\n"


def _chunks(text: str, size: int = 24):
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


ATTACHMENT_RE = re.compile(r"Attached file: (\S+?)\.? Open it with the read_file tool")


def _pick_tool(question: str, offered: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    """Pick by keyword, but only among the tools the request offers (a task set
    can offer its own list, e.g. GAIA's web_search/fetch_url/read_file)."""
    offered = [n for n in offered or [] if n]
    usable = (lambda n: n in offered) if offered else (lambda n: True)
    query = {"query": question[:120]}
    attached = ATTACHMENT_RE.search(question)
    if attached and usable("read_file"):
        return "read_file", {"file_name": attached.group(1)}
    q = question.lower()
    for keywords, name, args in KEYWORD_TOOLS:
        if usable(name) and any(k in q for k in keywords):
            return name, (args if args is not None else query)
    if usable(DEFAULT_TOOL):
        return DEFAULT_TOOL, query
    if usable("web_search"):
        return "web_search", query
    return offered[0], query


def _answer_from(messages: list[dict[str, Any]], question: str) -> str:
    observation = ""
    for m in reversed(messages):
        if m.get("role") == "tool":
            observation = str(m.get("content", ""))
            break
    head = observation.strip().splitlines()[:3]
    return f"Based on the tool result: {' '.join(head)[:400]}"
