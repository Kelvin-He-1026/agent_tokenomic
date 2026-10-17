"""OpenAI-compatible logging proxy: the source of record for LLM telemetry.

Every framework points at this instead of vLLM. It enforces the bound run's
call and output caps and records what actually went over the wire: prompts, token counts,
TTFT, latency, and whether reasoning content was echoed back into the message
history. Framework self-reports are never used.

Run binding: the harness POSTs /control/run before each run. At concurrency 1
that is unambiguous; an X-Run-Id header takes precedence when present, which is
what Phase 4 will use.

    uvicorn agent.proxy.server:app --port 8100
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..telemetry import LlmCallRecord, TelemetryWriter, monotonic_ms, new_id, now_ms

UPSTREAM = os.getenv("ATK_UPSTREAM", "http://127.0.0.1:8000")
OUT_DIR = os.getenv("ATK_OUT_DIR", "output/runs/scratch")
SAVE_BLOBS = os.getenv("ATK_SAVE_BLOBS", "1") == "1"

app = FastAPI(title="agent-tokenomic proxy")
state: dict[str, Any] = {
    "run_id": "unbound",
    "call_idx": 0,
    "out_dir": OUT_DIR,
    "inflight": 0,
    "session": "",
    "last_bind": 0.0,
}
# A second harness binding while one is active means two agents share this
# proxy. Their llm_call rows would be attributed to whichever run bound last,
# and their requests would queue behind each other, so Phase 1 would no longer
# be at concurrency 1. Sessions are refused rather than silently interleaved.
SESSION_TTL_S = float(os.getenv("ATK_SESSION_TTL", "600"))
_writers: dict[str, TelemetryWriter] = {}
client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))


def writer_for(out_dir: str) -> TelemetryWriter:
    """One writer per output directory, created on demand.

    The harness owns the run directory, so the proxy follows whatever it is
    told rather than its own env default. Without this, llm_call rows land
    somewhere other than the run rows and the joins come up empty.
    """
    if out_dir not in _writers:
        _writers[out_dir] = TelemetryWriter(out_dir)
    return _writers[out_dir]


@app.post("/control/run")
async def bind_run(request: Request) -> dict[str, Any]:
    """Bind subsequent requests to a run_id and an output directory."""
    body = await request.json()
    session = body.get("session", "")
    now = time.time()

    held = state["session"]
    if session and held and session != held and (now - state["last_bind"]) < SESSION_TTL_S:
        return JSONResponse(
            status_code=409,
            content={
                "conflict": True,
                "holder": held,
                "held_for_s": round(now - state["last_bind"], 1),
                "detail": "another harness session is using this proxy",
            },
        )

    if state["inflight"]:
        return JSONResponse(status_code=409, content={"detail": "previous run still has inflight requests"})
    limits = body.get("limits") or {}
    if limits:
        for key in ("max_llm_calls", "max_tokens_per_call", "max_total_output_tokens"):
            value = limits.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                return JSONResponse(status_code=400, content={"detail": f"invalid {key}"})
        if not isinstance(limits.get("task_timeout_s"), (int, float)) or limits["task_timeout_s"] <= 0:
            return JSONResponse(status_code=400, content={"detail": "invalid task_timeout_s"})
    state["session"] = session or held
    state["last_bind"] = now
    state["run_id"] = body.get("run_id", "unbound")
    state["out_dir"] = body.get("out_dir") or state["out_dir"]
    state["call_idx"] = 0
    state["limits"] = limits
    state["forced_final_markers"] = body.get("forced_final_markers") or []
    state["output_tokens"] = 0
    state["blocked_requests"] = 0
    state["termination_reason"] = ""
    state["deadline"] = time.monotonic() + limits["task_timeout_s"] if limits else None
    return {"bound": state["run_id"], "out_dir": state["out_dir"], "session": state["session"],
            "accounting_policy": "proxy-v1", "limits": limits}


@app.post("/control/release")
async def release(request: Request) -> dict[str, Any]:
    """Drop the session lock when a harness finishes."""
    body = await request.json()
    if body.get("session") == state["session"]:
        state["session"] = ""
    return {"released": True}


@app.get("/control/pending")
async def pending() -> dict[str, Any]:
    """How many calls are still being recorded.

    A streamed response finishes for the client before the proxy has written
    its llm_call row, so the harness waits on this before reading telemetry.
    Without it, the last call of a run is missing from the joins.
    """
    return {k: state.get(k, 0) for k in (
        "inflight", "run_id", "call_idx", "output_tokens", "blocked_requests", "termination_reason"
    )}


@app.get("/control/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "upstream": UPSTREAM,
        "run_id": state["run_id"],
        "out_dir": state["out_dir"],
        "accounting_policy": "proxy-v1",
    }


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def forward(path: str, request: Request):
    body = await request.body()
    run_id = request.headers.get("x-run-id") or state["run_id"]
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "x-run-id"}
    }

    if request.method == "GET" or path != "chat/completions":
        resp = await client.request(
            request.method, f"{UPSTREAM}/v1/{path}", content=body, headers=headers
        )
        return JSONResponse(
            status_code=resp.status_code,
            content=resp.json() if resp.content else {},
        )

    payload = json.loads(body) if body else {}
    if run_id != state["run_id"]:
        return JSONResponse(status_code=409, content={"error": {"message": "run binding mismatch"}})
    limits = state.get("limits") or {}
    requested_cap = payload.get("max_completion_tokens", payload.get("max_tokens"))
    effective_cap = requested_cap
    if limits:
        reason = ""
        if state.get("termination_reason"):
            reason = state["termination_reason"]
        elif time.monotonic() >= state["deadline"]:
            reason = "timeout"
        elif state["call_idx"] >= limits["max_llm_calls"]:
            reason = "call_cap"
        elif state["output_tokens"] >= limits["max_total_output_tokens"]:
            reason = "output_cap"
        elif state["inflight"] or payload.get("n", 1) != 1:
            reason = "concurrency_violation"
        if reason:
            state["blocked_requests"] += 1
            state["termination_reason"] = reason
            return JSONResponse(status_code=429, content={"error": {
                "message": f"benchmark stopped request: {reason}", "type": "benchmark_limit", "code": reason
            }})
        effective_cap = min(limits["max_tokens_per_call"],
                            limits["max_total_output_tokens"] - state["output_tokens"])
        # Use one canonical spelling and value across all frameworks. Thinking
        # tokens are part of this completion budget, as reported by upstream.
        payload.pop("max_tokens", None)
        payload["max_completion_tokens"] = effective_cap
        body = json.dumps(payload).encode()

    # vLLM omits usage from a streamed response unless asked, and frameworks
    # differ in whether they ask. Without this every token count is zero, so
    # the proxy requests usage itself, in addition to enforcing output caps.
    if payload.get("stream"):
        options = dict(payload.get("stream_options") or {})
        if not options.get("include_usage"):
            options["include_usage"] = True
            payload["stream_options"] = options
            body = json.dumps(payload).encode()

    writer = writer_for(state["out_dir"])
    state["call_idx"] += 1
    state["inflight"] += 1
    call = LlmCallRecord(
        call_id=new_id("lc_"),
        run_id=run_id,
        call_idx=state["call_idx"],
        request_ts=now_ms(),
        requested_output_cap=requested_cap,
        effective_output_cap=effective_cap,
    )
    _describe_request(call, payload)
    if SAVE_BLOBS:
        p = writer.blob_path(run_id, f"{call.call_idx:03d}_request.json")
        p.write_text(json.dumps(payload, indent=2))
        call.payload_path = str(p)

    started = monotonic_ms()
    if payload.get("stream"):
        return StreamingResponse(
            _stream(call, started, body, headers, run_id, writer),
            media_type="text/event-stream",
        )

    data = {}
    try:
        resp = await client.post(f"{UPSTREAM}/v1/chat/completions", content=body, headers=headers)
        call.http_status = resp.status_code
        data = resp.json() if resp.content else {}
        if resp.is_error:
            call.error = f"upstream HTTP {resp.status_code}"
        _describe_response(call, data)
        return JSONResponse(status_code=resp.status_code, content=data)
    except Exception as exc:
        call.error = f"{type(exc).__name__}: {exc}"
        call.usage_missing = True
        raise
    finally:
        call.latency_ms = monotonic_ms() - started
        _finish(call, data, run_id, writer)


async def _stream(
    call: LlmCallRecord,
    started: float,
    body: bytes,
    headers: dict,
    run_id: str,
    writer: TelemetryWriter,
):
    """Pass SSE chunks through untouched while timestamping the first one."""
    chunks: list[dict[str, Any]] = []
    try:
        async with client.stream(
            "POST", f"{UPSTREAM}/v1/chat/completions", content=body, headers=headers
        ) as resp:
            call.http_status = resp.status_code
            if resp.is_error:
                call.error = f"upstream HTTP {resp.status_code}"
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    blob = line[6:].strip()
                    if blob and blob != "[DONE]":
                        if call.ttft_ms is None:
                            call.ttft_ms = monotonic_ms() - started
                        try:
                            chunks.append(json.loads(blob))
                        except json.JSONDecodeError:
                            pass
                yield (line + "\n\n").encode() if line else b"\n"
    except Exception as exc:
        call.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Also reached when the client closes the stream as soon as it has read
        # [DONE], which cancels this generator at a yield. Without it the row
        # is lost and inflight never returns to zero, so every later run waits
        # out the flush timeout.
        call.latency_ms = monotonic_ms() - started
        merged = _merge_chunks(chunks)
        _describe_response(call, merged)
        _finish(call, merged, run_id, writer)


def _describe_request(call: LlmCallRecord, payload: dict[str, Any]) -> None:
    messages = payload.get("messages", [])
    call.n_messages = len(messages)
    call.prompt_chars = len(json.dumps(messages))
    call.forced_final_answer = any(
        m.get("role") == "user" and isinstance(m.get("content"), str)
        and m["content"] in state.get("forced_final_markers", []) for m in messages
    )
    # Residual row 2: did the framework put reasoning back into the history?
    call.contains_reasoning_in_history = any(
        m.get("reasoning_content") or m.get("reasoning") or "<think>" in str(m.get("content", ""))
        for m in messages
        if m.get("role") == "assistant"
    )


def _describe_response(call: LlmCallRecord, data: dict[str, Any]) -> None:
    usage = data.get("usage") or {}
    call.usage_missing = "completion_tokens" not in usage or "prompt_tokens" not in usage
    call.prompt_tokens = usage.get("prompt_tokens", 0)
    call.completion_tokens = usage.get("completion_tokens", 0)
    details = usage.get("completion_tokens_details") or {}
    call.reasoning_tokens = details.get("reasoning_tokens", 0) or 0
    prompt_details = usage.get("prompt_tokens_details") or {}
    call.cached_tokens = prompt_details.get("cached_tokens", 0) or 0

    choices = data.get("choices") or [{}]
    choice = choices[0] if choices else {}
    call.finish_reason = choice.get("finish_reason") or ""
    message = choice.get("message") or {}

    # Reasoning is measured whether or not the server reports its token count:
    # characters always, tokens when usage details carry them.
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    call.reasoning_chars = len(reasoning)
    call.answer_chars = len(message.get("content") or "")
    tool_calls = message.get("tool_calls") or []
    call.n_tool_calls_requested = len(tool_calls)
    call.tool_names = [tc.get("function", {}).get("name", "") for tc in tool_calls]

    if call.completion_tokens > 1 and call.ttft_ms is not None:
        call.tpot_ms = (call.latency_ms - call.ttft_ms) / (call.completion_tokens - 1)


def _merge_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Reassemble a streamed response into the non-streaming shape."""
    content, reasoning, finish_reason, usage = "", "", "", {}
    tool_calls: dict[int, dict[str, Any]] = {}
    for ch in chunks:
        if ch.get("usage"):
            usage = ch["usage"]
        for choice in ch.get("choices", []):
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            # vLLM has used both names across versions; read either.
            reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""
            finish_reason = choice.get("finish_reason") or finish_reason
            for tc in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(
                    tc.get("index", 0), {"function": {"name": "", "arguments": ""}}
                )
                fn = tc.get("function") or {}
                slot["function"]["name"] += fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
    message: dict[str, Any] = {"content": content}
    if reasoning:
        message["reasoning"] = reasoning
    if tool_calls:
        message["tool_calls"] = [tool_calls[k] for k in sorted(tool_calls)]
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


def _finish(
    call: LlmCallRecord, data: dict[str, Any], run_id: str, writer: TelemetryWriter
) -> None:
    try:
        if SAVE_BLOBS:
            p = writer.blob_path(run_id, f"{call.call_idx:03d}_response.json")
            p.write_text(json.dumps(data, indent=2))
            call.response_path = str(p)
        writer.write(call)
    finally:
        if run_id == state["run_id"]:
            state["output_tokens"] = state.get("output_tokens", 0) + call.completion_tokens
            if state.get("limits") and not call.error:
                if call.usage_missing:
                    state["termination_reason"] = "usage_missing"
                elif call.effective_output_cap and call.completion_tokens > call.effective_output_cap:
                    state["termination_reason"] = "output_cap_violation"
        state["inflight"] = max(0, state["inflight"] - 1)
