"""OpenAI-compatible logging proxy: the source of record for LLM telemetry.

Every framework points at this instead of vLLM. It enforces the bound run's
call and output caps and records what actually went over the wire: prompts, token counts,
TTFT, latency, and whether reasoning content was echoed back into the message
history. Framework self-reports are never used.

Run binding: the harness POSTs /control/run before each run, and the proxy
keeps that run's budget and counters. Requests reach a run in one of two ways:

    /v1/...            the run named by an X-Run-Id header (Hermes, whose
                       sandbox policy allows only these paths), else the run
                       bound last (concurrency 1)
    /r/<run_id>/v1/... that run, whichever else is bound (concurrent runs of
                       the in-process frameworks, the virtual-user load test):
                       the base URL carries the run, so no framework needs
                       header support

A run is bound with "concurrent": true for the second form; such binds do not
wait for, or displace, the run bound last. /control/end closes a run: any
request it still makes is refused, so an agent process that outlives its run
cannot spend tokens under it.

    uvicorn agent.proxy.server:app --port 8100
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..telemetry import LlmCallRecord, TelemetryWriter, monotonic_ms, new_id, now_ms

UPSTREAM = os.getenv("ATK_UPSTREAM", "http://127.0.0.1:8000")
OUT_DIR = os.getenv("ATK_OUT_DIR", "output/runs/scratch")
SAVE_BLOBS = os.getenv("ATK_SAVE_BLOBS", "1") == "1"

app = FastAPI(title="agent-tokenomic proxy")


@dataclass
class RunState:
    """One bound run: its budget, its counters, and where its rows go."""

    run_id: str
    out_dir: str
    limits: dict[str, Any] = field(default_factory=dict)
    forced_final_markers: list[str] = field(default_factory=list)
    deadline: float | None = None
    call_idx: int = 0
    inflight: int = 0
    output_tokens: int = 0
    blocked_requests: int = 0
    termination_reason: str = ""
    ended: bool = False
    # Who closed the run first: "finished" (the harness, after the agent
    # returned) or a benchmark reason such as "level_end" (a load test's
    # filler, closed while its agent was still working).
    ended_by: str = ""
    # Requests the benchmark refused or cut because the run was closed.
    refused_after_end: int = 0
    # Calls that failed at or past vLLM (HTTP errors, connection errors),
    # not counting streams the benchmark cut.
    upstream_errors: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "inflight": self.inflight, "call_idx": self.call_idx,
                "output_tokens": self.output_tokens, "blocked_requests": self.blocked_requests,
                "termination_reason": self.termination_reason, "ended": self.ended,
                "ended_by": self.ended_by, "refused_after_end": self.refused_after_end,
                "upstream_errors": self.upstream_errors}


# Requests before any bind are recorded under "unbound", with no budget.
runs: dict[str, RunState] = {"unbound": RunState("unbound", OUT_DIR)}
state: dict[str, Any] = {"current": "unbound", "session": "", "last_bind": 0.0}
MAX_KEPT_RUNS = 4096
# A second harness binding while one is active means two harnesses share this
# proxy. Their requests would queue behind each other, so neither would be
# measuring what it thinks. Sessions are refused rather than silently
# interleaved; one session may bind many concurrent runs.
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

    concurrent = bool(body.get("concurrent"))
    if not concurrent and runs[state["current"]].inflight:
        return JSONResponse(status_code=409, content={"detail": "previous run still has inflight requests"})
    limits = body.get("limits") or {}
    if limits:
        for key in ("max_llm_calls", "max_tokens_per_call", "max_total_output_tokens"):
            value = limits.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                return JSONResponse(status_code=400, content={"detail": f"invalid {key}"})
        if not isinstance(limits.get("task_timeout_s"), (int, float)) or limits["task_timeout_s"] <= 0:
            return JSONResponse(status_code=400, content={"detail": "invalid task_timeout_s"})
    run_id = body.get("run_id", "unbound")
    run = RunState(
        run_id=run_id,
        out_dir=body.get("out_dir") or runs[state["current"]].out_dir,
        limits=limits,
        forced_final_markers=body.get("forced_final_markers") or [],
        deadline=time.monotonic() + limits["task_timeout_s"] if limits else None,
    )
    runs[run_id] = run
    _prune()
    state["session"] = session or held
    state["last_bind"] = now
    if not concurrent:
        state["current"] = run_id
    return {"bound": run_id, "out_dir": run.out_dir, "session": state["session"],
            "accounting_policy": "proxy-v1", "limits": limits}


def _prune() -> None:
    """Forget the oldest ended runs once many are kept (a long load test)."""
    excess = len(runs) - MAX_KEPT_RUNS
    keep = {"unbound", state["current"]}
    for run_id in [r for r, st in runs.items()
                   if st.ended and not st.inflight and r not in keep][:max(0, excess)]:
        del runs[run_id]


@app.post("/control/end")
async def end_run(request: Request) -> dict[str, Any]:
    """Close a run. Its later requests are refused (run_ended), so an agent
    process that outlives its run spends nothing under it, and the counters
    the harness reads next stay final. reason (default "finished") is kept
    from the first close only."""
    body = await request.json()
    run = runs.get(body.get("run_id", ""))
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "unknown run"})
    if not run.ended:
        run.ended, run.ended_by = True, str(body.get("reason") or "finished")
    return run.snapshot()


@app.post("/control/release")
async def release(request: Request) -> dict[str, Any]:
    """Drop the session lock when a harness finishes."""
    body = await request.json()
    if body.get("session") == state["session"]:
        state["session"] = ""
    return {"released": True}


@app.get("/control/pending")
async def pending(run_id: str = "") -> dict[str, Any]:
    """How many of a run's calls are still being recorded (default: the run
    bound last).

    A streamed response finishes for the client before the proxy has written
    its llm_call row, so the harness waits on this before reading telemetry.
    Without it, the last call of a run is missing from the joins.
    """
    run = runs.get(run_id or state["current"])
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "unknown run"})
    return run.snapshot()


@app.get("/control/health")
async def health() -> dict[str, Any]:
    current = runs[state["current"]]
    return {
        "ok": True,
        "upstream": UPSTREAM,
        "run_id": current.run_id,
        "out_dir": current.out_dir,
        "accounting_policy": "proxy-v1",
        "active_runs": sum(1 for r in runs.values() if r.run_id != "unbound" and not r.ended),
        "inflight": sum(r.inflight for r in runs.values()),
    }


@app.api_route("/r/{run_id}/v1/{path:path}", methods=["GET", "POST"])
async def forward_run(run_id: str, path: str, request: Request):
    return await _forward(path, request, run_id)


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def forward(path: str, request: Request):
    return await _forward(path, request, request.headers.get("x-run-id") or state["current"])


async def _forward(path: str, request: Request, run_id: str):
    body = await request.body()
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
    run = runs.get(run_id)
    if run is None:
        return JSONResponse(status_code=409, content={"error": {"message": "run binding mismatch"}})
    if run.ended:
        # Not a termination reason: the run is already recorded.
        run.blocked_requests += 1
        run.refused_after_end += 1
        return JSONResponse(status_code=429, content={"error": {
            "message": "benchmark stopped request: run_ended", "type": "benchmark_limit", "code": "run_ended"
        }})
    limits = run.limits
    requested_cap = payload.get("max_completion_tokens", payload.get("max_tokens"))
    effective_cap = requested_cap
    if limits:
        reason = ""
        if run.termination_reason:
            reason = run.termination_reason
        elif time.monotonic() >= run.deadline:
            reason = "timeout"
        elif run.call_idx >= limits["max_llm_calls"]:
            reason = "call_cap"
        elif run.output_tokens >= limits["max_total_output_tokens"]:
            reason = "output_cap"
        elif run.inflight or payload.get("n", 1) != 1:
            reason = "concurrency_violation"
        if reason:
            run.blocked_requests += 1
            run.termination_reason = reason
            return JSONResponse(status_code=429, content={"error": {
                "message": f"benchmark stopped request: {reason}", "type": "benchmark_limit", "code": reason
            }})
        effective_cap = min(limits["max_tokens_per_call"],
                            limits["max_total_output_tokens"] - run.output_tokens)
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

    writer = writer_for(run.out_dir)
    run.call_idx += 1
    run.inflight += 1
    call = LlmCallRecord(
        call_id=new_id("lc_"),
        run_id=run_id,
        call_idx=run.call_idx,
        request_ts=now_ms(),
        requested_output_cap=requested_cap,
        effective_output_cap=effective_cap,
    )
    _describe_request(call, payload, run)
    if SAVE_BLOBS:
        p = writer.blob_path(run_id, f"{call.call_idx:03d}_request.json")
        p.write_text(json.dumps(payload, indent=2))
        call.payload_path = str(p)

    started = monotonic_ms()
    if payload.get("stream"):
        return StreamingResponse(
            _stream(call, started, body, headers, run, writer),
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
        _finish(call, data, run, writer)


async def _stream(
    call: LlmCallRecord,
    started: float,
    body: bytes,
    headers: dict,
    run: RunState,
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
                if run.ended:
                    # A closed run's agent is gone or being stopped (a load
                    # test's filler, a timed-out Hermes). Dropping the upstream
                    # stream makes vLLM abort the request and free its batch slot.
                    call.error = "run_ended: stream cut"
                    run.refused_after_end += 1
                    break
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
        _finish(call, merged, run, writer)


def _describe_request(call: LlmCallRecord, payload: dict[str, Any], run: RunState) -> None:
    messages = payload.get("messages", [])
    call.n_messages = len(messages)
    call.prompt_chars = len(json.dumps(messages))
    call.forced_final_answer = any(
        m.get("role") == "user" and isinstance(m.get("content"), str)
        and m["content"] in run.forced_final_markers for m in messages
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
    call: LlmCallRecord, data: dict[str, Any], run: RunState, writer: TelemetryWriter
) -> None:
    try:
        if SAVE_BLOBS:
            p = writer.blob_path(run.run_id, f"{call.call_idx:03d}_response.json")
            p.write_text(json.dumps(data, indent=2))
            call.response_path = str(p)
        writer.write(call)
    finally:
        run.output_tokens += call.completion_tokens
        if call.error and not call.error.startswith("run_ended"):
            run.upstream_errors += 1
        if run.limits and not call.error:
            if call.usage_missing:
                run.termination_reason = "usage_missing"
            elif call.effective_output_cap and call.completion_tokens > call.effective_output_cap:
                run.termination_reason = "output_cap_violation"
        run.inflight = max(0, run.inflight - 1)
