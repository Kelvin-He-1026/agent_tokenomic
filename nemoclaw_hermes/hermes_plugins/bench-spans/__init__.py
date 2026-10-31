"""Timestamped tool and API-request events for the agent_tokenomic harness.

Hermes's tools run inside the sandbox, out of the harness's sight; the proxy
sees only the model calls. This plugin appends one JSON line per event to the
file named by $HERMES_BENCH_SPANS, on the wall clock the harness and proxy use
(the sandbox shares the host's kernel clock):

    plugin_loaded                 Hermes has started and loaded its plugins
    api_start / api_end           Hermes's side of each model request
    tool_start / tool_end         each tool call; tool_end carries Hermes's own
                                  duration_ms, status and the result size

The harness (agent/hermes/oneshot_agent.py) reads the file back after the run.
Without the variable every hook returns at once, so the plugin is inert outside
benchmark runs.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_PATH = os.environ.get("HERMES_BENCH_SPANS", "")
_LOCK = threading.Lock()
_ARGS_MAX = 2000


def _emit(row: dict[str, Any]) -> None:
    if not _PATH:
        return
    row["t"] = round(time.time() * 1000, 1)
    line = json.dumps(row, separators=(",", ":"), default=str)
    with _LOCK, open(_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _args(args: Any) -> str:
    try:
        text = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(args)
    return text[:_ARGS_MAX]


def _pre_tool(*, tool_name: str = "", args: Any = None, tool_call_id: str = "", **_: Any) -> None:
    _emit({"ev": "tool_start", "tool": tool_name, "id": tool_call_id, "args": _args(args)})


def _post_tool(*, tool_name: str = "", tool_call_id: str = "", result: Any = None, duration_ms: int = 0,
               status: str | None = None, error_type: str | None = None, **_: Any) -> None:
    _emit({"ev": "tool_end", "tool": tool_name, "id": tool_call_id, "duration_ms": duration_ms,
           "status": status or "ok", "error_type": error_type or "",
           "result_bytes": len(str(result or "").encode())})


def _pre_api(**_: Any) -> None:
    _emit({"ev": "api_start"})


def _post_api(**_: Any) -> None:
    _emit({"ev": "api_end"})


def register(ctx) -> None:
    _emit({"ev": "plugin_loaded"})
    for name, fn in (("pre_tool_call", _pre_tool), ("post_tool_call", _post_tool),
                     ("pre_api_request", _pre_api), ("post_api_request", _post_api)):
        ctx.register_hook(name, fn)
