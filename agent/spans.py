"""Tile a run's window into llm, tool and orchestration spans.

The proxy times every model request and the tool layer (or the agent's own
plugin) every tool call; whatever time is left inside the run is
orchestration. Rather than leave that as one residual number (T_Orch), each
gap becomes a span with a start and an end, so a power or CPU trace can be cut
at the same moments, and named:

  - by an adapter phase covering it, when the adapter knows what was happening
    (Hermes: exec_launch, profile_restore, hermes_boot, agent_setup,
    exec_return); otherwise
  - by position: startup before the first model or tool call, loop between
    calls, finish after the last.
"""

from __future__ import annotations

from typing import Any

from .telemetry import SpanRecord

Interval = tuple[float, float]


def merge(intervals: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def subtract(window: Interval, cuts: list[Interval]) -> list[Interval]:
    a, b = window
    out, cur = [], a
    for x, y in merge(cuts):
        x, y = max(x, a), min(y, b)
        if y <= x:
            continue
        if x > cur:
            out.append((cur, x))
        cur = max(cur, y)
    if cur < b:
        out.append((cur, b))
    return out


def build_spans(run_id: str, window: Interval, calls: list[dict[str, Any]],
                tools: list[dict[str, Any]], phases: list[tuple[str, float, float]]) -> list[SpanRecord]:
    """llm spans from llm_call rows, tool spans from tool_call rows (both as
    written), orch spans for the rest of the window, named as above."""
    raw: list[tuple[str, str, float, float, str]] = []
    for c in calls:
        a = c["request_ts"]
        raw.append(("llm", f"call_{c.get('call_idx', 0)}", a, a + (c.get("latency_ms") or 0.0), c.get("call_id", "")))
    for t in tools:
        if t.get("start_ts"):
            a = t["start_ts"]
            raw.append(("tool", t["tool_name"], a, a + t["latency_ms"], t.get("tool_call_id", "")))

    busy = merge([(a, b) for _, _, a, b, _ in raw if b > a])
    first = busy[0][0] if busy else window[1]
    last = busy[-1][1] if busy else window[0]
    phases = sorted(p for p in phases if p[2] > p[1])
    orch: list[tuple[str, float, float]] = []
    for a, b in subtract(window, busy):
        cuts = sorted({a, b, *(t for _, x, y in phases for t in (x, y) if a < t < b)})
        for x, y in zip(cuts, cuts[1:]):
            if y - x < 1.0:  # clock-granularity slivers between two timestamps
                continue
            mid = (x + y) / 2
            name = next((n for n, p, q in phases if p <= mid < q), None)
            if name is None:
                name = "startup" if y <= first else "finish" if x >= last else "loop"
            if orch and orch[-1][0] == name and orch[-1][2] == x:
                orch[-1] = (name, orch[-1][1], y)
            else:
                orch.append((name, x, y))
    raw += [("orch", n, a, b, "") for n, a, b in orch]
    raw.sort(key=lambda s: (s[2], s[3]))
    return [SpanRecord(run_id=run_id, span_idx=i, kind=k, name=n, start_ts=round(a, 1), end_ts=round(b, 1),
                       dur_ms=round(b - a, 1), ref_id=ref)
            for i, (k, n, a, b, ref) in enumerate(raw)]
