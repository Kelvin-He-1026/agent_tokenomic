"""Per-run shares of energy measured for the whole machine, when runs overlap.

The GPU and CPU energy counters measure the machine, not a run. A run alone on
the machine owns all of it; runs in flight together (a load test, parallel
recording) share it. apportion() splits it at every moment among the runs in
flight then:

  GPU channels   among the runs with an LLM call at vLLM at that moment (the
                 requests the GPU is batching); among all runs in flight when
                 none has one (the GPU idling while they run tools)
  CPU channels   among all runs in flight

so per-run shares add up to the energy of the time any run was in flight.
Energy while no run was in flight (between levels, before the first run) is
returned under UNATTRIBUTED. An equal split between batched requests is a
convention, not a measurement: a request in prefill costs more than one in
decode. Sums over many runs (a level, a task, a framework) are what to compare.
"""

from __future__ import annotations

import bisect
import math
from typing import Callable

UNATTRIBUTED = "_unattributed"


def counter_at(times: list[float], values: list[float]) -> Callable[[float], float]:
    """Linear interpolation of a cumulative counter sampled at times."""
    def at(t: float) -> float:
        i = bisect.bisect_left(times, t)
        if i <= 0:
            return values[0]
        if i >= len(times):
            return values[-1]
        t0, t1 = times[i - 1], times[i]
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return values[i - 1] + w * (values[i] - values[i - 1])
    return at


def apportion(counters: dict[str, Callable[[float], float]], runs: dict[str, tuple[float, float]],
              busy: dict[str, list[tuple[float, float]]], busy_channels: set[str]) -> dict[str, dict[str, float]]:
    """{run_id: {channel: energy}} from cumulative counters (callables of wall
    ms), run windows, and each run's LLM-call windows. busy_channels are split
    among runs with a call in flight (the GPUs); the rest among all runs."""
    events: list[tuple[float, int, str, str]] = []  # (t, order, kind, run_id): ends sort first
    for rid, (a, b) in runs.items():
        if b > a:
            events += [(a, 1, "run+", rid), (b, 0, "run-", rid)]
    for rid, calls in busy.items():
        if rid not in runs:
            continue
        a_run, b_run = runs[rid]
        for a, b in calls:
            a, b = max(a, a_run), min(b, b_run)
            if b > a:
                events += [(a, 1, "llm+", rid), (b, 0, "llm-", rid)]
    events.sort()
    out: dict[str, dict[str, float]] = {rid: dict.fromkeys(counters, 0.0) for rid in runs}
    out[UNATTRIBUTED] = dict.fromkeys(counters, 0.0)
    if not events:
        return out
    active: set[str] = set()
    llm: dict[str, int] = {}
    prev_t = events[0][0]
    prev_val = {ch: f(prev_t) for ch, f in counters.items()}
    for t, _, kind, rid in events:
        if t > prev_t:
            val = {ch: f(t) for ch, f in counters.items()}
            calling = [r for r, n in llm.items() if n > 0]
            for ch in counters:
                e = val[ch] - prev_val[ch]
                if math.isnan(e):
                    continue
                share = calling if ch in busy_channels and calling else list(active)
                if not share:
                    out[UNATTRIBUTED][ch] += e
                    continue
                for r in share:
                    out[r][ch] += e / len(share)
            prev_t, prev_val = t, val
        if kind == "run+":
            active.add(rid)
        elif kind == "run-":
            active.discard(rid)
        elif kind == "llm+":
            llm[rid] = llm.get(rid, 0) + 1
        else:
            llm[rid] = llm.get(rid, 0) - 1
    return out


def llm_windows(calls: list[dict]) -> dict[str, list[tuple[float, float]]]:
    """{run_id: [(request_ts, request_ts + latency_ms)]} from llm_call rows."""
    out: dict[str, list[tuple[float, float]]] = {}
    for c in calls:
        a = c.get("request_ts")
        if a:
            out.setdefault(c["run_id"], []).append((a, a + (c.get("latency_ms") or 0.0)))
    return out
