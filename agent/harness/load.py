"""Closed-loop concurrency test: N virtual users running GAIA agent tasks.

    python -m agent.harness.load --stack h --config <cell.yaml> --tasks <tasks.json> \\
        --framework hermes --levels 1,4,8,16,32 --out output/runs/load_hermes

Each level N starts N virtual users (threads). A user takes the next task from
the level's queue as soon as its previous task has ended, so N tasks are always
in flight and a new request is only ever issued by a user whose last one
returned (closed loop). Every run is an ordinary benchmark run, with its own
budget and telemetry at the proxy (/r/<run_id>/v1), and Hermes users each get
their own profile slot in the sandbox.

The queue: the task mix repeated in whole passes, each pass in its own seeded
order, enough passes for max(--min-runs, --runs-per-user * N) runs. Every level
therefore runs the same mix, which is what makes its latency and success rate
comparable with the single-user baseline. These are the measured runs. Once
they have all started, users keep taking filler runs (the mix continued), so
the load stays at N until the last measured run ends; the fillers are then
closed at the proxy and recorded with measured=False, outside every figure.

Per level, from the measured runs: successful tasks per minute, P50/P95
end-to-end latency, task success rate, and the infrastructure error rate
(framework_error and timeout runs). From the run folder's power.jsonl over the
level's window: vLLM requests running and waiting, KV-cache usage, GPU
utilisation and energy per GPU, CPU per process group, CPU throttling and
pressure. One row per level goes to levels.jsonl and load_levels.csv.

Terms: unique_tasks is the task mix (distinct GAIA questions); measured_runs
are the runs counted in the quality figures; filler_runs are further runs of
the same tasks that keep the load at N. Which runs each figure uses:
  - agent quality (success rate, P50/P95 latency, infrastructure errors): the
    measured runs only, so every level answers the same mix as the baseline;
  - CPU/GPU capacity (vLLM queue, KV cache, GPU utilisation, CPU by group,
    throttling): the whole window, i.e. the real load of measured and filler
    runs together;
  - tasks/min and joules per success: every eligible task completed in the
    window, i.e. measured runs and the fillers that started and ended inside
    it on their own (not cancelled), over the window's time and its whole
    GPU and CPU energy. Fillers that finish inside the window lean toward
    the shorter tasks; for comparison, successful_tasks_per_min_little is
    the measured runs' steady rate by Little's law (N x successes / sum of
    their e2e), and gpu_/cpu_energy_measured_j their share of the window's
    energy (agent/energy_share.py: GPU energy split among the runs with an
    LLM call at vLLM, CPU energy among the runs in flight).
A filler the level closed while its agent was still working ends as status
"cancelled" (closed_by "level_end"); it is a benchmark shutdown, not a
failure, and is counted apart from fillers that failed on their own.
Infrastructure errors are split into inference errors (an LLM call failed at
or past vLLM: upstream_errors > 0) and the rest (the framework or harness).

The sweep stops at the first level that, against level 1 (the single-user
baseline on the same mix), has
  - P95 end-to-end latency more than --max-p95-ratio (2) times the baseline's,
  - task success more than --max-success-drop (5) points below the baseline's,
  - infrastructure errors and timeouts above --max-infra-pct (5) % of its runs.
The highest level that passed is the capacity.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
import statistics
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx

from ..adapters import FRAMEWORKS, get_adapter
from ..config import RunConfig
from ..energy_share import UNATTRIBUTED, apportion, counter_at, llm_windows
from ..env import load_env
from ..evaluation import evaluate
from ..telemetry import PlanRecord, TelemetryWriter, file_stamp, now_ms
from .cli import _release_proxy, _start_power_sampler, preflight
from .runner import Runner, Task

INFRA_STATUSES = ("framework_error", "timeout")


def queue_for_level(tasks: list[Task], n: int, min_runs: int, runs_per_user: int,
                    seed: int) -> tuple[list[tuple[Task, int]], int]:
    """(task, pass) for the measured runs, and how many passes they make."""
    passes = math.ceil(max(min_runs, runs_per_user * n) / len(tasks))
    return [(t, p) for p in range(passes) for t in pass_order(tasks, seed, p)], passes


def pass_order(tasks: list[Task], seed: int, p: int) -> list[Task]:
    order = list(tasks)
    random.Random(seed * 1000 + p).shuffle(order)
    return order


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


class Level:
    """One concurrency level: its users, its queue, its runs."""

    def __init__(self, n: int, tasks: list[Task], args, cfg: RunConfig, writer: TelemetryWriter,
                 session: str, config_runner: Runner):
        self.n, self.tasks, self.args, self.cfg = n, tasks, args, cfg
        self.writer, self.session = writer, session
        self.queue, self.passes = queue_for_level(tasks, n, args.min_runs, args.runs_per_user, args.seed)
        self.config_id = config_runner.config_id
        self.lock = threading.Lock()
        self.next_idx = 0
        self.measured_left = len(self.queue)
        self.measured_done = threading.Event()
        self.fillers: set[str] = set()
        self.results: list[dict] = []
        self.control = cfg.base_url.rsplit("/v1", 1)[0]

    def _item(self, i: int) -> tuple[Task, int, bool]:
        if i < len(self.queue):
            task, p = self.queue[i]
            return task, p, True
        j = i - len(self.queue)
        p = self.passes + j // len(self.tasks)
        return pass_order(self.tasks, self.args.seed, p)[j % len(self.tasks)], p, False

    def _close(self, run_id: str) -> None:
        try:
            httpx.post(f"{self.control}/control/end", json={"run_id": run_id, "reason": "level_end"},
                       timeout=5.0)
        except Exception:
            pass

    def user(self, k: int) -> None:
        fw = self.args.framework
        runner = Runner(self.cfg, get_adapter(fw, self.cfg, slot=f"s{k}"), self.args.out,
                        session=self.session, run_code=self.args.run_code, writer=self.writer,
                        adapter_factory=lambda c: get_adapter(fw, c, slot=f"s{k}"))
        runner._config_written = True  # written once, by the sweep
        while True:
            with self.lock:
                if self.measured_done.is_set():
                    return
                i = self.next_idx
                self.next_idx += 1
            task, p, measured = self._item(i)

            def on_start(run_id: str, measured: bool = measured) -> None:
                if measured:
                    return
                with self.lock:
                    self.fillers.add(run_id)
                    late = self.measured_done.is_set()
                if late:  # the level ended while this filler was being bound
                    self._close(run_id)

            started = now_ms()
            try:
                rec = runner.run_task(task, p, i, concurrent=True, concurrency=self.n, vu_idx=k,
                                      measured=measured, on_start=on_start)
                verdict = evaluate(task.raw, rec.final_answer, rec.status, [],
                                   completion_type=rec.completion_type)
                row = {"run_id": rec.run_id, "status": rec.status, "termination": rec.termination_reason,
                       "closed_by": rec.closed_by, "upstream_errors": rec.upstream_errors,
                       "success": verdict["success"] is True, "e2e_ms": rec.e2e_ms,
                       "start_ts": rec.start_ts, "end_ts": rec.end_ts, "llm_calls": rec.n_llm_calls,
                       "tokens": rec.input_tokens + rec.output_tokens, "t_llm_ms": rec.t_llm_ms,
                       "t_tool_ms": rec.t_tool_ms, "t_orch_ms": rec.t_orch_ms}
            except (Exception, SystemExit) as exc:
                # The harness itself failed (e.g. the proxy refused the bind).
                # Counted as an infrastructure error, so the level still ends.
                print(f"  c={self.n} vu{k}: harness error: {type(exc).__name__}: {exc}", flush=True)
                ended = now_ms()
                row = {"run_id": "", "status": "framework_error", "termination": "harness_error",
                       "closed_by": "", "upstream_errors": 0,
                       "success": False, "e2e_ms": ended - started, "start_ts": started, "end_ts": ended,
                       "llm_calls": 0, "tokens": 0, "t_llm_ms": 0.0, "t_tool_ms": 0.0, "t_orch_ms": 0.0}
                time.sleep(5)
            row.update({"task_id": task.task_id, "pass": p, "idx": i, "vu": k, "measured": measured})
            rec = SimpleNamespace(run_id=row["run_id"], status=row["status"], e2e_ms=row["e2e_ms"],
                                  n_llm_calls=row["llm_calls"])
            with self.lock:
                self.results.append(row)
                if not measured:
                    self.fillers.discard(rec.run_id)
                    continue
                self.measured_left -= 1
                done = self.measured_left == 0
                if done:
                    self.measured_done.set()
                    closing = list(self.fillers)
            print(f"  c={self.n:<3d} vu{k:<3d} {task.task_id[:13]:13s} p{p} {rec.status:15s} "
                  f"{'ok ' if row['success'] else '-- '} e2e={rec.e2e_ms / 1000:7.1f}s "
                  f"llm={rec.n_llm_calls:2d} ({self.measured_left} measured left)", flush=True)
            if done:
                for run_id in closing:
                    self._close(run_id)

    def run(self) -> tuple[float, float]:
        for i, (task, p) in enumerate(self.queue):
            self.writer.write(PlanRecord(self.session, self.config_id, task.task_id, p, i))
        started = now_ms()
        threads = [threading.Thread(target=self.user, args=(k,), name=f"vu{k}", daemon=True)
                   for k in range(self.n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        measured = [r for r in self.results if r["measured"]]
        return started, max((r["end_ts"] for r in measured), default=now_ms())


def summarize(level: Level, window: tuple[float, float], power: Path, calls: Path | None = None) -> dict:
    runs = [r for r in level.results if r["measured"]]
    fillers = [r for r in level.results if not r["measured"]]
    e2e = [r["e2e_ms"] / 1000 for r in runs]
    wins = sum(r["success"] for r in runs)
    infra = [r for r in runs if r["status"] in INFRA_STATUSES]
    minutes = (window[1] - window[0]) / 60000
    busy_min = sum(e2e) / 60  # run-minutes of the measured runs
    # Eligible completions: measured runs and fillers that ended on their own
    # inside the window (not closed by the level).
    done = [r for r in level.results if r["status"] != "cancelled"
            and window[0] <= r["start_ts"] and r["end_ts"] <= window[1]]
    done_wins = sum(r["success"] for r in done)
    row = {
        "concurrency": level.n, "session": level.session, "unique_tasks": len(level.tasks),
        "measured_runs": len(runs), "filler_runs": len(fillers), "passes": level.passes,
        "window_s": round(minutes * 60, 1), "start_ts": window[0], "end_ts": window[1],
        "successes": wins, "success_rate": round(wins / len(runs), 4) if runs else None,
        "completed_in_window": len(done), "successes_in_window": done_wins,
        "successful_tasks_per_min": round(done_wins / minutes, 3) if minutes else None,
        "tasks_per_min": round(len(done) / minutes, 3) if minutes else None,
        # Little's law for a closed loop with no think time: N / mean e2e.
        "successful_tasks_per_min_little": round(level.n * wins / busy_min, 3) if busy_min else None,
        "e2e_p50_s": _r(percentile(e2e, 50)), "e2e_p95_s": _r(percentile(e2e, 95)),
        "e2e_mean_s": _r(statistics.mean(e2e)) if e2e else None,
        "infra_errors": len(infra), "infra_error_rate": round(len(infra) / len(runs), 4) if runs else None,
        "inference_errors": sum(r.get("upstream_errors", 0) > 0 for r in infra),
        "framework_errors": sum(r["status"] == "framework_error" and not r.get("upstream_errors") for r in infra),
        "timeouts": sum(r["status"] == "timeout" for r in runs),
        "statuses": {s: sum(r["status"] == s for r in runs) for s in sorted({r["status"] for r in runs})},
        "fillers_cancelled": sum(r["status"] == "cancelled" for r in fillers),
        "fillers_finished": sum(r["status"] != "cancelled" for r in fillers),
        "filler_successes": sum(r["success"] for r in fillers),
        # Fillers that failed on their own, before the level closed them.
        "filler_infra_errors": sum(r["status"] in INFRA_STATUSES for r in fillers),
        "filler_inference_errors": sum(r.get("upstream_errors", 0) > 0 for r in fillers),
        "filler_statuses": {s: sum(r["status"] == s for r in fillers) for s in sorted({r["status"] for r in fillers})},
        "llm_calls_mean": _r(statistics.mean(r["llm_calls"] for r in runs)) if runs else None,
        "tokens_mean": _r(statistics.mean(r["tokens"] for r in runs)) if runs else None,
        "t_llm_mean_s": _r(statistics.mean(r["t_llm_ms"] for r in runs) / 1000) if runs else None,
        "t_tool_mean_s": _r(statistics.mean(r["t_tool_ms"] for r in runs) / 1000) if runs else None,
        "t_orch_mean_s": _r(statistics.mean(r["t_orch_ms"] for r in runs) / 1000) if runs else None,
    }
    fast, slow = read_trace(power, *window)
    row.update(hardware(fast, slow))
    call_rows = []
    if calls is not None and calls.exists():
        ids = {r["run_id"] for r in level.results if r["run_id"]}
        call_rows = [c for c in map(json.loads, calls.read_text().splitlines()) if c.get("run_id") in ids]
    row.update(measured_energy(fast, window, level.results, call_rows))
    if row.get("gpu_energy_window_j") is not None and done_wins:
        row["gpu_kj_per_success"] = _r(row["gpu_energy_window_j"] / done_wins / 1000)
        if row.get("cpu_energy_window_j") is not None:
            row["total_kj_per_success"] = _r((row["gpu_energy_window_j"] + row["cpu_energy_window_j"])
                                             / done_wins / 1000)
    if row.get("gpu_energy_measured_j") is not None and wins:
        row["gpu_kj_per_success_measured"] = _r(row["gpu_energy_measured_j"] / wins / 1000)
        if row.get("cpu_energy_measured_j") is not None:
            row["total_kj_per_success_measured"] = _r(
                (row["gpu_energy_measured_j"] + row["cpu_energy_measured_j"]) / wins / 1000)
    return row


def _r(x, nd: int = 3):
    return None if x is None else round(x, nd)


def read_trace(power: Path, a: float, b: float) -> tuple[list[dict], list[dict]]:
    """power_sampler.py's rows between a and b: every sample, and those that
    carry a slow sample (by its own timestamp)."""
    fast, slow = [], []
    if not power.exists():
        return fast, slow
    for line in power.read_text().splitlines():
        if not line.strip() or '"header"' in line[:12]:
            continue
        r = json.loads(line)
        t = r.get("ts", r["t"])
        if "ts" in r and a <= t <= b:
            slow.append(r)
        if a <= r["t"] <= b:
            fast.append(r)
    return fast, slow


def rapl_span() -> int:
    return max([int(Path(f).read_text()) for f in glob.glob(
        "/sys/class/powercap/intel-rapl:*/max_energy_range_uj")] or [2**32])


def counters(fast: list[dict]) -> dict:
    """Cumulative energy (J) per GPU and per CPU package, as functions of wall
    ms. RAPL counters wrap every few minutes at these powers: the steps are
    summed, adding the counter's range back at a wrap."""
    t = [r["t"] for r in fast]
    out = {}
    for i in range(len(fast[0]["g"]) if fast else 0):
        out[f"gpu{i}"] = counter_at(t, [r["g"][i][1] / 1000 for r in fast])
    span = rapl_span()
    for k in [k for k in (fast[0].get("r", {}) if fast else {}) if k.startswith("package")]:
        if not all(k in r.get("r", {}) for r in fast):
            continue
        vals, acc = [], 0
        for prev, cur in zip([fast[0]] + fast, fast):
            a, b = prev["r"][k], cur["r"][k]
            acc += b - a if b >= a else b - a + span
            vals.append(acc / 1e6)
        out[k] = counter_at(t, vals)
    return out


def measured_energy(fast: list[dict], window: tuple[float, float], results: list[dict],
                    calls: list[dict]) -> dict:
    """The window's energy, and the measured runs' share of it (see the
    module docstring)."""
    if len(fast) < 2:
        return {}
    ch = counters(fast)
    a, b = window
    gpus = [c for c in ch if c.startswith("gpu")]
    pkgs = [c for c in ch if c.startswith("package")]
    runs = {r["run_id"]: (max(r["start_ts"], a), min(r["end_ts"], b)) for r in results if r["run_id"]}
    share = apportion(ch, runs, llm_windows(calls), set(gpus))
    measured = {r["run_id"] for r in results if r["measured"] and r["run_id"]}
    total = {c: ch[c](b) - ch[c](a) for c in ch}

    def part(ids, chans):
        return sum(share[i][c] for i in ids for c in chans)

    out = {"gpu_energy_window_j": _r(sum(total[c] for c in gpus), 1),
           "gpu_energy_measured_j": _r(part(measured, gpus), 1),
           "gpu_energy_fillers_j": _r(part(set(runs) - measured, gpus), 1),
           "gpu_energy_unattributed_j": _r(part([UNATTRIBUTED], gpus), 1)}
    if pkgs:
        out.update({"cpu_energy_window_j": _r(sum(total[c] for c in pkgs), 1),
                    "cpu_energy_measured_j": _r(part(measured, pkgs), 1),
                    "cpu_energy_fillers_j": _r(part(set(runs) - measured, pkgs), 1)})
    if out["gpu_energy_window_j"]:
        out["measured_gpu_energy_share"] = _r(out["gpu_energy_measured_j"] / out["gpu_energy_window_j"])
    return out


def hardware(fast: list[dict], slow: list[dict]) -> dict:
    """What the hardware and vLLM did over these samples (read_trace)."""
    out: dict = {}
    if len(fast) >= 2:
        n_gpu = len(fast[0]["g"])
        secs = (fast[-1]["t"] - fast[0]["t"]) / 1000
        for i in range(n_gpu):
            out[f"gpu{i}_util_mean"] = _r(statistics.mean(r["g"][i][2] for r in fast), 1)
            out[f"gpu{i}_power_mean_w"] = _r(statistics.mean(r["g"][i][0] for r in fast), 1)
            out[f"gpu{i}_energy_j"] = _r((fast[-1]["g"][i][1] - fast[0]["g"][i][1]) / 1000, 1)
            if len(fast[0]["g"][i]) > 7:
                out[f"gpu{i}_sm_clock_mean_mhz"] = _r(statistics.mean(r["g"][i][5] or 0 for r in fast), 0)
                # Clock-event reasons other than idle (0x1) and application
                # clocks (0x4): power cap, thermal, power brake, sync boost.
                out[f"gpu{i}_throttled_frac"] = _r(sum(bool((r["g"][i][7] or 0) & ~0x5) for r in fast) / len(fast))
        out["host_cpu_util_mean"] = _r(statistics.mean(statistics.mean(r["c"].values()) for r in fast if r.get("c")), 3)
        cp = [r["cp"] for r in fast if "cp" in r]
        if cp:  # the sampler runs pinned (ATK_CPUS_PER_SOCKET): busy fraction of the pinned cores
            out["pinned_cpu_util_mean"] = _r(statistics.mean(cp), 3)
        out["window_trace_s"] = _r(secs, 1)
    if len(slow) >= 2:
        s0, s1 = slow[0], slow[-1]
        ms = s1["ts"] - s0["ts"]
        v = [r["v"] for r in slow if r.get("v")]
        if v:
            for key, name in (("run", "vllm_running"), ("wait", "vllm_waiting"), ("kv", "kv_cache_usage")):
                vals = [x.get(key, 0.0) for x in v]
                out[f"{name}_mean"] = _r(statistics.mean(vals))
                out[f"{name}_max"] = _r(max(vals))
            if len(v) >= 2:
                d = {k: v[-1].get(k, 0) - v[0].get(k, 0) for k in ("preempt", "pcq", "pch", "ptok", "gtok")}
                out["vllm_preemptions"] = int(d["preempt"])
                out["prefix_cache_hit_rate"] = _r(d["pch"] / d["pcq"]) if d["pcq"] else None
                out["prompt_tok_per_s"] = _r(d["ptok"] / (ms / 1000), 1)
                out["gen_tok_per_s"] = _r(d["gtok"] / (ms / 1000), 1)
        for group in sorted(set(s0.get("p", {})) | set(s1.get("p", {}))):
            used = s1.get("p", {}).get(group, [0, 0])[0] - s0.get("p", {}).get(group, [0, 0])[0]
            out[f"cpu_{group}_cores"] = _r(used / ms, 3)
        if s0.get("cg", {}).get("sandbox") and s1.get("cg", {}).get("sandbox"):
            c0, c1 = s0["cg"]["sandbox"], s1["cg"]["sandbox"]
            out["cpu_sandbox_cores"] = _r((c1[0] - c0[0]) / ms, 3)
            out["sandbox_cfs_throttled_periods"] = int(c1[1] - c0[1])
            out["sandbox_cfs_throttled_s"] = _r((c1[2] - c0[2]) / 1000, 2)
            out["sandbox_cpu_pressure_some_pct"] = _r(100 * (c1[3] - c0[3]) / ms, 2)
            out["sandbox_tasks_max"] = int(max(r["cg"]["sandbox"][6] for r in slow if r.get("cg", {}).get("sandbox")))
            out["sandbox_mem_max_mib"] = int(max(r["cg"]["sandbox"][5] for r in slow if r.get("cg", {}).get("sandbox")))
        if s0.get("h") and s1.get("h"):
            out["cpu_mhz_mean"] = _r(statistics.mean(r["h"][0] for r in slow if r.get("h")), 0)
            out["cpu_thermal_throttle_events"] = int((s1["h"][1] - s0["h"][1]) + (s1["h"][2] - s0["h"][2]))
            out["host_cpu_pressure_some_pct"] = _r(100 * (s1["h"][3] - s0["h"][3]) / ms, 2)
    return out


def verdict(row: dict, base: dict | None, args) -> list[str]:
    """Which stop rules this level breaks against the single-user baseline."""
    broken = []
    if (row.get("infra_error_rate") or 0) * 100 > args.max_infra_pct:
        broken.append(f"infra errors/timeouts {row['infra_error_rate']:.1%} > {args.max_infra_pct:g}%")
    if base is None or base is row:
        return broken
    if base.get("e2e_p95_s") and row.get("e2e_p95_s") and row["e2e_p95_s"] > args.max_p95_ratio * base["e2e_p95_s"]:
        broken.append(f"P95 e2e {row['e2e_p95_s']:.0f}s > {args.max_p95_ratio:g}x baseline "
                      f"{base['e2e_p95_s']:.0f}s")
    if base.get("success_rate") is not None and row.get("success_rate") is not None:
        drop = (base["success_rate"] - row["success_rate"]) * 100
        if drop > args.max_success_drop:
            broken.append(f"success {row['success_rate']:.1%} is {drop:.1f} points below baseline "
                          f"{base['success_rate']:.1%}")
    return broken


def wait_idle(control: str, vllm_metrics: str, limit_s: float = 900) -> None:
    """Until no run is in flight at the proxy and vLLM has nothing running."""
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline:
        try:
            busy = httpx.get(f"{control}/control/health", timeout=5).json().get("inflight", 0)
            text = httpx.get(vllm_metrics, timeout=5).text
            running = sum(float(l.rsplit(" ", 1)[1]) for l in text.splitlines()
                          if l.startswith("vllm:num_requests_running") or l.startswith("vllm:num_requests_waiting{"))
        except Exception:
            return
        if not busy and not running:
            return
        time.sleep(2)


COLUMNS = [
    "concurrency", "unique_tasks", "measured_runs", "filler_runs", "successes", "success_rate",
    "completed_in_window", "successes_in_window", "successful_tasks_per_min", "tasks_per_min",
    "e2e_p50_s", "e2e_p95_s", "e2e_mean_s", "infra_errors", "infra_error_rate", "timeouts",
    "vllm_running_mean", "vllm_running_max", "vllm_waiting_mean", "vllm_waiting_max",
    "kv_cache_usage_mean", "kv_cache_usage_max", "vllm_preemptions", "prefix_cache_hit_rate",
    "gen_tok_per_s", "prompt_tok_per_s", "gpu0_util_mean", "gpu1_util_mean",
    "gpu0_throttled_frac", "gpu1_throttled_frac", "gpu_energy_window_j", "cpu_energy_window_j",
    "gpu_kj_per_success", "total_kj_per_success", "successful_tasks_per_min_little",
    "gpu_energy_measured_j", "cpu_energy_measured_j", "gpu_kj_per_success_measured",
    "total_kj_per_success_measured", "gpu_energy_fillers_j", "cpu_energy_fillers_j",
    "measured_gpu_energy_share", "inference_errors", "framework_errors",
    "fillers_finished", "fillers_cancelled", "filler_successes", "filler_infra_errors",
    "filler_inference_errors", "cpuset", "pinned_cpu_util_mean", "host_cpu_util_mean", "cpu_vllm_cores",
    "cpu_proxy_cores", "cpu_harness_cores", "cpu_hermes_cli_cores", "cpu_openshell_gw_cores",
    "cpu_sandbox_cores", "cpu_sampler_cores", "sandbox_cfs_throttled_periods", "sandbox_cfs_throttled_s",
    "sandbox_cpu_pressure_some_pct", "sandbox_tasks_max", "sandbox_mem_max_mib", "cpu_mhz_mean",
    "cpu_thermal_throttle_events", "host_cpu_pressure_some_pct", "llm_calls_mean", "tokens_mean",
    "t_llm_mean_s", "t_tool_mean_s", "t_orch_mean_s", "window_s", "verdict",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Closed-loop concurrency test (virtual users)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--framework", required=True, choices=FRAMEWORKS)
    ap.add_argument("--levels", default="1,4,8,16,32",
                    help="virtual users per level, in order; the first is the baseline (use 1)")
    ap.add_argument("--out", required=True, help="run directory; a _YYYYmmdd_HHMMSS suffix is added")
    ap.add_argument("--exclude", default="",
                    help="comma-separated task_id prefixes to leave out of the mix")
    ap.add_argument("--min-runs", type=int, default=24, help="measured runs per level, at least")
    ap.add_argument("--runs-per-user", type=int, default=2,
                    help="measured runs per virtual user, at least (so P95 has samples at high N)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run-code", default="")
    ap.add_argument("--stack", default=None, choices=["a", "b", "h"])
    ap.add_argument("--tool-mode", choices=["replay", "record", "live"], default=None)
    ap.add_argument("--no-power", action="store_true")
    ap.add_argument("--cooldown-s", type=float, default=30.0,
                    help="idle time between levels, so their windows and the GPU state are separate")
    ap.add_argument("--baseline", default="",
                    help="levels.jsonl of an earlier sweep on the same mix, whose concurrency-1 "
                         "row is the baseline (to continue a sweep without rerunning level 1)")
    ap.add_argument("--max-p95-ratio", type=float, default=2.0)
    ap.add_argument("--max-success-drop", type=float, default=5.0, help="percentage points")
    ap.add_argument("--max-infra-pct", type=float, default=5.0)
    ap.add_argument("--no-stop", action="store_true", help="run every level, whatever the verdicts")
    args = ap.parse_args()
    args.out = f"{args.out.rstrip('/')}_{file_stamp()}"
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    load_env()
    cfg = RunConfig.from_yaml(args.config)
    if args.tool_mode:
        cfg = replace(cfg, tool_mode=args.tool_mode)
    from .. import stacks

    stack = stacks.get(args.stack)
    if stack.name:
        stacks.pin_current_process(stack)
        cfg = replace(cfg, base_url=stack.proxy_url, hardware_id=stack.hardware_id,
                      numa_pinning=stack.numa_pinning)
    cfg = replace(cfg, framework=args.framework)
    excluded = tuple(x.strip() for x in args.exclude.split(",") if x.strip())
    tasks = [t for t in Task.load_all(args.tasks) if not (excluded and t.task_id.startswith(excluded))]
    if not tasks:
        raise SystemExit("no tasks left after --exclude")
    print(f"{args.framework}: {len(tasks)} tasks in the mix, levels {levels} -> {args.out}")
    for n in levels:
        q, passes = queue_for_level(tasks, n, args.min_runs, args.runs_per_user, args.seed)
        print(f"  c={n}: {len(q)} measured runs ({passes} pass{'es' if passes > 1 else ''} over the mix)")

    preflight(cfg)
    writer = TelemetryWriter(args.out)
    base_session = f"sess_{uuid.uuid4().hex[:10]}"
    config_runner = Runner(cfg, get_adapter(args.framework, cfg), args.out, session=base_session,
                           run_code=args.run_code, writer=writer)
    config_runner.write_config()
    sampler = _start_power_sampler(SimpleNamespace(no_power=args.no_power, base_url="", resume="",
                                                   out=args.out), cfg, stack.vllm_url + "/metrics")
    control = cfg.base_url.rsplit("/v1", 1)[0]

    base = None
    if args.baseline:
        base = next((json.loads(l) for l in Path(args.baseline).read_text().splitlines()
                     if l.strip() and json.loads(l)["concurrency"] == 1), None)
        if base is None:
            raise SystemExit(f"{args.baseline} has no concurrency-1 row")
        print(f"baseline from {args.baseline}: P95 {base['e2e_p95_s']}s, success {base['success_rate']:.1%}")
    rows, capacity = [], None
    try:
        for li, n in enumerate(levels):
            if li:
                wait_idle(control, stack.vllm_url + "/metrics")
                time.sleep(args.cooldown_s)
            session = f"{base_session}-c{n}"
            level = Level(n, tasks, args, cfg, writer, session, config_runner)
            print(f"\n== concurrency {n}: {len(level.queue)} measured runs, session {session}", flush=True)
            window = level.run()
            _release_proxy(cfg, session)
            time.sleep(1.5)  # the sampler's next slow sample closes the window
            row = summarize(level, window, Path(args.out) / "power.jsonl", Path(args.out) / "llm_call.jsonl")
            row["cpuset"] = stack.numa_pinning
            if base is None and n == 1:
                base = row
            broken = verdict(row, base, args)
            row["verdict"] = "; ".join(broken) if broken else "pass"
            rows.append(row)
            with open(Path(args.out) / "levels.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"== c={n}: success {row['success_rate']:.1%} ({row['successes']}/{row['measured_runs']} "
                  f"measured, {row['filler_runs']} filler runs), {row['successful_tasks_per_min']} successful "
                  f"tasks/min ({row['successes_in_window']} of {row['completed_in_window']} completed in the "
                  f"window), {row.get('total_kj_per_success')} kJ/success, P50 {row['e2e_p50_s']}s "
                  f"P95 {row['e2e_p95_s']}s, infra {row['infra_error_rate']:.1%}, "
                  f"KV {row.get('kv_cache_usage_mean')}, waiting {row.get('vllm_waiting_mean')} -> "
                  f"{row['verdict']}", flush=True)
            if broken and base is not row:
                if not args.no_stop:
                    print(f"stopping: c={n} breaks the stop rule")
                    break
            elif not broken:
                capacity = n
    finally:
        if sampler is not None:
            time.sleep(0.5)
            sampler.terminate()
            sampler.wait(timeout=10)
        with open(Path(args.out) / "load_levels.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    print(f"\nmax concurrency within the limits: {capacity if capacity is not None else 'none'}"
          f"  (P95 <= {args.max_p95_ratio:g}x baseline, success within {args.max_success_drop:g} points, "
          f"infra errors <= {args.max_infra_pct:g}%)")
    print(f"wrote {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
