"""Split each run's energy and hardware use into LLM, tool and orchestration time.

Joins a power trace from scripts/power_sampler.py with the runs' own timings
(same wall clock): llm_call rows give the windows when a request was with vLLM
(the GPU generating), tool_call rows the windows when a tool ran (retrieval on
the CPU, replayed web results), and the rest of each run is orchestration
(framework code, prompt building, parsing).

    python scripts/attribute_energy.py --runs "output/runs/thinking_sweep_*/*" \\
        --out output/energy.csv [--stages-out output/stages.csv]
    # power files default to output/power/*.jsonl; pass --power to choose

Per run, for each phase (llm, tool, orch) and in total:
  gpu_*_j        GPU energy, both GPUs, from the driver's energy counters
  gpu_active_*_j the same minus each GPU's idle power over that time, i.e. the
                 energy the work added. Idle comes from --idle-w (measured with
                 the model loaded and nothing running); without it, the 5th
                 percentile of the trace, which is only right if the trace
                 includes idle periods
  cpu_*_j        CPU package energy, when RAPL was readable (else empty)
  gpu_util_*     mean SM utilisation (%), cpu_busy_* mean CPU busy fraction
  overlap_runs   other runs in flight at the same time. The figures above are
                 the whole machine over the run's window, so they are per-run
                 figures only where overlap_runs == 0; summed over
                 overlapping runs they count the same energy several times.
  gpu_share_j,   the run's share of the machine's energy (agent/energy_share.py:
  cpu_share_j    GPU energy split among the runs of its run folder with an LLM
                 call in flight, CPU energy among those in flight). Equal to
                 the totals for a run alone; for a load test's overlapping
                 runs, the figure to sum (energy_attribution "shared").

--stages-out: where the run folder has span.jsonl (harness 0.7+), one row per
run and stage, a stage being a span kind and name (llm, tool:<tool>,
orch:<name>, see agent/spans.py). Besides the columns above, per stage:
  gpu<i>_util     mean SM utilisation of each GPU (%)
  cpu_<group>_s   CPU-seconds each process group used in the stage (vllm,
                  proxy, harness, hermes_cli, openshell_gw, sampler, and
                  sandbox, the OpenShell container's cgroup), from the
                  trace's slow samples (power_sampler.py, every 500 ms), so a
                  stage shorter than that is apportioned by interpolation
A summary of the non-overlapping runs (each run alone on the machine) is
printed per framework: where the time, the GPU energy and the CPU went.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.energy_share import apportion, llm_windows  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class Trace:
    """A power trace with exact counter-based energy between any two moments."""

    def __init__(self, files: list[Path], idle_w: list[float] | None = None):
        rows: list[dict] = []
        self.header: dict = {}
        for f in files:
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("header"):
                    self.header = self.header or r
                else:
                    rows.append(r)
        rows.sort(key=lambda r: r["t"])
        self.t = [r["t"] for r in rows]
        self.rows = rows
        self.n_gpu = len(rows[0]["g"]) if rows else 0
        self.gpu_e = [[r["g"][i][1] / 1000.0 for r in rows] for i in range(self.n_gpu)]  # J
        self.idle_w = (idle_w * self.n_gpu if idle_w and len(idle_w) == 1 else idle_w) or \
            [self._pct([r["g"][i][0] for r in rows], 5) for i in range(self.n_gpu)]
        self.idle_from_trace = not idle_w
        self.rapl = sorted({k for r in rows for k in r.get("r", {})})
        self.cpu_e = {k: self._unwrap([r["r"].get(k) for r in rows], k) for k in self.rapl}
        # Slow samples (process groups, sandbox cgroup), on their own clock.
        slow = sorted((r for r in rows if "ts" in r and "p" in r), key=lambda r: r["ts"])
        self.slow_t = [r["ts"] for r in slow]
        self.groups = sorted({g for r in slow for g in r["p"]})
        self.group_ms = {g: self._carry([r["p"].get(g, [None])[0] for r in slow]) for g in self.groups}
        if any(r.get("cg", {}).get("sandbox") for r in slow):
            self.groups.append("sandbox")
            self.group_ms["sandbox"] = self._carry([(r.get("cg", {}).get("sandbox") or [None])[0] for r in slow])

    @staticmethod
    def _carry(values: list[float | None]) -> list[float]:
        """Fill gaps with the last known value (a group can appear mid-trace)."""
        out, last = [], 0.0
        for v in values:
            last = v if v is not None else last
            out.append(last)
        return out

    def group_cpu_s(self, a: float, b: float) -> dict[str, float]:
        """CPU-seconds per process group between a and b."""
        if len(self.slow_t) < 2:
            return {}
        def at(series, t):
            i = bisect.bisect_left(self.slow_t, t)
            if i <= 0:
                return series[0]
            if i >= len(self.slow_t):
                return series[-1]
            t0, t1 = self.slow_t[i - 1], self.slow_t[i]
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return series[i - 1] + w * (series[i] - series[i - 1])
        return {g: (at(v, b) - at(v, a)) / 1000.0 for g, v in self.group_ms.items()}

    @staticmethod
    def _pct(values: list[float], p: float) -> float:
        v = sorted(values)
        return v[int(len(v) * p / 100)] if v else 0.0

    @staticmethod
    def _unwrap(values: list[int | None], domain: str) -> list[float]:
        """RAPL counters wrap; rebuild a monotonic series in joules."""
        out, offset, last = [], 0.0, None
        span = 2**32  # fallback; real range read below when available
        for f in glob.glob("/sys/class/powercap/intel-rapl:*/max_energy_range_uj"):
            try:
                span = max(span, int(Path(f).read_text()))
            except OSError:
                pass
        for v in values:
            if v is None:  # not readable at that moment: unknown, not zero
                out.append(float("nan"))
                continue
            if last is not None and v < last:
                offset += span
            last = v
            out.append((v + offset) / 1e6)
        return out

    def covers(self, a: float, b: float) -> bool:
        return bool(self.t) and self.t[0] <= a and b <= self.t[-1]

    def _at(self, series: list[float], t: float) -> float:
        i = bisect.bisect_left(self.t, t)
        if i <= 0:
            return series[0]
        if i >= len(self.t):
            return series[-1]
        t0, t1 = self.t[i - 1], self.t[i]
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return series[i - 1] + w * (series[i] - series[i - 1])

    def gpu_energy(self, a: float, b: float) -> float:
        return sum(self.gpu_energy_each(a, b))

    def gpu_energy_each(self, a: float, b: float) -> list[float]:
        """Per GPU: the driver's cumulative energy counter at b minus at a."""
        return [self._at(s, b) - self._at(s, a) for s in self.gpu_e]

    def gpu_power_integral(self, a: float, b: float) -> list[float]:
        """Per GPU: trapezoid integral of the sampled power. Only a cross-check
        on the counter; the counter is what the CSV reports."""
        rows = self.samples(a, b)
        out = []
        for i in range(self.n_gpu):
            pts = [(r["t"], r["g"][i][0]) for r in rows]
            out.append(sum((t1 - t0) / 1000 * (p0 + p1) / 2 for (t0, p0), (t1, p1) in zip(pts, pts[1:])))
        return out

    def gpu_idle_energy(self, a: float, b: float) -> float:
        return sum(self.idle_w) * (b - a) / 1000.0

    def cpu_energy(self, a: float, b: float) -> float | None:
        pkgs = [k for k in self.rapl if k.startswith("package")]
        if not pkgs:
            return None
        total = sum(self._at(self.cpu_e[k], b) - self._at(self.cpu_e[k], a) for k in pkgs)
        return None if total != total else total  # NaN: RAPL was unreadable at an endpoint

    def samples(self, a: float, b: float) -> list[dict]:
        return self.rows[bisect.bisect_left(self.t, a):bisect.bisect_right(self.t, b)]


def merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def subtract(window: tuple[float, float], cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
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


def clip(intervals, window):
    a, b = window
    return [(max(x, a), min(y, b)) for x, y in intervals if min(y, b) > max(x, a)]


def phase_stats(trace: Trace, intervals: list[tuple[float, float]]) -> dict:
    gpu = sum(trace.gpu_energy(a, b) for a, b in intervals)
    idle = sum(trace.gpu_idle_energy(a, b) for a, b in intervals)
    cpu_parts = [trace.cpu_energy(a, b) for a, b in intervals]
    samples = [s for a, b in intervals for s in trace.samples(a, b)]
    util = [statistics.mean(g[2] for g in s["g"]) for s in samples]
    busy = [statistics.mean(s["c"].values()) for s in samples if s.get("c")]
    return {
        "s": sum(b - a for a, b in intervals) / 1000.0,
        "gpu_j": gpu,
        "gpu_active_j": max(0.0, gpu - idle),
        "cpu_j": None if not intervals or any(c is None for c in cpu_parts) else sum(cpu_parts),
        "gpu_util": statistics.mean(util) if util else None,
        "cpu_busy": statistics.mean(busy) if busy else None,
    }


def stage_stats(trace: Trace, intervals: list[tuple[float, float]]) -> dict:
    """phase_stats, plus per-GPU utilisation and CPU-seconds per process group."""
    out = phase_stats(trace, intervals)
    samples = [s for a, b in intervals for s in trace.samples(a, b)]
    for i in range(trace.n_gpu):
        out[f"gpu{i}_util"] = statistics.mean(s["g"][i][2] for s in samples) if samples else None
    cpu: dict[str, float] = {}
    for a, b in intervals:
        for g, sec in trace.group_cpu_s(a, b).items():
            cpu[g] = cpu.get(g, 0.0) + sec
    for g in trace.groups:
        out[f"cpu_{g}_s"] = cpu.get(g)
    return out


def write_stages(path: Path, trace: Trace, runs: list, windows: list) -> None:
    """One row per run and stage (span kind and name), and a printed summary."""
    rows = []
    for d, r, cfg, calls, tools in runs:
        spans = [s for s in read_jsonl(d / "span.jsonl") if s["run_id"] == r["run_id"]] \
            if (d / "span.jsonl").exists() else []
        win = (r["start_ts"], r["end_ts"])
        if not spans or not trace.covers(*win):
            continue
        overlap = sum(1 for a, b in windows if a < win[1] and b > win[0]) - 1
        stages: dict[tuple[str, str], list] = {}
        for sp in spans:
            name = sp["kind"] if sp["kind"] == "llm" else f"{sp['kind']}:{sp['name']}"
            stages.setdefault((sp["kind"], name), []).append((sp["start_ts"], sp["end_ts"]))
        # Tool spans can overlap each other (parallel tool calls) and, in
        # Hermes, the model calls around them; energy is cut on merged time.
        for (kind, name), iv in stages.items():
            st = stage_stats(trace, clip(merge(iv), win))
            row = {"run_id": r["run_id"], "run_dir": d.name, "task_id": r["task_id"],
                   "framework": cfg.get("framework", ""), "concurrency": r.get("concurrency", 1),
                   "overlap_runs": overlap, "status": r["status"], "stage_kind": kind, "stage": name,
                   "n_spans": len(iv)}
            row.update({k: "" if v is None else round(v, 3) for k, v in st.items()})
            rows.append(row)
    if not rows:
        print("stages: no run folder has span.jsonl inside the trace")
        return
    cols = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} run-stage rows)")

    solo = [x for x in rows if x["overlap_runs"] == 0]
    groups = [c[4:-2] for c in cols if c.startswith("cpu_") and c.endswith("_s")]
    for fw in sorted({x["framework"] for x in solo}):
        mine = [x for x in solo if x["framework"] == fw]
        n_runs = len({x["run_id"] for x in mine})
        tot_s = sum(x["s"] for x in mine) or 1
        tot_gpu = sum(x["gpu_active_j"] or 0 for x in mine) or 1
        print(f"\n{fw}: {n_runs} runs alone on the machine; per run, by stage")
        print(f"  {'stage':26s} {'sec':>7s} {'time%':>6s} {'GPU kJ':>7s} {'GPU+%':>6s} {'CPU kJ':>7s} "
              f"{'GPU util':>8s}  CPU cores by group")
        by_stage: dict[str, list] = {}
        for x in mine:
            by_stage.setdefault(x["stage"], []).append(x)
        for stage, xs in sorted(by_stage.items(), key=lambda kv: -sum(x["s"] for x in kv[1])):
            sec = sum(x["s"] for x in xs)
            cores = {g: sum(x.get(f"cpu_{g}_s") or 0 for x in xs) / sec for g in groups if sec}
            util = [x["gpu_util"] for x in xs if x["gpu_util"] != ""]
            cpu_kj = sum(x["cpu_j"] or 0 for x in xs) / 1000 / n_runs
            print(f"  {stage:26s} {sec / n_runs:7.1f} {100 * sec / tot_s:5.1f}% "
                  f"{sum(x['gpu_j'] for x in xs) / 1000 / n_runs:7.2f} "
                  f"{100 * sum(x['gpu_active_j'] or 0 for x in xs) / tot_gpu:5.1f}% {cpu_kj:7.2f} "
                  f"{statistics.mean(util) if util else 0:7.0f}%  "
                  + " ".join(f"{g}={c:.2f}" for g, c in cores.items() if c >= 0.005))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (globs allowed)")
    ap.add_argument("--power", nargs="*", default=["output/power/*.jsonl"])
    ap.add_argument("--out", default="output/energy.csv")
    ap.add_argument("--stages-out", default="",
                    help="also write per-run, per-stage figures from span.jsonl (see above)")
    ap.add_argument("--idle-w", default="",
                    help="idle power per GPU in W, one value or comma-separated per GPU "
                         "(measure with the model loaded and no requests: nvidia-smi)")
    args = ap.parse_args()

    idle = [float(x) for x in args.idle_w.split(",") if x.strip()] or None
    run_dirs = [Path(p) for pat in args.runs for p in sorted(glob.glob(pat))]
    # The harness writes <run folder>/power.jsonl; a hand-started sampler, output/power/.
    files = [Path(p) for pat in args.power for p in sorted(glob.glob(pat))]
    files += [d / "power.jsonl" for d in run_dirs if (d / "power.jsonl").exists()]
    trace = Trace(sorted(set(files)), idle)
    if not trace.t:
        raise SystemExit("no power samples found; start scripts/power_sampler.py before the runs")
    dirs = [Path(p) for pat in args.runs for p in sorted(glob.glob(pat))
            if (Path(p) / "run.jsonl").exists()]

    runs = []
    for d in dirs:
        configs = {c["config_id"]: c for c in read_jsonl(d / "config.jsonl")}
        calls, tools = {}, {}
        for c in read_jsonl(d / "llm_call.jsonl"):
            calls.setdefault(c["run_id"], []).append(c)
        for t in read_jsonl(d / "tool_call.jsonl"):
            tools.setdefault(t["run_id"], []).append(t)
        for r in read_jsonl(d / "run.jsonl"):
            runs.append((d, r, configs.get(r["config_id"], {}), calls.get(r["run_id"], []),
                         tools.get(r["run_id"], [])))

    windows = sorted((r["start_ts"], r["end_ts"]) for _, r, *_ in runs)
    counters = {f"gpu{i}": (lambda x, v=v: trace._at(v, x)) for i, v in enumerate(trace.gpu_e)}
    counters.update({k: (lambda x, v=trace.cpu_e[k]: trace._at(v, x)) for k in trace.rapl if k.startswith("package")})
    shared: dict[str, dict[str, float]] = {}
    for d in dirs:
        mine = {r["run_id"]: (r["start_ts"], r["end_ts"]) for dd, r, *_ in runs if dd == d}
        calls_d = [c for dd, _, _, cs, _ in runs if dd == d for c in cs]
        shared.update(apportion(counters, mine, llm_windows(calls_d), {c for c in counters if c.startswith("gpu")}))
    rows, skipped = [], 0
    for d, r, cfg, calls, tools in runs:
        win = (r["start_ts"], r["end_ts"])
        if not trace.covers(*win):
            skipped += 1
            continue
        llm = clip(merge([(c["request_ts"], c["request_ts"] + (c.get("latency_ms") or 0)) for c in calls]), win)
        tool = clip(merge([(t["start_ts"], t["start_ts"] + t["latency_ms"])
                           for t in tools if t.get("start_ts")]), win)
        tool = [seg for x in tool for seg in subtract(x, llm)]
        orch = subtract(win, llm + tool)
        overlap = sum(1 for a, b in windows if a < win[1] and b > win[0]) - 1
        row = {
            "run_id": r["run_id"], "run_dir": d.name, "task_id": r["task_id"],
            "framework": cfg.get("framework", ""), "model": cfg.get("model_repo", ""),
            "thinking": cfg.get("thinking_setting", ""), "effort": cfg.get("reasoning_effort", ""),
            "tp_size": cfg.get("tp_size", ""), "status": r["status"], "overlap_runs": overlap,
            "concurrency": r.get("concurrency", 1), "measured": r.get("measured", True),
            "tool_times_recorded": all(t.get("start_ts") for t in tools),
        }
        sh = shared.get(r["run_id"], {})
        row["energy_attribution"] = "exclusive" if overlap == 0 else "shared"
        row["gpu_share_j"] = round(sum(v for k, v in sh.items() if k.startswith("gpu")), 3)
        cpu_sh = [v for k, v in sh.items() if k.startswith("package")]
        row["cpu_share_j"] = round(sum(cpu_sh), 3) if cpu_sh and trace.cpu_energy(*win) is not None else ""
        for name, iv in (("total", [win]), ("llm", llm), ("tool", tool), ("orch", orch)):
            st = phase_stats(trace, iv)
            for k, v in st.items():
                row[f"{k}_{name}" if k == "s" else f"{k.replace('_j', '')}_{name}_j" if k.endswith("_j")
                    else f"{k}_{name}"] = "" if v is None else round(v, 3)
        rows.append(row)

    if not rows:
        raise SystemExit(f"no run fell inside the power trace ({skipped} outside it)")
    if args.stages_out:
        write_stages(Path(args.stages_out), trace, runs, windows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out}  ({len(rows)} runs; {skipped} outside the power trace)")
    if trace.idle_from_trace and max(trace.idle_w) > 150:
        print(f"warning: the trace's lowest power is {max(trace.idle_w):.0f} W, so it probably has no "
              "idle period; pass --idle-w for gpu_active_* figures")
    print(f"GPU idle baseline: {', '.join(f'{x:.0f} W' for x in trace.idle_w)}; "
          f"CPU power: {'measured (' + ', '.join(trace.rapl) + ')' if trace.rapl else 'not measured (RAPL not readable)'}")
    solo = [x for x in rows if x["overlap_runs"] == 0]
    if solo:
        def share(key):
            tot = sum(x["gpu_active_total_j"] or 0 for x in solo)
            return 100 * sum(x[key] or 0 for x in solo) / tot if tot else 0.0
        print(f"{len(solo)} non-overlapping runs: GPU energy above idle by phase -> "
              f"llm {share('gpu_active_llm_j'):.0f}%, tool {share('gpu_active_tool_j'):.0f}%, "
              f"orch {share('gpu_active_orch_j'):.0f}%")
    else:
        print("every run overlapped another (parallel recording): totals only, no per-run split")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
