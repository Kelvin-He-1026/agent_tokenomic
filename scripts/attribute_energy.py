"""Split each run's energy and hardware use into LLM, tool and orchestration time.

Joins a power trace from scripts/power_sampler.py with the runs' own timings
(same wall clock): llm_call rows give the windows when a request was with vLLM
(the GPU generating), tool_call rows the windows when a tool ran (retrieval on
the CPU, replayed web results), and the rest of each run is orchestration
(framework code, prompt building, parsing).

    python scripts/attribute_energy.py --runs "output/runs/thinking_sweep_*/*" \\
        --out output/energy.csv
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
  overlap_runs   other runs in flight at the same time. Parallel recording
                 overlaps runs on purpose; their energy cannot be split per
                 run, so only rows with overlap_runs == 0 are per-run figures.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import statistics
from pathlib import Path


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
            if v is None:
                out.append(out[-1] if out else 0.0)
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
        return sum(self._at(s, b) - self._at(s, a) for s in self.gpu_e)

    def gpu_idle_energy(self, a: float, b: float) -> float:
        return sum(self.idle_w) * (b - a) / 1000.0

    def cpu_energy(self, a: float, b: float) -> float | None:
        pkgs = [k for k in self.rapl if k.startswith("package")]
        if not pkgs:
            return None
        return sum(self._at(self.cpu_e[k], b) - self._at(self.cpu_e[k], a) for k in pkgs)

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (globs allowed)")
    ap.add_argument("--power", nargs="*", default=["output/power/*.jsonl"])
    ap.add_argument("--out", default="output/energy.csv")
    ap.add_argument("--idle-w", default="",
                    help="idle power per GPU in W, one value or comma-separated per GPU "
                         "(measure with the model loaded and no requests: nvidia-smi)")
    args = ap.parse_args()

    idle = [float(x) for x in args.idle_w.split(",") if x.strip()] or None
    trace = Trace([Path(p) for pat in args.power for p in sorted(glob.glob(pat))], idle)
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
            "tool_times_recorded": all(t.get("start_ts") for t in tools),
        }
        for name, iv in (("total", [win]), ("llm", llm), ("tool", tool), ("orch", orch)):
            st = phase_stats(trace, iv)
            for k, v in st.items():
                row[f"{k}_{name}" if k == "s" else f"{k.replace('_j', '')}_{name}_j" if k.endswith("_j")
                    else f"{k}_{name}"] = "" if v is None else round(v, 3)
        rows.append(row)

    if not rows:
        raise SystemExit(f"no run fell inside the power trace ({skipped} outside it)")
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
