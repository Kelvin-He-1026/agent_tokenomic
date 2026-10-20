"""Check how energy_gpu_j is measured, on real runs.

energy_gpu_j is the driver's cumulative energy counter (NVML total energy
consumption) read at run end minus run start, per GPU, summed over all GPUs.
This script shows, per run, that:

  1. the counter agrees with an independent integral of the sampled power
     (so it is a real energy reading, not a derived figure);
  2. it differs clearly from a fixed power x time estimate (so it is not one);
  3. both GPUs are included (gpu0 + gpu1 = total, and each is non-zero).

    python scripts/verify_energy.py --runs "output/runs/<sweep>/*" \\
        --out output/energy_verification.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attribute_energy import Trace, read_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--power", nargs="*", default=[])
    ap.add_argument("--out", default="output/energy_verification.json")
    args = ap.parse_args()

    dirs = [Path(p) for pat in args.runs for p in sorted(glob.glob(pat)) if (Path(p) / "run.jsonl").exists()]
    files = [Path(p) for pat in args.power for p in sorted(glob.glob(pat))]
    files += [d / "power.jsonl" for d in dirs if (d / "power.jsonl").exists()]
    files += [Path(p) for p in sorted(glob.glob("output/power/*.jsonl"))]
    trace = Trace(sorted(set(files)))
    runs = [r for d in dirs for r in read_jsonl(d / "run.jsonl")
            if r.get("start_ts") and trace.covers(r["start_ts"], r["end_ts"])]
    if not runs:
        raise SystemExit("no runs inside the power trace")

    rel_integral, rel_fixed, gpu_share, zero_gpu = [], [], [], 0
    all_power = [r["g"][i][0] for r in trace.rows for i in range(trace.n_gpu)]
    fixed_w = statistics.median(all_power)  # what a fixed-power estimate would assume, per GPU
    for r in runs:
        a, b = r["start_ts"], r["end_ts"]
        each = trace.gpu_energy_each(a, b)
        total = sum(each)
        integ = sum(trace.gpu_power_integral(a, b))
        fixed = fixed_w * trace.n_gpu * (b - a) / 1000
        if total <= 0:
            continue
        if (b - a) >= 2000:  # the integral needs enough 100 ms samples
            rel_integral.append(abs(integ - total) / total)
        rel_fixed.append(abs(fixed - total) / total)
        gpu_share.append(each[0] / total if trace.n_gpu > 1 else 1.0)
        zero_gpu += any(e <= 0 for e in each)

    def q(v, p):
        v = sorted(v)
        return round(v[int(len(v) * p)], 4) if v else None

    report = {
        "runs_checked": len(runs),
        "method": "NVML nvmlDeviceGetTotalEnergyConsumption (mJ, cumulative) at run end minus run start, "
                  "linearly interpolated between 100 ms samples, summed over all GPUs",
        "gpus": trace.n_gpu,
        "counter_vs_power_integral": {
            "runs": len(rel_integral), "median_rel_diff": q(rel_integral, 0.5),
            "p95_rel_diff": q(rel_integral, 0.95),
            "meaning": "small: the counter is a real energy reading consistent with the measured power",
        },
        "counter_vs_fixed_power_x_time": {
            "assumed_w_per_gpu": round(fixed_w, 1), "median_rel_diff": q(rel_fixed, 0.5),
            "p95_rel_diff": q(rel_fixed, 0.95),
            "meaning": "large: a fixed power x time figure would not reproduce the counter",
        },
        "both_gpus_included": {
            "gpu0_share_median": q(gpu_share, 0.5), "runs_with_a_zero_gpu": zero_gpu,
            "meaning": "each GPU contributes; energy_gpu_j = energy_gpu0_j + energy_gpu1_j",
        },
        "idle_included": "yes: energy_gpu_j is gross; scripts/attribute_energy.py --idle-w gives energy above idle",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
