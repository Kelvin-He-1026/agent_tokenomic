"""Sample GPU and CPU power and utilisation into a time series, alongside a run.

The benchmark records when every LLM call and tool call happened; this records
what the hardware was doing at the same moments, on the same wall clock, so
scripts/attribute_energy.py can split each run's energy into LLM time (the GPU
generating), tool time (retrieval on the CPU, replayed web results) and
orchestration (framework code between the two).

    setsid nohup .venv/bin/python -u scripts/power_sampler.py --until-pid <pid> \\
        > /dev/null 2>&1 < /dev/null &
    # -> output/power/power_<stamp>.jsonl, until that process exits (or Ctrl-C)

Per sample (default every 100 ms):
  gpu    per GPU: power (W), cumulative energy (mJ, the driver's own counter,
         so energy between two moments is exact, not an integral of samples),
         SM and memory utilisation (%), VRAM used (MiB)
  cpu    busy fraction per NUMA node, from /proc/stat
  rapl   CPU package and DRAM energy counters (uJ) per socket, when readable.
         The counters are root-only by default; to enable until the next reboot:
             sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj
         Without that, CPU power is not measured (rapl is empty) and only CPU
         utilisation is recorded.

The sampler pins itself to one core on NUMA node 0, away from the GPU-attached
nodes the stacks use, and reads only kernel counters, so its own load is small.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.telemetry import file_stamp  # noqa: E402


def numa_nodes() -> dict[int, set[int]]:
    nodes: dict[int, set[int]] = {}
    for d in sorted(glob.glob("/sys/devices/system/node/node[0-9]*")):
        cpus: set[int] = set()
        for part in Path(d, "cpulist").read_text().strip().split(","):
            lo, _, hi = part.partition("-")
            cpus.update(range(int(lo), int(hi or lo) + 1))
        nodes[int(d.rsplit("node", 1)[1])] = cpus
    return nodes


def cpu_times() -> dict[int, tuple[int, int]]:
    """Per-CPU (busy, total) jiffies."""
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            vals = list(map(int, f[1:9]))
            idle = vals[3] + vals[4]  # idle + iowait
            out[int(f[0][3:])] = (sum(vals) - idle, sum(vals))
    return out


def rapl_domains() -> list[tuple[str, Path]]:
    """(name, energy file) for every readable RAPL domain, e.g. package-0, dram-0."""
    found = []
    for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
        f = Path(d, "energy_uj")
        name = Path(d, "name").read_text().strip()
        if name == "dram":  # name the DRAM domain after its socket
            name = f"dram-{d.split(':')[1]}"
        if os.access(f, os.R_OK):
            found.append((name, f))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--until-pid", type=int, default=0,
                    help="stop when this process exits (e.g. a sweep's driver)")
    ap.add_argument("--out", default="", help="default output/power/power_<stamp>.jsonl")
    args = ap.parse_args()

    try:  # one core on node 0, away from the GPU-attached nodes
        os.sched_setaffinity(0, {min(numa_nodes().get(0, {0}))})
    except OSError:
        pass

    import pynvml

    pynvml.nvmlInit()
    gpus = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    nodes = numa_nodes()
    rapl = rapl_domains()

    out = Path(args.out or ROOT / "output" / "power" / f"power_{file_stamp()}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def handle(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    with out.open("w") as fh:
        fh.write(json.dumps({
            "header": True, "host": socket.gethostname(), "interval_ms": args.interval_ms,
            "gpus": [pynvml.nvmlDeviceGetName(h) for h in gpus],
            "gpu_power_limit_w": [pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000 for h in gpus],
            "numa_nodes": {n: f"{min(c)}-{max(c)}" for n, c in nodes.items()},
            "rapl": [name for name, _ in rapl],
            "rapl_note": "" if rapl else "RAPL counters not readable; CPU power not measured",
            "fields": "t=wall ms; g=[[power_w, energy_mj, sm_util, mem_util, vram_mib] per GPU]; "
                      "c={numa node: busy fraction}; r={rapl domain: energy_uj}",
        }) + "\n")
        prev = cpu_times()
        period = args.interval_ms / 1000
        next_t = time.monotonic()
        n = 0
        while not stop:
            next_t += period
            g = []
            for h in gpus:
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                g.append([
                    round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000, 2),
                    pynvml.nvmlDeviceGetTotalEnergyConsumption(h),
                    util.gpu, util.memory,
                    round(pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20),
                ])
            cur = cpu_times()
            c = {}
            for node, cpus in nodes.items():
                busy = sum(cur[i][0] - prev[i][0] for i in cpus if i in cur and i in prev)
                total = sum(cur[i][1] - prev[i][1] for i in cpus if i in cur and i in prev)
                c[node] = round(busy / total, 3) if total else 0.0
            prev = cur
            r = {}
            for name, f in rapl:
                try:
                    r[name] = int(f.read_text())
                except OSError:
                    pass
            fh.write(json.dumps({"t": round(time.time() * 1000, 1), "g": g, "c": c, "r": r},
                                separators=(",", ":")) + "\n")
            n += 1
            if n == 1:
                fh.flush()  # lets a waiting harness see the trace has started
            if n % 50 == 0:
                fh.flush()
                if not rapl:  # e.g. someone ran the chmod after the sampler started
                    rapl = rapl_domains()
                if args.until_pid:
                    try:
                        os.kill(args.until_pid, 0)
                    except ProcessLookupError:
                        break
            time.sleep(max(0.0, next_t - time.monotonic()))
    pynvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
