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
         SM and memory utilisation (%), VRAM used (MiB), SM clock (MHz),
         temperature (C), and the driver's clock-event (throttle) reasons
  cpu    busy fraction per NUMA node, from /proc/stat; and, when the sampler
         is started on some of the CPUs (it inherits the harness's affinity,
         ATK_CPUS_PER_SOCKET), the busy fraction of those CPUs (cp). The
         sampler itself then moves to one core of node 0, off the GPU nodes
  rapl   CPU package and DRAM energy counters (uJ) per socket, when readable.
         The counters are root-only by default; to enable until the next reboot:
             sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj
         Without that, CPU power is not measured (rapl is empty) and only CPU
         utilisation is recorded.

Every --slow-ms (default 500 ms), from a side thread so the power cadence holds:
  p      CPU time (ms, cumulative) per process group, from /proc:
           vllm          the vLLM server and its engine and worker processes
           proxy         the logging proxy
           harness       the harness (--until-pid): LangGraph and the other
                         in-process frameworks, and the load test's threads
           hermes_cli    nemohermes/openshell commands the harness starts
                         (the Hermes adapter's exec into the sandbox)
           openshell_gw  the OpenShell gateway and port forwards on the host
           sampler       this sampler (a child of the harness, kept apart)
         Each process's own utime+stime, summed over every process ever seen
         in the group, so a process that exits keeps what it used up to its
         last sample (up to --slow-ms of its final CPU time is missed).
  cg     cgroup counters of the OpenShell sandbox container (Hermes and its
         tools): cumulative CPU usage, CFS throttling (periods throttled,
         time throttled), CPU pressure stall time (some, full), memory and
         task count. Exact, including short-lived processes.
  h      host: mean CPU clock (MHz), thermal throttle event counts (core,
         package), and host CPU pressure stall time
  v      vLLM's /metrics: requests running and waiting, KV-cache usage,
         preemptions, prefix-cache queries and hits, prompt and generation
         tokens (counters cumulative)

The sampler pins itself to one core on NUMA node 0, away from the GPU-attached
nodes the stacks use, and reads only kernel counters, so its own load is small.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import signal
import socket
import sys
import threading
import time
import urllib.request
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


def _cpulist(cpus: list[int]) -> str:
    """[0, 1, 2, 3, 8] -> '0-3,8'"""
    runs: list[list[int]] = []
    for c in cpus:
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(f"{a}-{b}" if b > a else str(a) for a, b in runs)


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
        # Listed even when unreadable: reads fail (and are skipped) until
        # someone makes them readable, which then takes effect mid-trace.
        if f.exists():
            found.append((name, f))
    return found


CLK_TCK = os.sysconf("SC_CLK_TCK")
# Process groups by what a process (or its nearest classified ancestor) runs.
_GROUP_MARKERS = (("sampler", ("power_sampler.py",)),
                  ("vllm", ("vllm serve", "vllm.entrypoints")),
                  ("proxy", ("agent.proxy.server",)),
                  ("openshell_gw", ("openshell-gateway", "/openshell ")))
_HERMES_CLI = ("nemohermes", "openshell")


class ProcGroups:
    """Cumulative CPU time per process group, from /proc (see the module doc)."""

    def __init__(self, harness_pid: int):
        self.harness_pid = harness_pid
        self.own: dict[tuple[int, int], str | None] = {}  # (pid, starttime) -> own group
        self.used: dict[tuple[int, int], tuple[str, int]] = {}  # -> (group, ticks)

    @staticmethod
    def _stat(pid: str) -> tuple[int, int, int, int] | None:
        """(ppid, utime+stime ticks, starttime, in a container) for one pid."""
        try:
            text = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return None
        f = text.rsplit(")", 1)[1].split()
        return int(f[1]), int(f[11]) + int(f[12]), int(f[19]), 0

    def _own_group(self, pid: int, key: tuple[int, int], under_harness: bool) -> str | None:
        if key in self.own:
            return self.own[key]
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            cgroup = Path(f"/proc/{pid}/cgroup").read_text()
        except OSError:
            return None
        group = None
        if "/docker-" in cgroup:
            group = "container"  # the sandbox: counted from its cgroup instead
        elif pid == self.harness_pid:
            group = "harness"
        elif under_harness and any(m in cmd for m in _HERMES_CLI):
            group = "hermes_cli"
        else:
            group = next((g for g, marks in _GROUP_MARKERS if any(m in cmd for m in marks)), None)
        self.own[key] = group
        return group

    def sample(self) -> dict[str, list[float]]:
        procs = {}
        for d in os.listdir("/proc"):
            if d.isdigit():
                st = self._stat(d)
                if st:
                    procs[int(d)] = st
        resolved: dict[int, str | None] = {}

        def group_of(pid: int, depth: int = 0) -> str | None:
            if pid in resolved:
                return resolved[pid]
            st = procs.get(pid)
            if st is None or depth > 64:
                return None
            parent = group_of(st[0], depth + 1) if st[0] > 1 else None
            own = self._own_group(pid, (pid, st[2]), parent in ("harness", "hermes_cli"))
            # A Hermes command's own children stay hermes_cli; a classified
            # process's unclassified children inherit its group.
            g = "hermes_cli" if parent == "hermes_cli" and own != "container" else own or parent
            resolved[pid] = g
            return g

        for pid, (_, ticks, start, _) in procs.items():
            g = group_of(pid)
            if g and g != "container":
                self.used[(pid, start)] = (g, ticks)
        out: dict[str, list[float]] = {}
        alive = {(pid, st[2]) for pid, st in procs.items()}
        for key, (g, ticks) in self.used.items():
            row = out.setdefault(g, [0.0, 0])
            row[0] += ticks * 1000.0 / CLK_TCK
            row[1] += key in alive
        return {g: [round(ms, 1), n] for g, (ms, n) in out.items()}


def _kv(text: str) -> dict[str, float]:
    return {k: float(v) for k, _, v in (w.partition("=") for w in text.split()) if v}


def sandbox_cgroups() -> list[Path]:
    """The cgroup of each OpenShell sandbox container on this host."""
    found = set()
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            if Path(f"/proc/{d}/comm").read_text().strip() != "openshell-sandb":
                continue
            rel = Path(f"/proc/{d}/cgroup").read_text().strip().split("::", 1)[1]
        except (OSError, IndexError):
            continue
        found.add(Path("/sys/fs/cgroup" + rel))
    return sorted(found)


def cgroup_counters(dirs: list[Path]) -> list[float]:
    """[cpu ms, periods throttled, throttled ms, pressure some ms, pressure
    full ms, memory MiB, tasks], summed over the given cgroups."""
    tot = [0.0] * 7
    for d in dirs:
        try:
            st = dict(line.split() for line in (d / "cpu.stat").read_text().splitlines())
            tot[0] += int(st.get("usage_usec", 0)) / 1000
            tot[1] += int(st.get("nr_throttled", 0))
            tot[2] += int(st.get("throttled_usec", 0)) / 1000
            for line in (d / "cpu.pressure").read_text().splitlines():
                kind, _, rest = line.partition(" ")
                tot[3 if kind == "some" else 4] += _kv(rest).get("total", 0) / 1000
            tot[5] += int((d / "memory.current").read_text()) / 2**20
            tot[6] += int((d / "pids.current").read_text())
        except (OSError, ValueError):
            continue
    return [round(x, 1) for x in tot]


def host_counters() -> list[float]:
    """[mean CPU MHz, core throttle events, package throttle events, CPU
    pressure some ms]."""
    mhz, core, pkg = [], 0, 0
    for c in glob.glob("/sys/devices/system/cpu/cpu[0-9]*"):
        try:
            mhz.append(int(Path(c, "cpufreq/scaling_cur_freq").read_text()) / 1000)
            core += int(Path(c, "thermal_throttle/core_throttle_count").read_text())
            pkg += int(Path(c, "thermal_throttle/package_throttle_count").read_text())
        except (OSError, ValueError):
            pass
    some = 0.0
    try:
        some = _kv(Path("/proc/pressure/cpu").read_text().splitlines()[0].partition(" ")[2]).get("total", 0) / 1000
    except (OSError, IndexError):
        pass
    return [round(sum(mhz) / len(mhz)) if mhz else 0, core, pkg, round(some, 1)]


_VLLM_METRICS = {
    "vllm:num_requests_running": "run", "vllm:num_requests_waiting": "wait",
    "vllm:kv_cache_usage_perc": "kv", "vllm:num_preemptions_total": "preempt",
    "vllm:prefix_cache_queries_total": "pcq", "vllm:prefix_cache_hits_total": "pch",
    "vllm:prompt_tokens_total": "ptok", "vllm:generation_tokens_total": "gtok",
}


def vllm_metrics(url: str) -> dict[str, float] | None:
    try:
        with urllib.request.urlopen(url, timeout=0.5) as resp:
            text = resp.read().decode()
    except Exception:
        return None
    out: dict[str, float] = {}
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in _VLLM_METRICS:
            key = _VLLM_METRICS[name]
            out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return {k: round(v, 4) for k, v in out.items()}


def slow_sampler(args, stop: threading.Event, out: "queue.Queue[dict]") -> None:
    """Process groups, cgroups, host counters and vLLM metrics, every
    --slow-ms, handed to the main loop to attach to its next row."""
    groups = ProcGroups(args.until_pid or os.getppid())
    cgroups, found_at = [], 0.0
    while not stop.is_set():
        started = time.monotonic()
        if started - found_at > 10:  # a sandbox can be recreated mid-trace
            cgroups, found_at = sandbox_cgroups(), started
        row = {"p": groups.sample(), "h": host_counters()}
        if cgroups:
            row["cg"] = {"sandbox": cgroup_counters(cgroups)}
        if args.vllm_url:
            v = vllm_metrics(args.vllm_url)
            if v is not None:
                row["v"] = v
        row["ts"] = round(time.time() * 1000, 1)
        out.put(row)
        stop.wait(max(0.0, args.slow_ms / 1000 - (time.monotonic() - started)))


def _gpu_extra(pynvml, h) -> list:
    """SM clock (MHz), temperature (C), clock-event (throttle) reason bits."""
    out = []
    for fn in (lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
               lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
               lambda: pynvml.nvmlDeviceGetCurrentClocksEventReasons(h)):
        try:
            out.append(fn())
        except Exception:
            out.append(None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--until-pid", type=int, default=0,
                    help="stop when this process exits (e.g. a sweep's driver)")
    ap.add_argument("--out", default="", help="default output/power/power_<stamp>.jsonl")
    ap.add_argument("--append", action="store_true",
                    help="add to --out instead of replacing it (a resumed run folder); "
                         "readers skip the second header and sort samples by time")
    ap.add_argument("--slow-ms", type=int, default=500,
                    help="interval for process groups, cgroups, host counters and vLLM metrics")
    ap.add_argument("--vllm-url", default=None,
                    help="vLLM metrics URL; default the stack's ($ATK_STACK), '' to skip")
    args = ap.parse_args()
    if args.vllm_url is None:
        from agent import stacks
        args.vllm_url = stacks.get().vllm_url + "/metrics"

    # The CPUs the test runs on (inherited from the harness, ATK_CPUS_PER_SOCKET),
    # read before the sampler moves itself off them.
    pinned = sorted(os.sched_getaffinity(0))
    pinned = pinned if len(pinned) < (os.cpu_count() or 0) else []
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

    with out.open("a" if args.append else "w") as fh:
        fh.write(json.dumps({
            "header": True, "host": socket.gethostname(), "interval_ms": args.interval_ms,
            "gpus": [pynvml.nvmlDeviceGetName(h) for h in gpus],
            "gpu_power_limit_w": [pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000 for h in gpus],
            "numa_nodes": {n: _cpulist(sorted(c)) for n, c in nodes.items()},
            "rapl": [name for name, _ in rapl],
            "rapl_note": "" if all(os.access(f, os.R_OK) for _, f in rapl) and rapl
                         else "RAPL counters not readable at start; CPU power only from when they are",
            "cpus": os.cpu_count(),
            "cpuset": _cpulist(pinned),
            "slow_ms": args.slow_ms,
            "vllm_url": args.vllm_url,
            "fields": "t=wall ms; g=[[power_w, energy_mj, sm_util, mem_util, vram_mib, sm_clock_mhz, "
                      "temp_c, clock_event_reasons] per GPU]; c={numa node: busy fraction}; "
                      "cp=busy fraction of the cpuset (when pinned); "
                      "r={rapl domain: energy_uj}; every slow_ms, on the next row: "
                      "ts=wall ms of the slow sample; p={group: [cpu_ms cumulative, live processes]}; "
                      "cg={sandbox: [cpu_ms, nr_throttled, throttled_ms, psi_some_ms, psi_full_ms, "
                      "mem_mib, tasks]}; h=[cpu_mhz_mean, core_throttle_events, package_throttle_events, "
                      "psi_some_ms]; v={run, wait, kv, preempt, pcq, pch, ptok, gtok}",
        }) + "\n")
        prev = cpu_times()
        slow: "queue.Queue[dict]" = queue.Queue()
        slow_stop = threading.Event()
        threading.Thread(target=slow_sampler, args=(args, slow_stop, slow), daemon=True).start()
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
                    *_gpu_extra(pynvml, h),
                ])
            cur = cpu_times()
            c = {}
            for node, cpus in nodes.items():
                busy = sum(cur[i][0] - prev[i][0] for i in cpus if i in cur and i in prev)
                total = sum(cur[i][1] - prev[i][1] for i in cpus if i in cur and i in prev)
                c[node] = round(busy / total, 3) if total else 0.0
            cp = None
            if pinned:
                total = sum(cur[i][1] - prev[i][1] for i in pinned if i in cur and i in prev)
                cp = round(sum(cur[i][0] - prev[i][0] for i in pinned if i in cur and i in prev) / total, 3) \
                    if total else 0.0
            prev = cur
            r = {}
            for name, f in rapl:
                try:
                    r[name] = int(f.read_text())
                except OSError:
                    pass
            row = {"t": round(time.time() * 1000, 1), "g": g, "c": c, "r": r}
            if cp is not None:
                row["cp"] = cp
            while not slow.empty():
                row.update(slow.get_nowait())
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
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
        slow_stop.set()
    pynvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
