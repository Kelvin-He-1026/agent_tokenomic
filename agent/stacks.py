"""Serving stacks: one vLLM + proxy pair per GPU, for running two cells at once.

Each stack is pinned to the GPU and the NUMA node that GPU is attached to, so
two stacks share almost nothing: not the GPU, not the CPU cores the embedding
model and orchestration run on, not the memory. Within a stack everything stays
at concurrency 1, so per-run measurements are comparable with single-stack runs.

    stack a   GPU0, NUMA node 0, vLLM :8000, proxy :8100
    stack b   GPU1, NUMA node 1, vLLM :8001, proxy :8101
    stack h   all GPUs, unpinned, vLLM 127.0.0.1:8002, proxy 172.18.0.1:8000
    (none)    the original single stack: all GPUs visible, :8000/:8100, no pinning

The GPU-to-NUMA mapping is this machine's (`nvidia-smi topo -m`). Since
2026-10-29 it has 2x Xeon 6740P, 48 cores per socket with SMT (192 CPUs) and
one NUMA node per socket: GPU0 has CPU affinity 0-47,96-143, node 0; GPU1 has
48-95,144-191, node 1. (Before, 2x Xeon 6787P, 86 cores, four nodes: GPU0 on
node 1, GPU1 on node 3.) A model with tp_size 2
needs both GPUs and runs only without a stack, or on stack h.

ATK_CPUS_PER_SOCKET=N restricts an unpinned stack (h, or none) to N cores per
socket: each socket's GPU-local NUMA node first, then the CPUs numbered
nearest to it. Unset (the default) uses every CPU. It is an affinity, not CPU
hot-unplug: the other cores stay online and idle, and RAPL package energy
still covers the whole package.

Stack h serves a NemoClaw Hermes sandbox (nemoclaw_hermes/). The sandbox
reaches host.openshell.internal:8000, which OpenShell resolves to the
openshell-docker bridge gateway, so the proxy takes that address and port and
vLLM moves to loopback, where the sandbox cannot reach it around the proxy.
The sandbox's recorded inference route and network policy stay unchanged.

Shell scripts read their settings from here, so the mapping lives in one place:

    eval "$(.venv/bin/python -m agent.stacks --shell)"     # uses $ATK_STACK
"""

from __future__ import annotations

import argparse
import os
import shlex
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stack:
    name: str  # "" for the original single stack
    gpus: str  # CUDA_VISIBLE_DEVICES; "" leaves every GPU visible
    vllm_port: int
    proxy_port: int
    numa_node: int | None  # None: no pinning
    gpu_nodes: tuple[int, ...] = (0, 1)  # NUMA nodes the GPUs hang off
    proxy_host: str = "127.0.0.1"
    vllm_host: str = ""  # "" leaves vLLM's own default (all interfaces)

    @property
    def proxy_url(self) -> str:
        return f"http://{self.proxy_host}:{self.proxy_port}/v1"

    @property
    def vllm_url(self) -> str:
        return f"http://127.0.0.1:{self.vllm_port}"

    @property
    def cpus_per_socket(self) -> int:
        """ATK_CPUS_PER_SOCKET, for a stack not already bound to a node; 0: unset."""
        if self.numa_node is not None:
            return 0
        return int(os.environ.get("ATK_CPUS_PER_SOCKET", "0") or 0)

    def cpus(self) -> set[int]:
        """The CPU ids this stack runs on: its NUMA node, its cores per
        socket, or an empty set (unpinned)."""
        if self.numa_node is not None:
            return _cpulist(Path(f"/sys/devices/system/node/node{self.numa_node}/cpulist").read_text())
        if self.cpus_per_socket:
            return socket_cpus(self.cpus_per_socket, self.gpu_nodes)
        return set()

    def pin_prefix(self) -> str:
        """Command prefix that binds a process to this stack's CPUs."""
        if self.numa_node is not None:
            return f"numactl --cpunodebind={self.numa_node} --membind={self.numa_node}"
        if self.cpus_per_socket:
            return f"taskset -c {cpulist(self.cpus())}"
        return ""

    @property
    def hardware_id(self) -> str:
        return f"gpu{self.gpus}" if self.gpus else ""

    @property
    def numa_pinning(self) -> str:
        if self.numa_node is not None:
            return f"node{self.numa_node}"
        if self.cpus_per_socket:
            return f"cpus {cpulist(self.cpus())} ({self.cpus_per_socket}/socket)"
        return ""


def _cpulist(text: str) -> set[int]:
    """'0-3,8' -> {0, 1, 2, 3, 8}"""
    out: set[int] = set()
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def cpulist(cpus: set[int]) -> str:
    """{0, 1, 2, 3, 8} -> '0-3,8'"""
    runs: list[list[int]] = []
    for c in sorted(cpus):
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    return ",".join(f"{a}-{b}" if b > a else str(a) for a, b in runs)


def socket_cpus(per_socket: int, gpu_nodes: tuple[int, ...] = (),
                sys_root: Path = Path("/sys/devices/system")) -> set[int]:
    """per_socket online cores on every socket: the GPU-local NUMA node's
    first, then the other cores of that socket nearest to them by number."""
    local: set[int] = set()
    for node in gpu_nodes:
        f = sys_root / f"node/node{node}/cpulist"
        if f.exists():
            local |= _cpulist(f.read_text())
    sockets: dict[int, list[int]] = {}
    for cpu in sorted(_cpulist((sys_root / "cpu/online").read_text())):
        pkg = int((sys_root / f"cpu/cpu{cpu}/topology/physical_package_id").read_text())
        sockets.setdefault(pkg, []).append(cpu)
    out: set[int] = set()
    for cpus in sockets.values():
        if per_socket > len(cpus):
            raise SystemExit(f"ATK_CPUS_PER_SOCKET={per_socket}, but a socket has {len(cpus)} cores")
        near = [c for c in cpus if c in local] or cpus[-1:]
        out.update(sorted(cpus, key=lambda c: (c not in local, min(abs(c - n) for n in near), c))[:per_socket])
    return out


DEFAULT = Stack(name="", gpus="", vllm_port=8000, proxy_port=8100, numa_node=None)
STACKS = {
    "a": Stack(name="a", gpus="0", vllm_port=8000, proxy_port=8100, numa_node=0),
    "b": Stack(name="b", gpus="1", vllm_port=8001, proxy_port=8101, numa_node=1),
    "h": Stack(name="h", gpus="", vllm_port=8002, proxy_port=8000, numa_node=None,
               proxy_host="172.18.0.1", vllm_host="127.0.0.1"),
}


def get(name: str | None = None) -> Stack:
    """The named stack, $ATK_STACK when no name is given, or the default."""
    name = (name if name is not None else os.environ.get("ATK_STACK", "")).strip().lower()
    if not name:
        return DEFAULT
    if name not in STACKS:
        raise SystemExit(f"unknown stack '{name}'; expected one of {sorted(STACKS)}")
    return STACKS[name]


def pin_current_process(stack: Stack) -> None:
    """Restrict this process, and threads it starts later, to the stack's CPUs."""
    cpus = stack.cpus()
    if cpus:
        os.sched_setaffinity(0, cpus)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell", action="store_true", help="print shell assignments for eval")
    ap.add_argument("--stack", default=None, help="stack name; default $ATK_STACK")
    args = ap.parse_args()
    s = get(args.stack)
    values = {
        "STACK_NAME": s.name,
        "STACK_GPUS": s.gpus,
        "VLLM_PORT": str(s.vllm_port),
        "PROXY_PORT": str(s.proxy_port),
        "PROXY_HOST": s.proxy_host,
        "VLLM_HOST": s.vllm_host,
        "NUMA_NODE": "" if s.numa_node is None else str(s.numa_node),
        "CPUSET": cpulist(s.cpus()),
        "PIN": s.pin_prefix(),
    }
    for k, v in values.items():
        print(f"{k}={shlex.quote(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
