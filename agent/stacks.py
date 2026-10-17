"""Serving stacks: one vLLM + proxy pair per GPU, for running two cells at once.

Each stack is pinned to the GPU and the NUMA node that GPU is attached to, so
two stacks share almost nothing: not the GPU, not the CPU cores the embedding
model and orchestration run on, not the memory. Within a stack everything stays
at concurrency 1, so per-run measurements are comparable with single-stack runs.

    stack a   GPU0, NUMA node 1, vLLM :8000, proxy :8100
    stack b   GPU1, NUMA node 3, vLLM :8001, proxy :8101
    (none)    the original single stack: all GPUs visible, :8000/:8100, no pinning

The GPU-to-NUMA mapping is this machine's (`nvidia-smi topo -m`: GPU0 has CPU
affinity 43-85, node 1; GPU1 has 129-171, node 3). A model with tp_size 2
needs both GPUs and runs only without a stack.

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

    @property
    def proxy_url(self) -> str:
        return f"http://127.0.0.1:{self.proxy_port}/v1"

    @property
    def vllm_url(self) -> str:
        return f"http://127.0.0.1:{self.vllm_port}"

    def cpus(self) -> set[int]:
        """The CPU ids of this stack's NUMA node, or an empty set."""
        if self.numa_node is None:
            return set()
        text = Path(f"/sys/devices/system/node/node{self.numa_node}/cpulist").read_text()
        out: set[int] = set()
        for part in text.strip().split(","):
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
        return out

    def pin_prefix(self) -> str:
        """Command prefix that binds a process to this stack's node."""
        if self.numa_node is None:
            return ""
        return f"numactl --cpunodebind={self.numa_node} --membind={self.numa_node}"

    @property
    def hardware_id(self) -> str:
        return f"gpu{self.gpus}" if self.gpus else ""

    @property
    def numa_pinning(self) -> str:
        return f"node{self.numa_node}" if self.numa_node is not None else ""


DEFAULT = Stack(name="", gpus="", vllm_port=8000, proxy_port=8100, numa_node=None)
STACKS = {
    "a": Stack(name="a", gpus="0", vllm_port=8000, proxy_port=8100, numa_node=1),
    "b": Stack(name="b", gpus="1", vllm_port=8001, proxy_port=8101, numa_node=3),
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
        "NUMA_NODE": "" if s.numa_node is None else str(s.numa_node),
        "PIN": s.pin_prefix(),
    }
    for k, v in values.items():
        print(f"{k}={shlex.quote(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
