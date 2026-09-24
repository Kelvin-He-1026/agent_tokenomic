"""Adapter interface: what each framework must expose, and what it must not.

An adapter's only job is to run one ReAct task and report what happened. It
never implements a tool, never writes a prompt, and never decides sampling
parameters: those come from the shared modules so parity is structural rather
than a matter of remembering to copy values.

Each adapter also declares how it translates the canonical LLM-call budget into
its own framework's setting (parity row 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import RunConfig
from ..tools.registry import ToolRegistry


@dataclass
class AgentResult:
    """What the harness needs from a framework run."""

    final_answer: str = ""
    status: str = "completed"  # completed|step_limit|timeout|framework_error
    error: str = ""
    # Framework's own view, recorded for comparison against the proxy but never
    # used as the measurement. A gap between the two is itself a finding.
    self_reported_steps: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class AgentAdapter(Protocol):
    name: str

    def framework_version(self) -> str: ...

    def run(self, question: str, registry: ToolRegistry) -> AgentResult: ...


def recursion_limit_for(max_llm_calls: int) -> int:
    """LangGraph counts graph supersteps, not LLM calls.

    A ReAct loop alternates agent and tool nodes, so N LLM calls need 2N-1
    supersteps; +1 keeps the final agent node from tripping the limit.
    """
    return 2 * max_llm_calls


def client_kwargs(cfg: RunConfig) -> dict[str, Any]:
    """Connection settings every adapter must use (parity rows 4, 5, 10, 11)."""
    return {
        "base_url": cfg.base_url,
        "api_key": cfg.api_key,
        "timeout": cfg.limits.request_timeout_s,
        "max_retries": cfg.limits.client_retries,
    }


def sampling_kwargs(cfg: RunConfig) -> dict[str, Any]:
    """Sampling settings every adapter must use (parity rows 6, 7)."""
    return {
        "temperature": cfg.sampling.temperature,
        "top_p": cfg.sampling.top_p,
        "max_tokens": cfg.sampling.max_tokens,
        "seed": cfg.sampling.seed,
        "stream": cfg.sampling.stream,
    }
