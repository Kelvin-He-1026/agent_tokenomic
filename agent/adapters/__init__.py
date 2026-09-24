"""Adapter registry. Adding a framework means adding one entry here."""

from __future__ import annotations

from ..config import RunConfig
from .base import AgentAdapter, AgentResult

FRAMEWORKS = ("langgraph", "crewai", "openai_agents")


def get_adapter(name: str, cfg: RunConfig) -> AgentAdapter:
    if name == "langgraph":
        from ..langGraph.react_agent import LangGraphReActAgent

        return LangGraphReActAgent(cfg)
    if name == "crewai":
        from .crewai_agent import CrewAIAdapter

        return CrewAIAdapter(cfg)
    if name == "openai_agents":
        from .openai_agents_agent import OpenAIAgentsAdapter

        return OpenAIAgentsAdapter(cfg)
    raise ValueError(f"unknown framework '{name}'; expected one of {FRAMEWORKS}")


__all__ = ["AgentAdapter", "AgentResult", "FRAMEWORKS", "get_adapter"]
