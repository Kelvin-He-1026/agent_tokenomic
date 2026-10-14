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
        from ..crewAI.react_agent import CrewAIReActAgent

        return CrewAIReActAgent(cfg)
    if name == "openai_agents":
        from ..openai_agents.react_agent import OpenAIAgentsReActAgent

        return OpenAIAgentsReActAgent(cfg)
    raise ValueError(f"unknown framework '{name}'; expected one of {FRAMEWORKS}")


__all__ = ["AgentAdapter", "AgentResult", "FRAMEWORKS", "get_adapter"]
