"""LangGraph ReAct agent (Phase 1 architecture).

Phase 2 adds judge_loop.py and router_judge.py beside this file, all on the
same runtime, so the architecture axis varies while the runtime stays fixed.

Uses the prebuilt ReAct agent with everything the parity checklist pins:
the shared system prompt, tools built from the shared schemas, native function
calling through an OpenAI-compatible endpoint, parallel tool calls off, zero
client retries, and a recursion limit derived from the canonical LLM-call
budget.
"""

from __future__ import annotations

from typing import Any

from ..config import RunConfig
from ..prompts import build_system_prompt
from ..tools import schemas
from ..tools.registry import ToolRegistry
from ..adapters.base import AgentResult, client_kwargs, recursion_limit_for


class LangGraphReActAgent:
    name = "langgraph"

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    def framework_version(self) -> str:
        try:
            from importlib.metadata import version

            return f"langgraph=={version('langgraph')},langchain-openai=={version('langchain-openai')}"
        except Exception:
            return "unknown"

    # -- construction ------------------------------------------------------
    def _model(self):
        from langchain_openai import ChatOpenAI

        ck = client_kwargs(self.cfg)
        return ChatOpenAI(
            model=self.cfg.model.served_name,
            base_url=ck["base_url"],
            api_key=ck["api_key"],
            request_timeout=ck["timeout"],
            max_retries=ck["max_retries"],
            temperature=self.cfg.sampling.temperature,
            top_p=self.cfg.sampling.top_p,
            max_tokens=self.cfg.sampling.max_tokens,
            streaming=self.cfg.sampling.stream,
            seed=self.cfg.sampling.seed,
        )

    def _tools(self, registry: ToolRegistry) -> list[Any]:
        from langchain_core.tools import StructuredTool

        # The shared JSON Schema is passed through as-is rather than rebuilt as
        # a pydantic model. Rebuilding drops the per-argument descriptions and
        # the enum on fred_releases.window, which would change what the model
        # sees and quietly break parity with the other frameworks.
        return [
            StructuredTool.from_function(
                func=registry.callable_for(spec["name"]),
                name=spec["name"],
                description=spec["description"],
                args_schema=spec["parameters"],
            )
            for spec in schemas.TOOL_SCHEMAS
        ]

    # -- run ---------------------------------------------------------------
    def run(self, question: str, registry: ToolRegistry) -> AgentResult:
        try:
            from langgraph.prebuilt import create_react_agent
        except ImportError as exc:
            return AgentResult(status="framework_error", error=f"langgraph missing: {exc}")

        tools = self._tools(registry)
        model = self._model().bind_tools(
            tools, parallel_tool_calls=self.cfg.limits.parallel_tool_calls
        )
        agent = create_react_agent(
            model, tools, prompt=build_system_prompt(self.cfg.as_of_date)
        )

        config = {"recursion_limit": recursion_limit_for(self.cfg.limits.max_llm_calls)}
        try:
            state = agent.invoke({"messages": [("user", question)]}, config=config)
        except Exception as exc:
            # GraphRecursionError means the step budget was exhausted, which is
            # a measured outcome rather than a failure.
            if type(exc).__name__ == "GraphRecursionError":
                return AgentResult(status="step_limit", error=str(exc))
            return AgentResult(status="framework_error", error=f"{type(exc).__name__}: {exc}")

        messages = state.get("messages", [])
        final = ""
        for m in reversed(messages):
            content = getattr(m, "content", "")
            if content and not getattr(m, "tool_calls", None):
                final = content if isinstance(content, str) else str(content)
                break
        return AgentResult(
            final_answer=final,
            status="completed",
            self_reported_steps=len(messages),
        )
