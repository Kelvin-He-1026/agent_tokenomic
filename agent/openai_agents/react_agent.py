"""OpenAI Agents SDK ReAct agent (Phase 1 architecture).

A single Agent with tools is the SDK's ReAct loop: call the model, run the
tool it asked for, call the model again, stop on a plain answer. Everything the
parity checklist pins comes from the shared modules: the system prompt, tools
built from the shared schemas, native function calling through an
OpenAI-compatible endpoint, parallel tool calls off, zero client retries, and a
turn limit equal to the canonical LLM-call budget.

Three SDK defaults are switched off because each would break parity silently:
the Responses API (vLLM and the proxy speak Chat Completions), strict JSON
schemas (the SDK would rewrite the shared schemas), and tracing (it uploads to
OpenAI in the background).

Known residual, kept on purpose: the SDK adds "strict": false to every tool
definition and has no setting to omit it. It is left in rather than stripped
outside the SDK, so parity_check.py reports the tools field as differing.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..config import RunConfig
from ..prompts import build_system_prompt
from ..tools import schemas
from ..tools.registry import ToolRegistry
from ..adapters.base import AgentResult, client_kwargs, sampling_kwargs


class OpenAIAgentsReActAgent:
    name = "openai_agents"

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    def framework_version(self) -> str:
        try:
            from importlib.metadata import version

            return f"openai-agents=={version('openai-agents')},openai=={version('openai')}"
        except Exception:
            return "unknown"

    # -- construction ------------------------------------------------------
    def _model_settings(self):
        from agents import ModelSettings

        sk = sampling_kwargs(self.cfg)
        return ModelSettings(
            temperature=sk["temperature"],
            top_p=sk["top_p"],
            parallel_tool_calls=self.cfg.limits.parallel_tool_calls,
            # extra_args goes into the request body as top-level keys.
            # ModelSettings has no seed field, and its max_tokens is sent under
            # that name, where LangGraph sends max_completion_tokens: the same
            # limit to vLLM, but one name keeps the request bodies identical.
            extra_args={"seed": sk["seed"], "max_completion_tokens": sk["max_tokens"]},
        )

    def _tools(self, registry: ToolRegistry) -> list[Any]:
        from agents import FunctionTool

        def invoker(tool_name: str):
            async def _invoke(_ctx: Any, raw_args: str) -> str:
                # Arguments that are not a JSON object go to the registry as an
                # empty dict, so they are counted as a malformed call and get
                # the shared error string instead of an SDK exception.
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                # Called inline rather than through a worker thread: nothing
                # else runs on this loop at concurrency 1, and a thread hop
                # would be charged to T_Orch.
                return registry.call(tool_name, args)

            return _invoke

        # FunctionTool takes the shared JSON Schema as-is. The function_tool
        # decorator would rebuild it from a Python signature, and strict mode
        # would rewrite it, either of which changes what the model sees.
        return [
            FunctionTool(
                name=spec["name"],
                description=spec["description"],
                params_json_schema=spec["parameters"],
                on_invoke_tool=invoker(spec["name"]),
                strict_json_schema=False,
            )
            for spec in schemas.tool_specs(self.cfg.tools)
        ]

    # -- run ---------------------------------------------------------------
    def run(self, question: str, registry: ToolRegistry) -> AgentResult:
        try:
            import agents  # noqa: F401
        except ImportError as exc:
            return AgentResult(status="framework_error", error=f"openai-agents missing: {exc}")
        return asyncio.run(self._run(question, registry))

    async def _run(self, question: str, registry: ToolRegistry) -> AgentResult:
        from agents import Agent, OpenAIChatCompletionsModel, RunConfig as SdkRunConfig, Runner
        from agents.exceptions import MaxTurnsExceeded
        from openai import AsyncOpenAI

        ck = client_kwargs(self.cfg)
        # Built inside the coroutine: the client's connection pool belongs to
        # the event loop, and each run gets a fresh loop.
        async with AsyncOpenAI(
            base_url=ck["base_url"],
            api_key=ck["api_key"],
            timeout=ck["timeout"],
            max_retries=ck["max_retries"],
        ) as client:
            agent = Agent(
                name="react_agent",
                instructions=build_system_prompt(self.cfg.as_of_date),
                model=OpenAIChatCompletionsModel(
                    model=self.cfg.model.served_name, openai_client=client
                ),
                model_settings=self._model_settings(),
                tools=self._tools(registry),
            )
            # A turn is one model call, so the canonical budget maps 1:1.
            options = {
                "max_turns": self.cfg.limits.max_llm_calls,
                "run_config": SdkRunConfig(tracing_disabled=True),
            }
            try:
                if self.cfg.sampling.stream:
                    result = Runner.run_streamed(agent, question, **options)
                    async for _ in result.stream_events():
                        pass
                else:
                    result = await Runner.run(agent, question, **options)
            except MaxTurnsExceeded as exc:
                # The step budget was exhausted, which is a measured outcome
                # rather than a failure.
                return AgentResult(status="step_limit", error=str(exc))
            except Exception as exc:
                return AgentResult(status="framework_error", error=f"{type(exc).__name__}: {exc}")

        final = result.final_output
        return AgentResult(
            final_answer=final if isinstance(final, str) else str(final or ""),
            status="completed",
            self_reported_steps=len(result.raw_responses),
        )
