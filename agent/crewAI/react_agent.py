"""CrewAI ReAct agent (Phase 1 architecture).

A single Agent run directly with `Agent.kickoff`: one agent, native function
calling, no Crew, Task or delegation. That is the CrewAI entry point closest to
the other runtimes; a Task would append CrewAI's "expected criteria for your
final answer" block to every question.

CrewAI builds its own prompt and has no supported way to replace it (templates
merge everything into one user message; `prompt_file` no longer reaches agent
prompts). So the shared system prompt is carried in the role and backstory
fields, and what CrewAI adds around it stays as measured residuals:

- the system message gains one space and a "Your personal goal is:" line;
- the question is prefixed with "Current Task: ";
- every tool is sent with "strict": true, and requests carry tool_choice "auto";
- at the step budget CrewAI makes one last call without tools, asking for a
  final answer, instead of stopping.

Everything CrewAI would otherwise do behind the harness's back is switched off:
tool-result caching, retries, context-window summarising, console panels and
telemetry.
"""

from __future__ import annotations

import copy
import os
from typing import Any

# Read by CrewAI at import time, so set before the first import: no usage
# telemetry or trace upload during a measured run.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_DISABLE_VERSION_CHECK", "true")

from ..config import RunConfig
from ..prompts import build_system_prompt
from ..tools import schemas
from ..tools.registry import ToolRegistry
from ..adapters.base import AgentResult, client_kwargs, sampling_kwargs

# CrewAI requires a goal and always prints it. Kept to a restatement of the
# task so it adds no instruction the other runtimes do not have.
GOAL = "Answer the user's question."


def max_iter_for(max_llm_calls: int) -> int:
    """CrewAI counts iterations, then makes one more call for a forced answer.

    With max_iter = N - 1 the agent gets N - 1 ordinary calls and, if it is
    still asking for tools, a final call without them: N calls at most, the
    same ceiling as the other runtimes. A budget of one call cannot be
    expressed: CrewAI fails with max_iter = 0.
    """
    if max_llm_calls < 2:
        raise ValueError("CrewAI needs a budget of at least 2 LLM calls")
    return max_llm_calls - 1


def split_prompt(prompt: str) -> tuple[str, str]:
    """Shared prompt -> (role, backstory) for "You are {role}. {backstory}".

    CrewAI renders exactly that, so the prompt comes out verbatim apart from
    the space CrewAI puts after the first sentence.
    """
    head, sep, rest = prompt.partition(".")
    if not prompt.startswith("You are ") or not sep:
        raise ValueError("the shared system prompt must open with 'You are <role>.'")
    return head[len("You are "):], rest


class CrewAIReActAgent:
    name = "crewai"

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    def framework_version(self) -> str:
        try:
            from importlib.metadata import version

            return f"crewai=={version('crewai')},openai=={version('openai')}"
        except Exception:
            return "unknown"

    # -- construction ------------------------------------------------------
    def _llm(self):
        from crewai import LLM

        ck, sk = client_kwargs(self.cfg), sampling_kwargs(self.cfg)
        return LLM(
            # The prefix picks CrewAI's OpenAI client; it is not sent upstream.
            model=f"openai/{self.cfg.model.served_name}",
            base_url=ck["base_url"],
            api_key=ck["api_key"],
            timeout=ck["timeout"],
            max_retries=ck["max_retries"],
            temperature=sk["temperature"],
            top_p=sk["top_p"],
            # Sent under this name, as LangGraph does, rather than max_tokens.
            max_completion_tokens=sk["max_tokens"],
            seed=sk["seed"],
            stream=sk["stream"],
            parallel_tool_calls=self.cfg.limits.parallel_tool_calls,
        )

    def _tools(self, registry: ToolRegistry) -> list[Any]:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, ConfigDict

        def build(spec: dict[str, Any]) -> Any:
            class Args(BaseModel):
                # No declared fields and extras allowed, so CrewAI's own
                # argument validation never rejects a call: every call reaches
                # the registry, which counts it and owns the one error string.
                model_config = ConfigDict(extra="allow")

                @classmethod
                def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
                    # CrewAI derives the tool schema from this model. Returning
                    # the shared schema keeps it identical to the other runtimes
                    # instead of a pydantic re-rendering of it.
                    return copy.deepcopy(spec["parameters"])

            call = registry.callable_for(spec["name"])

            class SharedTool(BaseTool):
                name: str = spec["name"]
                description: str = spec["description"]
                args_schema: type[BaseModel] = Args

                def _run(self, **kwargs: Any) -> str:
                    return call(**kwargs)

            return SharedTool()

        return [build(spec) for spec in schemas.TOOL_SCHEMAS]

    # -- run ---------------------------------------------------------------
    def run(self, question: str, registry: ToolRegistry) -> AgentResult:
        try:
            from crewai import Agent
            from crewai.utilities.i18n import I18N_DEFAULT
            from crewai_core.printer import set_suppress_console_output
        except ImportError as exc:
            return AgentResult(status="framework_error", error=f"crewai missing: {exc}")

        try:
            max_iter = max_iter_for(self.cfg.limits.max_llm_calls)
        except ValueError as exc:
            return AgentResult(status="framework_error", error=str(exc))

        # CrewAI renders status panels even when verbose is off.
        set_suppress_console_output(True)
        role, backstory = split_prompt(build_system_prompt(self.cfg.as_of_date))
        agent = Agent(
            role=role,
            goal=GOAL,
            backstory=backstory,
            llm=self._llm(),
            tools=self._tools(registry),
            max_iter=max_iter,
            verbose=False,
            allow_delegation=False,
            # Each of these would add work the harness cannot see: a cached
            # tool result skips the tool layer, a retry repeats the run, and
            # summarising makes extra LLM calls.
            cache=False,
            max_retry_limit=0,
            respect_context_window=False,
        )
        try:
            output = agent.kickoff(question)
        except Exception as exc:
            return AgentResult(status="framework_error", error=f"{type(exc).__name__}: {exc}")

        messages = list(getattr(output, "messages", None) or [])
        forced = I18N_DEFAULT.errors("force_final_answer")
        return AgentResult(
            final_answer=str(output.raw or ""),
            status="completed",
            self_reported_steps=len(messages),
            # The run record's status comes from the proxy; this keeps CrewAI's
            # forced last call visible alongside it.
            raw={"forced_final_answer": any(m.get("content") == forced for m in messages)},
        )
