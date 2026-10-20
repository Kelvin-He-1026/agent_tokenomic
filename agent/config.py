"""Run configuration: the controlled variables, hashed into a config_id.

Every field here is something Phase 1 holds constant across frameworks. Anything
that differs between frameworks without appearing in this file is either a bug
or a documented residual.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelConfig:
    """Identity of the served model. Recorded, never varied inside a cell.

    Weights are served from a local directory under model/, never pulled from
    the Hub at run time: a benchmark that downloads mid-run is neither
    reproducible nor offline-safe, and `revision` is recorded so the local copy
    stays traceable to a specific upstream commit.
    """

    repo: str  # upstream id, for provenance only
    local_dir: str = ""  # path under model/; defaults to repo's basename
    revision: str = "main"
    precision: str = "bf16"
    served_name: str = ""  # name vLLM answers to; defaults to repo
    tp_size: int = 1
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.90
    tool_call_parser: str = ""
    reasoning_parser: str = ""
    # Thinking mode, pinned per cell and never inherited from the chat template.
    # The server is launched from these (serve_vllm.sh --config) and preflight
    # checks the running server against them. Effort is part of the cell: two
    # efforts are two different experimental conditions, with different config_ids.
    thinking_setting: str = "off"  # off | on
    reasoning_effort: str = ""  # when on: one the model's template accepts
    # Values the chat template accepts, so a typo fails at config load rather
    # than as a template exception mid-run. Qwen3.8: xhigh, medium, low.
    # Empty for a model whose template has on/off but no effort (Nemotron),
    # or no thinking at all (Llama 3).
    reasoning_efforts: tuple[str, ...] = ("xhigh", "medium", "low")
    # Chat-template variable that injects its own "today" into the prompt, if
    # any. Llama 3 writes "Today Date: 26 Jul 2024" unless date_string is set,
    # which would contradict the frozen as_of_date in the system prompt.
    template_date_kwarg: str = ""
    chat_template_hash: str = ""

    def __post_init__(self) -> None:
        if not self.served_name:
            object.__setattr__(self, "served_name", self.repo)
        if not self.local_dir:
            object.__setattr__(self, "local_dir", self.repo.split("/")[-1])
        if isinstance(self.reasoning_efforts, list):
            object.__setattr__(self, "reasoning_efforts", tuple(self.reasoning_efforts))

        if self.thinking_setting not in ("off", "on"):
            raise ValueError(f"thinking_setting must be 'off' or 'on', got '{self.thinking_setting}'")
        if self.thinking_setting == "off" and self.reasoning_effort:
            raise ValueError(
                f"reasoning_effort='{self.reasoning_effort}' has no effect with thinking off; "
                "leave it empty so the recorded cell matches what ran"
            )
        if self.thinking_setting == "on":
            if self.reasoning_efforts:
                if not self.reasoning_effort:
                    raise ValueError(
                        "thinking_setting=on needs an explicit reasoning_effort "
                        f"(one of {list(self.reasoning_efforts)}); the template default is "
                        "never inherited"
                    )
                if self.reasoning_effort not in self.reasoning_efforts:
                    raise ValueError(
                        f"reasoning_effort='{self.reasoning_effort}' is not one of "
                        f"{list(self.reasoning_efforts)} for this model"
                    )
            elif self.reasoning_effort:
                raise ValueError(
                    f"this model has no effort levels (reasoning_efforts is empty), "
                    f"so reasoning_effort must be empty, got '{self.reasoning_effort}'"
                )
            if not self.reasoning_parser:
                raise ValueError(
                    "thinking_setting=on needs a reasoning_parser, or reasoning is glued "
                    "into the answer and thinking tokens cannot be counted"
                )

    def chat_template_kwargs(self, as_of_date: str = "") -> dict[str, Any]:
        """What the server passes to the chat template for every request.

        Templates ignore variables they do not use, so enable_thinking is safe
        to send to a model without thinking (Llama 3).
        """
        kwargs: dict[str, Any] = {"enable_thinking": self.thinking_setting == "on"}
        if self.thinking_setting == "on" and self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.template_date_kwarg and as_of_date:
            from datetime import datetime

            # Llama 3's own format: "06 Apr 2026".
            kwargs[self.template_date_kwarg] = datetime.strptime(
                as_of_date, "%Y-%m-%d"
            ).strftime("%d %b %Y")
        return kwargs


@dataclass(frozen=True)
class SamplingConfig:
    """Parity rows 6 and 7. Identical across all three frameworks."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    seed: int = 1234
    stream: bool = True  # required for TTFT at the proxy

    def __post_init__(self) -> None:
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")


@dataclass(frozen=True)
class LimitsConfig:
    """Parity rows 9, 10, 11.

    max_llm_calls is the canonical budget. Each adapter translates it into its
    own framework's setting; see adapters/base.py.
    """

    max_llm_calls: int = 10
    request_timeout_s: float = 120.0
    task_timeout_s: float = 600.0
    client_retries: int = 0
    parallel_tool_calls: bool = False
    # Zero derives a shared run ceiling of max_llm_calls * sampling.max_tokens.
    max_total_output_tokens: int = 0
    accounting_policy: str = "proxy-v1"

    def __post_init__(self) -> None:
        for key in ("max_llm_calls", "max_total_output_tokens"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key == "max_llm_calls" else 0):
                raise ValueError(f"invalid {key}: {value!r}")
        if self.request_timeout_s <= 0 or self.task_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        if self.accounting_policy != "proxy-v1":
            raise ValueError("unsupported accounting_policy")


@dataclass(frozen=True)
class RunConfig:
    """One experimental cell."""

    framework: str  # langgraph | crewai | openai_agents
    model: ModelConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)

    architecture: str = "react"
    as_of_date: str = "2026-04-06"  # frozen clock; see prompts.py
    base_url: str = "http://127.0.0.1:8100/v1"  # the proxy, never vLLM directly
    api_key: str = "EMPTY"

    tool_mode: str = "replay"  # replay | record | live
    # Jaccard overlap required for a "nearest" fixture match. Higher is safer;
    # matches below this become tool errors rather than wrong answers.
    fixture_match_threshold: float = 0.6
    fixtures_dir: str = "dataset/fixtures"
    retrieval_top_k: int = 5
    # Embedding weights are local too, so retrieval never reaches the network.
    embedding_model: str = "BAAI/bge-large-en-v1.5"  # upstream id, provenance
    embedding_local_dir: str = "bge-large-en-v1.5"  # under models_dir
    embedding_device: str = "cpu"  # cpu keeps retrieval out of vLLM's VRAM and power
    models_dir: str = "model"
    chroma_dir: str = "db/chroma"
    chroma_collection: str = "documents"
    # Tools offered to the agent. Empty = the Phase 1 three (retrieve,
    # web_search, fred_releases). The choice enters config_id through
    # tool_schema_hash, so it is not hashed a second time here.
    tools: tuple[str, ...] = ()
    # Where read_file looks for task attachments (GAIA).
    attachments_dir: str = "GAIA_stack/documents"
    # Search engine behind web_search: "tavily" (API key, credits) or "ddg"
    # (DuckDuckGo via ddgs, no key). Hashed into config_id only when not the
    # default, so Phase 1 config_ids do not change.
    web_search_backend: str = "tavily"

    # Environment provenance, filled by the harness at run time.
    framework_version: str = ""
    vllm_version: str = ""
    hardware_id: str = ""
    numa_pinning: str = ""
    harness_version: str = "0.1.0"

    def config_id(self, prompt_hash: str, tool_schema_hash: str) -> str:
        """Stable hash over everything that defines the cell.

        The two hashes are passed in rather than recomputed so the id changes
        whenever the prompt or the tool schemas change, which is exactly the
        parity evidence the methodology asks for.
        """
        payload = {
            "cell": _plain(self),
            "prompt_hash": prompt_hash,
            "tool_schema_hash": tool_schema_hash,
        }
        if self.web_search_backend != "tavily":
            payload["web_search_backend"] = self.web_search_backend
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        raw = dict(raw)
        nested = {
            "model": ModelConfig,
            "sampling": SamplingConfig,
            "limits": LimitsConfig,
        }
        for key, klass in nested.items():
            if isinstance(raw.get(key), dict):
                raw[key] = klass(**raw[key])
        if isinstance(raw.get("tools"), list):
            raw["tools"] = tuple(raw["tools"])
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def resolve(self, relative: str) -> Path:
        """Resolve a config path against the repo root."""
        p = Path(relative)
        return p if p.is_absolute() else REPO_ROOT / p

    def model_path(self) -> Path:
        """Local weights directory for the model under test."""
        return self.resolve(self.models_dir) / self.model.local_dir

    def embedding_path(self) -> Path:
        """Local weights directory for the retrieval embedding model."""
        return self.resolve(self.models_dir) / self.embedding_local_dir

    def require_local_models(self) -> list[str]:
        """Return the local paths that are missing, for a clear early failure."""
        missing = []
        for label, path in (
            (self.model.repo, self.model_path()),
            (self.embedding_model, self.embedding_path()),
        ):
            if not path.exists():
                missing.append(f"{label} -> {path}")
        return missing


def _plain(obj: Any) -> Any:
    """asdict() that leaves provenance fields out of the hash input."""
    d = asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
    if isinstance(d, dict):
        return {k: _plain(v) for k, v in d.items() if k not in _PROVENANCE}
    return d


# Filled in after the fact; they describe the environment rather than define
# the cell, so they must not change the config_id.
# base_url too: which stack's proxy a run went through says where it ran, not
# what was run, so the same cell on stack a and stack b shares one config_id.
_PROVENANCE = {"framework_version", "vllm_version", "hardware_id", "numa_pinning", "base_url"}
# Covered by another hash (tools -> tool_schema_hash), a location rather than a
# condition (attachments_dir), or added to config_id only when not the default
# (web_search_backend). Left out so existing config_ids do not change.
_PROVENANCE |= {"tools", "attachments_dir", "web_search_backend"}
