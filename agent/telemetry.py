"""The six telemetry record types and a JSONL writer.

Schema mirrors the Phase 1 methodology doc. Records join on run_id. Nothing a
framework reports about itself is trusted: llm_call rows come from the proxy,
tool_call rows from the shared tool layer, hw_sample rows from NVML/RAPL.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCHEMA_VERSION = "0.4.0"


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def now_ms() -> float:
    """Wall clock in milliseconds. Use monotonic_ms for durations."""
    return time.time() * 1000.0


def file_stamp() -> str:
    """YYYYmmdd_HHMMSS suffix for output directories and files.

    US Eastern time, whatever the machine's own clock is set to (the servers
    run on UTC), so a name reads the same wherever the run happened. Follows
    daylight saving. Timestamps inside the records stay epoch milliseconds.
    """
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
@dataclass
class ConfigRecord:
    config_id: str
    framework: str
    framework_version: str
    architecture: str
    model_repo: str
    model_revision: str
    precision: str
    vllm_version: str
    vllm_flags_hash: str
    tool_parser: str
    reasoning_parser: str
    thinking_setting: str
    reasoning_effort: str
    hardware_id: str
    tp_size: int
    numa_pinning: str
    system_prompt_hash: str
    tool_schema_hash: str
    max_llm_calls: int
    max_tokens: int
    temperature: float
    seed: int
    timeout_s: float
    schema_version: str = SCHEMA_VERSION


@dataclass
class RunRecord:
    run_id: str
    config_id: str
    task_id: str
    repeat_idx: int
    run_order_idx: int
    start_ts: float
    end_ts: float = 0.0
    e2e_ms: float = 0.0
    # completed | step_limit (budget ran out) | truncated (answer hit max_tokens)
    # | timeout | framework_error. Decided from the proxy, not the framework.
    status: str = "completed"
    final_answer: str = ""
    error: str = ""

    n_llm_calls: int = 0
    n_tool_calls: int = 0
    n_failed_tool_calls: int = 0
    n_malformed_tool_calls: int = 0
    n_tool_switches: int = 0

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0

    t_llm_ms: float = 0.0
    t_tool_ms: float = 0.0
    t_queue_ms: float = 0.0
    t_orch_ms: float = 0.0

    proc_cpu_ms: float = 0.0
    peak_rss_mb: float = 0.0
    prefix_cache_hit_rate: float | None = None
    energy_gpu_j: float | None = None
    energy_cpu_j: float | None = None
    harness_version: str = ""
    # Free-text label for what this execution was for (--run-code). Not part of
    # config_id: it names the run, it is not a controlled variable.
    run_code: str = ""
    schema_version: str = SCHEMA_VERSION


@dataclass
class LlmCallRecord:
    call_id: str
    run_id: str
    call_idx: int
    request_ts: float
    ttft_ms: float | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    tpot_ms: float | None = None
    finish_reason: str = ""
    n_tool_calls_requested: int = 0
    tool_names: list[str] = field(default_factory=list)
    prompt_chars: int = 0
    n_messages: int = 0
    contains_reasoning_in_history: bool = False
    # Characters of reasoning vs. answer in this response. Tokens are not always
    # reported separately, so characters are the measure that always exists.
    reasoning_chars: int = 0
    answer_chars: int = 0
    payload_path: str = ""
    response_path: str = ""
    error: str = ""
    schema_version: str = SCHEMA_VERSION


@dataclass
class ToolCallRecord:
    tool_call_id: str
    run_id: str
    parent_call_id: str
    tool_idx: int
    tool_name: str
    args_json: str
    args_schema_valid: bool
    latency_ms: float
    status: str  # ok|error|timeout
    error_type: str = ""
    result_bytes: int = 0
    replay_fixture_id: str = ""
    # How the fixture was found: exact | normalized | nearest | recorded | live.
    # "nearest" is approximate and must be excluded or reported separately.
    fixture_match: str = ""
    fixture_similarity: float | None = None
    schema_version: str = SCHEMA_VERSION


@dataclass
class HwSampleRecord:
    ts: float
    host: str
    gpu_idx: int
    gpu_util: float
    vram_used_mb: float
    gpu_power_w: float
    cpu_util: float
    cpu_package_w: float
    ram_mb: float
    schema_version: str = SCHEMA_VERSION


@dataclass
class EvalRecord:
    run_id: str
    metric_name: str
    score: float | None
    threshold: float | None
    passed: bool | None
    judge_model: str = ""
    judge_version: str = ""
    rationale: str = ""
    deterministic_check_passed: bool | None = None
    # What judging this one metric cost: evaluation has a token bill too.
    judge_calls: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_seconds: float = 0.0
    schema_version: str = SCHEMA_VERSION


_TABLES = {
    ConfigRecord: "config",
    RunRecord: "run",
    LlmCallRecord: "llm_call",
    ToolCallRecord: "tool_call",
    HwSampleRecord: "hw_sample",
    EvalRecord: "eval",
}


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------
class TelemetryWriter:
    """Append-only JSONL, one file per table, under a single output directory.

    Thread-safe because the proxy writes llm_call rows from its own thread
    while the harness writes run and tool_call rows.
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._global = threading.Lock()

    def _lock_for(self, table: str) -> threading.Lock:
        with self._global:
            return self._locks.setdefault(table, threading.Lock())

    def write(self, record: Any) -> None:
        table = _TABLES.get(type(record))
        if table is None:
            raise TypeError(f"not a telemetry record: {type(record).__name__}")
        self.write_row(table, asdict(record))

    def write_row(self, table: str, row: dict[str, Any]) -> None:
        line = json.dumps(row, separators=(",", ":"), default=str)
        path = self.out_dir / f"{table}.jsonl"
        with self._lock_for(table):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def blob_path(self, run_id: str, name: str) -> Path:
        """Where raw request/response payloads go."""
        d = self.out_dir / "blobs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    def read(self, table: str) -> list[dict[str, Any]]:
        path = self.out_dir / f"{table}.jsonl"
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
