"""Run one task on one framework and assemble the run record.

The harness owns the clock, the run_id, and the joins. It binds the proxy to
the run before the agent starts, so llm_call rows land under the right run_id
without depending on framework header support.
"""

from __future__ import annotations

import json
import os
import resource
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import httpx

from ..adapters.base import AgentAdapter, AgentResult
from ..config import RunConfig
from ..evaluation import OUTCOME_POLICY_VERSION, answer_fingerprint
from ..accounting import cap_hits, completion_type
from ..prompts import build_system_prompt, prompt_hash
from ..spans import build_spans
from ..telemetry import (
    ConfigRecord,
    RunRecord,
    TelemetryWriter,
    ToolCallRecord,
    monotonic_ms,
    new_id,
    now_ms,
)
from ..tools import schemas
from ..tools.registry import ToolRegistry


@dataclass
class Task:
    task_id: str
    question: str
    raw: dict[str, Any]

    @classmethod
    def load_all(cls, path: str | Path) -> list["Task"]:
        items = json.loads(Path(path).read_text())
        from ..prompts import task_prompt

        return [cls(task_id=str(it["task_id"]), question=task_prompt(it), raw=it) for it in items]


class Runner:
    def __init__(
        self,
        cfg: RunConfig,
        adapter: AgentAdapter,
        out_dir: str | Path,
        session: str = "",
        run_code: str = "",
        writer: TelemetryWriter | None = None,
        adapter_factory: Callable[[RunConfig], AgentAdapter] | None = None,
    ):
        self.session = session or new_id("sess_")
        self.run_code = run_code
        self.cfg = cfg
        self.adapter = adapter
        # Concurrent runs (the load test) share one writer, so rows from
        # several threads append to each table through one lock, and build a
        # fresh adapter per run around the run's own proxy URL.
        self.writer = writer or TelemetryWriter(out_dir)
        self.adapter_factory = adapter_factory
        self.prompt_hash = prompt_hash(build_system_prompt(cfg.as_of_date))
        self.schema_hash = schemas.tool_schema_hash(cfg.tools)
        self.config_id = cfg.config_id(self.prompt_hash, self.schema_hash)
        self._config_written = False
        self.last_fixture_matches: dict[str, int] = {}

    # -- config row --------------------------------------------------------
    def write_config(self) -> str:
        if self._config_written:
            return self.config_id
        m = self.cfg.model
        self.writer.write(
            ConfigRecord(
                config_id=self.config_id,
                framework=self.adapter.name,
                framework_version=self.adapter.framework_version(),
                architecture=self.cfg.architecture,
                model_repo=m.repo,
                model_revision=m.revision,
                precision=m.precision,
                vllm_version=self.cfg.vllm_version,
                vllm_flags_hash="",
                tool_parser=m.tool_call_parser,
                reasoning_parser=m.reasoning_parser,
                thinking_setting=m.thinking_setting,
                reasoning_effort=m.reasoning_effort,
                hardware_id=self.cfg.hardware_id,
                tp_size=m.tp_size,
                numa_pinning=self.cfg.numa_pinning,
                system_prompt_hash=self.prompt_hash,
                tool_schema_hash=self.schema_hash,
                max_llm_calls=self.cfg.limits.max_llm_calls,
                max_tokens=self.cfg.sampling.max_tokens,
                temperature=self.cfg.sampling.temperature,
                seed=self.cfg.sampling.seed,
                timeout_s=self.cfg.limits.task_timeout_s,
                max_total_output_tokens=self.output_budget,
                accounting_policy=self.cfg.limits.accounting_policy,
            )
        )
        self._config_written = True
        return self.config_id

    # -- one run -----------------------------------------------------------
    def run_task(self, task: Task, repeat_idx: int, run_order_idx: int, *, concurrent: bool = False,
                 concurrency: int = 1, vu_idx: int = -1, measured: bool = True,
                 on_start: Callable[[str], None] | None = None) -> RunRecord:
        """One run. concurrent=True (the load test) addresses the proxy as
        /r/<run_id>/v1, so other runs may be bound at the same time; it needs
        an adapter_factory, since each run's adapter carries its own URL."""
        self.write_config()
        run_id = new_id("run_")
        cfg, adapter = self.cfg, self.adapter
        if concurrent:
            root = self.cfg.base_url.rsplit("/v1", 1)[0]
            cfg = replace(self.cfg, base_url=f"{root}/r/{run_id}/v1")
            adapter = self.adapter_factory(cfg)
        registry = ToolRegistry(cfg, self.writer, run_id)
        self._bind_proxy(run_id, concurrent)
        if on_start:
            on_start(run_id)

        cpu_before = _cpu_ms()
        started_wall, started = now_ms(), monotonic_ms()
        try:
            result = adapter.run(task.question, registry)
        except Exception as exc:  # an adapter bug, not a framework outcome
            result = AgentResult(status="framework_error", error=f"{type(exc).__name__}: {exc}")
        e2e_ms = monotonic_ms() - started
        ended_wall = now_ms()
        self._end_run(run_id)
        self._await_proxy_flush(run_id)
        agent_tools = self._write_agent_tools(run_id, result.raw.get("tool_calls") or [])

        record = RunRecord(
            run_id=run_id,
            config_id=self.config_id,
            task_id=task.task_id,
            repeat_idx=repeat_idx,
            run_order_idx=run_order_idx,
            start_ts=started_wall,
            end_ts=ended_wall,
            e2e_ms=e2e_ms,
            status=result.status,
            final_answer=result.final_answer,
            error=result.error,
            framework_status=result.status,
            forced_final_answer=bool(result.raw.get("forced_final_answer")),
            max_llm_calls=self.cfg.limits.max_llm_calls,
            max_tokens_per_call=self.cfg.sampling.max_tokens,
            max_total_output_tokens=self.output_budget,
            accounting_policy=self.cfg.limits.accounting_policy,
            budget_enforced=True,
            session=self.session,
            task_fingerprint=answer_fingerprint(task.raw, ""),
            outcome_policy_version=OUTCOME_POLICY_VERSION,
            n_tool_calls=registry.n_calls,
            n_failed_tool_calls=registry.n_failed + sum(1 for t in agent_tools if t.status != "ok"),
            n_malformed_tool_calls=registry.n_malformed,
            n_tool_switches=registry.n_switches,
            t_tool_ms=registry.total_ms + sum(t.latency_ms for t in agent_tools),
            proc_cpu_ms=_cpu_ms() - cpu_before,
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            harness_version=self.cfg.harness_version,
            run_code=self.run_code,
            concurrency=concurrency,
            vu_idx=vu_idx,
            measured=measured,
        )
        self.last_fixture_matches = registry.fixture_matches
        self._fold_in_llm_calls(record)
        # A slow tool can cross the deadline without another model request.
        if record.e2e_ms > self.cfg.limits.task_timeout_s * 1000:
            record.status, record.termination_reason = "timeout", "timeout"
        if (not calls_complete(self._proxy_snapshot, run_id)
                or self._proxy_snapshot.get("call_idx") != record.n_llm_calls):
            record.status, record.termination_reason = "framework_error", "telemetry_missing"
        snap = self._proxy_snapshot
        record.closed_by = snap.get("ended_by", "")
        record.upstream_errors = int(snap.get("upstream_errors", 0))
        if record.closed_by not in ("", "finished") and (snap.get("refused_after_end")
                                                          or record.status != "completed"):
            # The benchmark closed it mid-task (a load test's filler): what the
            # agent did after that is the benchmark's doing, not an outcome.
            record.error = f"closed by the benchmark ({record.closed_by}); was {record.status}: {record.error}"[:2000]
            record.status, record.termination_reason = "cancelled", "closed_by_benchmark"
        record.completion_type = completion_type(record.status, record.termination_reason,
                                                 record.forced_final_answer, record.last_finish_reason)
        self.writer.write(record)
        tools = [asdict(t) for t in registry.calls + agent_tools]
        for span in build_spans(run_id, (record.start_ts, record.end_ts), self._run_calls, tools,
                                result.raw.get("phases") or []):
            self.writer.write(span)
        return record

    def _write_agent_tools(self, run_id: str, events: list[dict]) -> list[ToolCallRecord]:
        """tool_call rows for tools the agent ran itself (Hermes, in its
        sandbox), from the timings its plugin recorded. Same table as the
        shared tool layer's rows; fixture_match "agent" tells them apart."""
        rows = []
        for i, t in enumerate(events, 1):
            row = ToolCallRecord(
                tool_call_id=new_id("tc_"), run_id=run_id, parent_call_id=t.get("id", ""), tool_idx=i,
                tool_name=t.get("tool", ""), args_json=t.get("args", ""), args_schema_valid=True,
                latency_ms=float(t.get("latency_ms", 0.0)), status=t.get("status", "ok"),
                error_type=t.get("error_type", ""), result_bytes=int(t.get("result_bytes", 0)),
                fixture_match="agent", start_ts=float(t.get("start_ts", 0.0)),
            )
            self.writer.write(row)
            rows.append(row)
        return rows

    # -- joins -------------------------------------------------------------
    def _fold_in_llm_calls(self, record: RunRecord) -> None:
        """Take LLM numbers from the proxy's rows, never from the framework."""
        calls = [c for c in self.writer.read("llm_call") if c["run_id"] == record.run_id]
        self._run_calls = calls
        record.n_llm_calls = len(calls)
        record.forced_final_answer = record.forced_final_answer or any(c.get("forced_final_answer") for c in calls)
        record.input_tokens = sum(c.get("prompt_tokens", 0) for c in calls)
        record.output_tokens = sum(c.get("completion_tokens", 0) for c in calls)
        record.reasoning_tokens = sum(c.get("reasoning_tokens", 0) for c in calls)
        record.cached_input_tokens = sum(c.get("cached_tokens", 0) for c in calls)
        record.t_llm_ms = sum(c.get("latency_ms", 0.0) for c in calls)
        record.usage_complete = all(not c.get("usage_missing", False) for c in calls)
        if getattr(self.adapter, "tools_outside_registry", False):
            # An agent whose tools run on its own side (Hermes) never reaches the
            # shared tool layer: count the tool calls the model asked for, from
            # the proxy. Their time stays in T_Orch; nothing here can time them.
            record.n_tool_calls = sum(c.get("n_tool_calls_requested", 0) for c in calls)
        snapshot = self._proxy_snapshot
        record.blocked_llm_requests = snapshot.get("blocked_requests", 0)
        for key, value in cap_hits(asdict(record), calls).items():
            setattr(record, key, value)
        proxy_reason = snapshot.get("termination_reason", "")
        if proxy_reason:
            record.termination_reason = proxy_reason
            record.status = {"call_cap": "step_limit", "output_cap": "truncated",
                             "timeout": "timeout"}.get(proxy_reason, "framework_error")
        if calls and not record.final_answer:
            last = max(calls, key=lambda c: c.get("call_idx", 0))
            path = last.get("response_path")
            if path and last.get("finish_reason") in ("stop", "length"):
                data = json.loads(Path(path).read_text())
                message = (data.get("choices") or [{}])[0].get("message") or {}
                if not message.get("tool_calls"):
                    record.final_answer = message.get("content") or ""
        self._classify_ending(record, calls)
        record.completion_type = completion_type(record.status, record.termination_reason,
                                                 record.forced_final_answer, record.last_finish_reason)
        # T_Orch is a residual: no replay arm, so it absorbs measurement error.
        record.t_orch_ms = max(
            0.0, record.e2e_ms - record.t_llm_ms - record.t_tool_ms - record.t_queue_ms
        )
        if record.input_tokens:
            record.prefix_cache_hit_rate = record.cached_input_tokens / record.input_tokens

    @staticmethod
    def _classify_ending(record: RunRecord, calls: list[dict]) -> None:
        """Decide how the run really ended, from the proxy's view of the last call.

        Frameworks disagree about running out of budget: LangGraph's prebuilt
        agent does not raise, it substitutes "Sorry, need more steps to process
        this request." as the answer. Trusting the framework would record that
        as a completed run. The last call's finish_reason says what happened
        whatever the framework did afterwards:

            tool_calls  the model asked to keep going and was stopped
            length      the final answer hit max_tokens and was cut off
        """
        if record.termination_reason:
            return
        if not calls:
            if record.status != "completed":
                record.termination_reason = record.status
                return
            record.termination_reason = "telemetry_missing" if record.budget_enforced else record.status
            if record.budget_enforced:
                record.status = "framework_error"
            return
        last = max(calls, key=lambda c: c.get("call_idx", 0))
        reason = last.get("finish_reason", "")
        if reason == "length":
            record.status = "truncated"
            record.termination_reason = "output_cap" if record.total_output_cap_reached else "per_call_output_cap"
            record.error = (
                f"final answer cut off at max_tokens ({last.get('completion_tokens')} tokens, "
                f"{last.get('reasoning_tokens', 0)} of them thinking)"
            )
        elif record.status == "step_limit":
            record.termination_reason = "call_cap"
        elif record.status != "completed":
            record.termination_reason = "upstream_error" if last.get("error") else record.status
        elif reason == "tool_calls":
            at_cap = not record.max_llm_calls or record.call_cap_reached
            record.status = "step_limit" if at_cap else "framework_error"
            record.termination_reason = "call_cap" if at_cap else "incomplete_trajectory"
            record.error = (
                "run ended while the model was requesting a tool; "
                f"framework returned {record.final_answer[:60]!r}"
            )
        elif last.get("error"):
            record.status, record.termination_reason = "framework_error", "upstream_error"
        elif reason != "stop":
            record.status, record.termination_reason = "framework_error", "incomplete_trajectory"
        else:
            # A forced response still consumes one of the same N calls. A
            # valid stop within that cap is eligible for completion, exactly
            # like the other frameworks; preserve its provenance separately.
            record.termination_reason = "forced_final_answer" if record.forced_final_answer else "natural_stop"

    @property
    def output_budget(self) -> int:
        return (self.cfg.limits.max_total_output_tokens
                or self.cfg.limits.max_llm_calls * self.cfg.sampling.max_tokens)

    def _bind_proxy(self, run_id: str, concurrent: bool = False) -> None:
        """Tell the proxy which run and which directory its rows belong to."""
        url = self.cfg.base_url.rsplit("/v1", 1)[0] + "/control/run"
        body = {
            "run_id": run_id,
            "concurrent": concurrent,
            "out_dir": str(self.writer.out_dir.resolve()),
            "session": self.session,
            "forced_final_markers": getattr(self.adapter, "forced_final_markers", lambda: [])(),
            "limits": {
                "max_llm_calls": self.cfg.limits.max_llm_calls,
                "max_tokens_per_call": self.cfg.sampling.max_tokens,
                "max_total_output_tokens": self.output_budget,
                "task_timeout_s": self.cfg.limits.task_timeout_s,
            },
        }
        try:
            # A run that timed out can leave its last call streaming (an agent
            # process outliving its run). The proxy refuses to rebind until
            # that call ends, which is at most one request timeout away.
            wait_until = time.monotonic() + self.cfg.limits.request_timeout_s + 30
            while True:
                resp = httpx.post(url, json=body, timeout=5.0)
                busy = resp.status_code == 409 and not resp.json().get("conflict")
                if not busy or concurrent or time.monotonic() > wait_until:
                    break
                time.sleep(2.0)
            if busy:
                raise SystemExit(
                    f"proxy still has a request in flight from the previous run after "
                    f"{self.cfg.limits.request_timeout_s + 30:.0f}s; "
                    "restart it with scripts/stop_stack.sh and scripts/start_stack.sh"
                )
            if resp.status_code == 409:
                info = resp.json()
                raise SystemExit(
                    f"proxy is already in use by session {info.get('holder')} "
                    f"({info.get('held_for_s')}s ago).\n"
                    "  Two harnesses on one proxy interleave their telemetry and "
                    "queue behind each other, so the data would be unusable.\n"
                    "  Wait for the other run, or start a second proxy on another port."
                )
            resp.raise_for_status()
            bound = resp.json()
            if (bound.get("bound") != run_id or bound.get("accounting_policy") != "proxy-v1"
                    or bound.get("limits") != {"max_llm_calls": self.cfg.limits.max_llm_calls,
                        "max_tokens_per_call": self.cfg.sampling.max_tokens,
                        "max_total_output_tokens": self.output_budget,
                        "task_timeout_s": self.cfg.limits.task_timeout_s}):
                raise RuntimeError("proxy does not acknowledge canonical budget enforcement; restart the proxy")
        except Exception as exc:
            raise RuntimeError(f"cannot bind benchmark proxy: {exc}") from exc

    def _end_run(self, run_id: str) -> None:
        """Close the run at the proxy: a request it still makes is refused.
        Older proxies have no /control/end; nothing is lost without it."""
        try:
            httpx.post(self.cfg.base_url.rsplit("/v1", 1)[0] + "/control/end",
                       json={"run_id": run_id, "reason": "finished"}, timeout=5.0)
        except Exception:
            pass

    def _await_proxy_flush(self, run_id: str, timeout_s: float = 10.0) -> None:
        """Wait for the proxy to finish writing this run's llm_call rows.

        A streamed response completes for the client before the proxy has
        recorded it, so reading telemetry immediately loses the last call.
        Measured latency is unaffected: e2e is stopped before this runs.
        """
        url = self.cfg.base_url.rsplit("/v1", 1)[0] + "/control/pending"
        deadline = time.monotonic() + timeout_s
        self._proxy_snapshot = {}
        while time.monotonic() < deadline:
            try:
                self._proxy_snapshot = httpx.get(url, params={"run_id": run_id}, timeout=2.0).json()
                if self._proxy_snapshot.get("inflight", 0) == 0:
                    return
            except Exception:
                return
            time.sleep(0.01)


def calls_complete(snapshot: dict, run_id: str) -> bool:
    return snapshot.get("run_id") == run_id and snapshot.get("inflight") == 0


def _cpu_ms() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return (ru.ru_utime + ru.ru_stime) * 1000.0
