"""Per-run tool registry: times every call, logs it, and owns the error path.

Parity row 12. Every exception is caught here and returned to the model as one
fixed error string, so no framework-level error handler ever fires and all
three frameworks see byte-identical error text.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..config import RunConfig
from ..env import redact
from ..telemetry import ToolCallRecord, TelemetryWriter, monotonic_ms, new_id
from . import impl, schemas


class ToolRegistry:
    """Created once per run. Adapters call `.call()`; nothing else touches tools."""

    def __init__(self, cfg: RunConfig, writer: TelemetryWriter, run_id: str):
        self.cfg = cfg
        self.writer = writer
        self.run_id = run_id
        self.store = impl.FixtureStore(
            cfg.resolve(cfg.fixtures_dir), threshold=cfg.fixture_match_threshold
        )
        # Stamped into fixtures recorded during a probe, so the fixture set
        # shows which model's phrasing produced each query.
        self.source = f"{cfg.framework}/{cfg.model.served_name}"
        self.tool_idx = 0
        self.parent_call_id = ""  # set by the proxy hook when known
        self.calls: list[ToolCallRecord] = []

    # -- counters the run record needs ------------------------------------
    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.calls if c.status != "ok")

    @property
    def n_malformed(self) -> int:
        return sum(1 for c in self.calls if not c.args_schema_valid)

    @property
    def total_ms(self) -> float:
        return sum(c.latency_ms for c in self.calls)

    @property
    def fixture_matches(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.calls:
            if c.fixture_match:
                out[c.fixture_match] = out.get(c.fixture_match, 0) + 1
        return out

    @property
    def n_switches(self) -> int:
        names = [c.tool_name for c in self.calls]
        return sum(1 for a, b in zip(names, names[1:]) if a != b)

    # -- the one entry point ----------------------------------------------
    def call(self, tool_name: str, args: dict[str, Any]) -> str:
        started = monotonic_ms()
        self.tool_idx += 1
        valid, reason = schemas.validate_args(tool_name, args)

        status, error_type, fid, result = "ok", "", "", ""
        match, sim = "", None
        if not valid:
            # A malformed call is a model/framework failure, not a tool failure:
            # counted separately because quantization is expected to move it.
            status, error_type = "error", "schema"
            result = redact(f"{impl.TOOL_ERROR_PREFIX} invalid arguments: {reason}")
        else:
            try:
                impl.sleep_replay(tool_name)
                result, hit = impl.execute(
                    tool_name, args, self.cfg, self.store, source=self.source
                )
                if hit is not None:
                    fid, match, sim = hit.fixture_id, hit.match, hit.similarity
                else:
                    match = "live"
            except impl.ToolFailure as exc:
                status, error_type = "error", "tool_failure"
                result = redact(f"{impl.TOOL_ERROR_PREFIX} {exc}")
            except Exception as exc:  # never let a framework handler see this
                status, error_type = "error", type(exc).__name__
                result = redact(f"{impl.TOOL_ERROR_PREFIX} {type(exc).__name__}: {exc}")

        record = ToolCallRecord(
            tool_call_id=new_id("tc_"),
            run_id=self.run_id,
            parent_call_id=self.parent_call_id,
            tool_idx=self.tool_idx,
            tool_name=tool_name,
            args_json=json.dumps(args, sort_keys=True),
            args_schema_valid=valid,
            latency_ms=monotonic_ms() - started,
            status=status,
            error_type=error_type,
            result_bytes=len(result.encode()),
            replay_fixture_id=fid,
            fixture_match=match,
            fixture_similarity=sim,
        )
        self.calls.append(record)
        self.writer.write(record)
        return result

    # -- adapter helpers ---------------------------------------------------
    def callable_for(self, tool_name: str) -> Callable[..., str]:
        """A plain keyword-argument function each framework can wrap."""

        def _fn(**kwargs: Any) -> str:
            return self.call(tool_name, kwargs)

        _fn.__name__ = tool_name
        _fn.__doc__ = schemas.BY_NAME[tool_name]["description"]
        return _fn

    def specs(self) -> list[dict[str, Any]]:
        return schemas.TOOL_SCHEMAS
