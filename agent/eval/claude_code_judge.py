"""A DeepEval judge model that runs on a Claude subscription via Claude Code.

DeepEval normally calls a judge through a provider API, which needs an API key
and separate billing. This wraps Claude Code's headless mode instead
(`claude --print`), so judging runs under the same subscription login as
Claude Code itself. It is the legitimate route: the subscription is used by
Anthropic's own CLI, not by extracting its token into another client.

Isolation matters, because a judge that carries a coding agent's context is not
a clean judge. Each call:

  --system-prompt           replaces Claude Code's system prompt with JUDGE_SYSTEM
  --tools ""                no tools
  --strict-mcp-config       no MCP connectors (Gmail, Drive, ...)
  --setting-sources ""      no user/project/local settings or hooks
  --disable-slash-commands  no skills
  --no-session-persistence  nothing written to the session history
  empty working directory   no CLAUDE.md or project memory
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
                            suppresses Claude Code's background model calls,
                            which otherwise add a second model (Haiku) per call

Measured residue: about 430 tokens of Claude Code preamble around each judge
prompt. `--bare` would remove it but cannot use a subscription login. For
numbers that go in a paper, prefer a direct API call to a pinned model; this
judge is for pilots and development, and every call is logged so the judge's
own cost is visible.
"""

from __future__ import annotations

import asyncio
import copy
import glob
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from deepeval.models import DeepEvalBaseLLM

JUDGE_SYSTEM = (
    "You are an impartial evaluation judge for an AI-agent benchmark. Follow the "
    "instructions in the user message exactly. When a JSON schema is required, return "
    "only data matching it. Base every judgement only on the material provided."
)
JUDGE_SYSTEM_HASH = hashlib.sha256(JUDGE_SYSTEM.encode()).hexdigest()[:16]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def find_claude_binary() -> str:
    """The Claude Code CLI: $ATK_CLAUDE_BIN, `claude` on PATH, or the newest
    copy bundled with the VS Code extension."""
    if os.environ.get("ATK_CLAUDE_BIN"):
        return os.environ["ATK_CLAUDE_BIN"]
    if shutil.which("claude"):
        return shutil.which("claude")  # type: ignore[return-value]
    bundled = sorted(glob.glob(os.path.expanduser(
        "~/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude"
    )) + glob.glob(os.path.expanduser(
        "~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"
    )))
    if not bundled:
        raise RuntimeError("Claude Code CLI not found; set ATK_CLAUDE_BIN")
    return bundled[-1]


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve pydantic's $defs/$ref into one self-contained schema."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


class JudgeError(RuntimeError):
    pass


class ClaudeCodeJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = "claude-opus-5",
        effort: str = "high",
        timeout_s: float = 300.0,
        retries: int = 3,
        call_log: str | Path | None = None,
    ):
        self.model_id = model
        self.effort = effort
        self.timeout_s = timeout_s
        self.retries = retries
        self.binary = find_claude_binary()
        self.version = self._version()
        self.cwd = REPO_ROOT / "output" / ".judge_cwd"  # empty on purpose
        self.cwd.mkdir(parents=True, exist_ok=True)
        self.call_log = Path(call_log) if call_log else None
        self._lock = threading.Lock()
        self.reset_stats()
        super().__init__(model=model)

    # -- per-metric accounting: the judge's own cost is part of the story --
    def reset_stats(self) -> None:
        self.stats = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}

    # -- DeepEvalBaseLLM interface -----------------------------------------
    def load_model(self, *args, **kwargs):
        return self

    def get_model_name(self, *args, **kwargs) -> str:
        return f"{self.model_id} via Claude Code {self.version} (headless, effort={self.effort})"

    def generate(self, prompt: str, schema: Any = None, **kwargs) -> Any:
        out = self._call(prompt, schema)
        if schema is None:
            return out["result"]
        data = out.get("structured_output")
        if data is None:  # fall back to parsing the text result
            data = json.loads(out["result"])
        return schema.model_validate(data)

    async def a_generate(self, prompt: str, schema: Any = None, **kwargs) -> Any:
        return await asyncio.to_thread(self.generate, prompt, schema)

    # -- the subprocess ------------------------------------------------------
    def _version(self) -> str:
        try:
            out = subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=30)
            return out.stdout.split()[0]
        except Exception:
            return "unknown"

    def _command(self, schema: Any) -> list[str]:
        cmd = [
            self.binary, "--print",
            "--model", self.model_id,
            "--effort", self.effort,
            "--system-prompt", JUDGE_SYSTEM,
            "--tools", "",
            "--strict-mcp-config",
            "--setting-sources", "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--output-format", "json",
        ]
        if schema is not None:
            cmd += ["--json-schema", json.dumps(_inline_refs(schema.model_json_schema()))]
        return cmd

    def _call(self, prompt: str, schema: Any) -> dict[str, Any]:
        env = {**os.environ, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
        # Never let a stray key redirect judging to API billing silently.
        env.pop("ANTHROPIC_API_KEY", None)
        last = ""
        for attempt in range(1, self.retries + 1):
            started = time.perf_counter()
            try:
                proc = subprocess.run(
                    self._command(schema), input=prompt, capture_output=True, text=True,
                    timeout=self.timeout_s, cwd=self.cwd, env=env,
                )
                out = json.loads(proc.stdout)
            except subprocess.TimeoutExpired:
                last = f"timeout after {self.timeout_s}s"
                out = None
            except json.JSONDecodeError:
                last = f"non-JSON output: {(proc.stdout or proc.stderr)[:200]}"
                out = None
            if out is not None and not out.get("is_error"):
                self._log(prompt, schema, out, time.perf_counter() - started, attempt)
                return out
            if out is not None:
                last = str(out.get("result"))[:200]
                if "not logged in" in last.lower():
                    raise JudgeError("Claude Code is not logged in; run /login in Claude Code")
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))  # usage windows: back off
        raise JudgeError(f"judge call failed after {self.retries} attempts: {last}")

    def _log(self, prompt: str, schema: Any, out: dict[str, Any], seconds: float, attempt: int) -> None:
        usage = out.get("modelUsage") or {}
        self.stats["calls"] += 1
        self.stats["seconds"] += seconds
        for v in usage.values():
            self.stats["input_tokens"] += (v.get("inputTokens", 0) + v.get("cacheReadInputTokens", 0)
                                           + v.get("cacheCreationInputTokens", 0))
            self.stats["output_tokens"] += v.get("outputTokens", 0)
        if self.call_log is None:
            return
        row = {
            "ts": time.time(),
            "judge": self.get_model_name(),
            "system_prompt_hash": JUDGE_SYSTEM_HASH,
            "schema": getattr(schema, "__name__", None),
            "prompt_chars": len(prompt),
            "seconds": round(seconds, 2),
            "attempt": attempt,
            "models": {
                m: {k: v.get(k, 0) for k in ("inputTokens", "outputTokens",
                                            "cacheReadInputTokens", "cacheCreationInputTokens")}
                for m, v in usage.items()
            },
        }
        with self._lock, open(self.call_log, "a") as fh:
            fh.write(json.dumps(row) + "\n")
