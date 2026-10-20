"""Where each LLM call's prompt tokens come from.

Every request the proxy forwards is saved as blobs/<run_id>/<idx>_request.json.
This splits its messages into four parts and counts each with the served
model's own tokenizer:

  system_prompt_tokens         system / developer messages
  conversation_history_tokens  earlier turns: assistant messages (text, any
                               reasoning sent back, tool-call names and
                               arguments) and earlier user messages
  tool_response_tokens         every tool result in the request
  current_user_prompt_tokens   the latest user message (the question, or a
                               framework's own follow-up such as CrewAI's
                               forced-final request)

The four count message text only. The model's chat template also adds role
markers and the tool definitions, so they sum to less than the server's
prompt_tokens; `other_prompt_tokens` is that remainder (template markup plus
tool schemas), so all five add up to prompt_tokens.

Counting happens offline, at export, from the saved requests: tokenising in the
proxy would block its event loop between calls and show up in the latency
being measured.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

CATEGORIES = (
    "system_prompt_tokens",
    "conversation_history_tokens",
    "tool_response_tokens",
    "current_user_prompt_tokens",
)
REMAINDER = "other_prompt_tokens"
COLUMNS = (*CATEGORIES, REMAINDER)

ROOT = Path(__file__).resolve().parent.parent


def message_text(message: dict[str, Any]) -> str:
    """All the text a message contributes, whatever shape its content has."""
    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):  # OpenAI content parts
        parts += [p.get("text", "") for p in content if isinstance(p, dict)]
    for key in ("reasoning_content", "reasoning"):
        if isinstance(message.get(key), str):
            parts.append(message[key])
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        parts += [fn.get("name", ""), fn.get("arguments", "") or ""]
    return "\n".join(p for p in parts if p)


def split(messages: list[dict[str, Any]], count: Callable[[str], int]) -> dict[str, int]:
    """Token counts per category for one request's messages."""
    out = dict.fromkeys(CATEGORIES, 0)
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    for i, m in enumerate(messages):
        role = m.get("role")
        if role in ("system", "developer"):
            key = "system_prompt_tokens"
        elif role == "tool":
            key = "tool_response_tokens"
        elif i == last_user:
            key = "current_user_prompt_tokens"
        else:
            key = "conversation_history_tokens"
        out[key] += count(message_text(m))
    return out


@lru_cache(maxsize=8)
def _tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def counter_for(model_repo: str) -> Callable[[str], int] | None:
    """A token counter for a run's model, from its local weights folder.

    Local folders are named after the repo's last part (model/Qwen3.8-27B for
    Qwen/Qwen3.8-27B). None if the folder or tokenizer is missing.
    """
    if not model_repo:
        return None
    model_dir = ROOT / "model" / model_repo.split("/")[-1]
    if not (model_dir / "tokenizer_config.json").exists():
        return None
    tok = _tokenizer(str(model_dir))

    @lru_cache(maxsize=65536)  # history repeats in every later call of a run
    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False)) if text else 0

    return count


def for_call(call: dict[str, Any], run_dir: Path, count: Callable[[str], int] | None) -> dict[str, Any]:
    """The five columns for one llm_call row; blanks when they cannot be counted."""
    blank = dict.fromkeys(COLUMNS, "")
    if count is None:
        return blank
    path = run_dir / "blobs" / call["run_id"] / f"{call['call_idx']:03d}_request.json"
    if not path.exists() and call.get("payload_path"):
        path = Path(call["payload_path"])
    if not path.exists():
        return blank
    try:
        messages = json.loads(path.read_text()).get("messages") or []
    except (OSError, json.JSONDecodeError):
        return blank
    parts = split(messages, count)
    prompt = call.get("prompt_tokens") or 0
    parts[REMAINDER] = prompt - sum(parts.values()) if prompt else ""
    return parts
