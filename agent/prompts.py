"""The single system prompt every framework must send, byte for byte.

Two deliberate differences from the reference implementation:

1. No Thought/Action/Action Input scaffolding. Phase 1 uses native function
   calling in all three frameworks (parity row 2), so the loop format is the
   model's tool-call channel, not parsed text.
2. The date is frozen to config.as_of_date instead of datetime.now(). A prompt
   that changes daily breaks prefix caching, reproducibility, and any FRED
   ground truth that depends on "this week".
"""

from __future__ import annotations

import hashlib

SYSTEM_PROMPT_TEMPLATE = """You are a research assistant that answers questions using the tools available to you.

Today's date is {as_of_date}.

Guidelines:
- Call a tool when you need information you do not already have.
- Call only one tool at a time, and wait for its result before deciding what to do next.
- Do not call a tool when you can already answer the question.
- If a tool returns an error, decide whether to try a different tool or answer without it. Do not repeat a failing call unchanged.
- When you have enough information, answer directly and concisely.
- Base your answer only on the tool results and the question. Do not invent sources, figures, or dates."""


def build_system_prompt(as_of_date: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(as_of_date=as_of_date)


def prompt_hash(prompt: str) -> str:
    """Parity evidence: all three frameworks must produce this same value."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]
