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


# Appended to a task's question (not the system prompt, which every framework
# must send unchanged) when the task asks for GAIA's answer format. Wording
# follows the GAIA paper's reference prompt.
GAIA_ANSWER_FORMAT = (
    "Report your reasoning, and finish your answer with the following template: "
    "FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words "
    "as possible OR a comma separated list of numbers and/or strings. If you are asked for a "
    "number, don't use commas to write your number, nor units such as $ or percent signs, "
    "unless specified otherwise. If you are asked for a string, don't use articles or "
    "abbreviations (e.g. for cities), and write digits in plain text unless specified "
    "otherwise. If you are asked for a comma separated list, apply the above rules to each "
    "element depending on whether it is a number or a string."
)


def task_prompt(item: dict) -> str:
    """The question a run sends: the task's question, plus its attachment notice
    and answer-format instructions when the task carries them."""
    question = item["question"]
    if item.get("requires_attachment") and item.get("file_name"):
        question += (f"\n\nAttached file: {item['file_name']}. "
                     "Open it with the read_file tool.")
    if item.get("answer_format") == "gaia_final_answer":
        question += "\n\n" + GAIA_ANSWER_FORMAT
    return question

