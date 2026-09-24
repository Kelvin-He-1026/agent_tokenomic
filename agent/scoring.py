"""Deterministic success: did the final answer contain what a correct one must?

This is the primary success measure; an LLM judge (DeepEval) is secondary and
reported alongside, so the headline number is never at the mercy of judge
noise. Four check modes, set per task in `answer_check`:

  key_facts          each fact is a string or a list of alternatives (any one
                     counts, so paraphrase passes); success = at least
                     `min_hits` facts present
  numeric_tolerance  some number in the answer is within `tolerance` of
                     `value`; matched on number boundaries, so 22500 is not 2250
  abstain            for unanswerable and out-of-scope tasks: the answer
                     declines (an abstention phrase is present) and does not
                     fabricate (no `forbid` pattern matches)
  judge_only         no stable ground truth (e.g. "the S&P 500 today"); not
                     scored here, left to the LLM judge

A run that did not complete (step_limit, truncated, ...) is never a success,
whatever its partial answer happens to contain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Phrases that signal the answer declines rather than invents. Kept specific:
# a bare "not" appears in almost any long answer, fabricated or not.
ABSTAIN_PHRASES = [
    "not available", "not mentioned", "not reported", "not provided", "not found",
    "not specified", "not stated", "not included", "not contain", "not include",
    "not report", "not mention", "not provide", "not specify", "does not appear",
    "do not appear", "no information",
    "no such", "no data", "no record", "no mention", "no figure", "no paper",
    "none of the", "cannot find", "could not find", "couldn't find",
    "cannot be found",
    "not present", "is unavailable", "no indexed paper",
    "does not exist", "not possible to",
    # Added after auditing the first full run, where each of these was a
    # correct refusal scored as a failure. Generic phrasings, not answer-specific.
    "not explicitly", "not directly", "not able to", "unable to", "not sure",
    "clarify", "not cover", "not comprehensive", "not sufficient", "insufficient",
    "not enough information", "beyond the scope", "out of scope", "outside the scope",
    "no way to",
]


@dataclass
class Score:
    success: bool | None  # None = not scored deterministically (judge_only)
    mode: str
    detail: str


def normalize(text: str) -> str:
    """Lowercase, flatten markdown and hyphenation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[*_`#>|]", " ", text)
    text = re.sub(r"[‐-―\-]", " ", text)  # hyphens and dashes
    text = text.replace("’", "'")
    # Expand contractions, so "doesn't report" matches "not report". Missing
    # this scored correct refusals as failures in the first full run.
    text = text.replace("can't", "cannot").replace("won't", "will not").replace("n't", " not")
    return re.sub(r"\s+", " ", text).strip()


def _alternatives(fact: Any) -> list[str]:
    return [normalize(a) for a in (fact if isinstance(fact, list) else [fact])]


def fact_hit(fact: Any, answer_norm: str) -> bool:
    return any(alt and alt in answer_norm for alt in _alternatives(fact))


def _numbers(text: str) -> list[float]:
    out = []
    for m in re.finditer(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])", text):
        try:
            out.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return out


def score(answer: str, check: dict[str, Any], status: str = "completed") -> Score:
    mode = check.get("mode", "key_facts")
    params = check.get("params", {})

    if mode == "judge_only":
        return Score(None, mode, params.get("reason", "no stable ground truth"))
    if status != "completed":
        return Score(False, mode, f"run did not complete ({status})")

    norm = normalize(answer or "")
    if mode == "key_facts":
        facts = params.get("key_facts") or []
        if not facts:
            return Score(None, mode, "no key facts defined")
        hits = [fact_hit(f, norm) for f in facts]
        need = params.get("min_hits", len(facts))
        missing = [_alternatives(f)[0] for f, h in zip(facts, hits) if not h]
        detail = f"{sum(hits)}/{len(facts)} facts (need {need})"
        if missing:
            detail += "; missing: " + ", ".join(missing[:4])
        return Score(sum(hits) >= need, mode, detail)

    if mode == "numeric_tolerance":
        value, tol = float(params["value"]), float(params.get("tolerance", 0))
        found = _numbers(answer or "")
        ok = any(abs(x - value) <= tol for x in found)
        return Score(ok, mode, f"want {value:g}±{tol:g}, found {found[:5]}")

    if mode == "abstain":
        # A task's own phrases extend the general refusal phrases, never
        # replace them: a task-specific list would otherwise lose "not able to"
        # and the like, and score a plain refusal as a failure.
        phrases = [normalize(p) for p in ABSTAIN_PHRASES + params.get("phrases", [])]
        declined = next((p for p in phrases if p in norm), None)
        fabricated = next(
            (pat for pat in params.get("forbid", []) if re.search(pat, answer or "", re.I)), None
        )
        if fabricated:
            return Score(False, mode, f"fabricated: matched /{fabricated}/")
        return Score(bool(declined), mode, f"declined via '{declined}'" if declined else "did not decline")

    raise ValueError(f"unknown answer_check mode '{mode}'")


def trivial_facts(question: str, check: dict[str, Any]) -> list[str]:
    """Key-fact alternatives already present in the question.

    Such a fact is satisfied by any answer that restates the question, so it
    says nothing about correctness. The task builder refuses them.
    """
    if check.get("mode") != "key_facts":
        return []
    q = normalize(question)
    return [
        alt for f in check.get("params", {}).get("key_facts", [])
        for alt in _alternatives(f) if alt and alt in q
    ]
