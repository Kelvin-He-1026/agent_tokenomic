"""Deterministic success: did the final answer contain what a correct one must?

These checks supply the deterministic part of agent.evaluation's unified
quality result. Twenty v2 tasks instead use a semantic correctness judge.
Five check modes, set per task in `answer_check`:

  key_facts          each fact is a string or a list of alternatives (any one
                     counts, so paraphrase passes); success = at least
                     `min_hits` facts present
  numeric_tolerance  some number in the answer is within `tolerance` of
                     `value`; matched on number boundaries, so 22500 is not 2250
                     (`single_number: true` requires a scalar-only answer)
  structured        a JSON answer matches all fields of a typed contract,
                     including numeric tolerances, aliases, exact sets and nulls.
                     Content and format are scored separately: a JSON object
                     surrounded by prose is still graded on its fields, and
                     `format_ok` records whether the reply was the bare object
                     (or one whole-answer code fence). `strict_format: true`
                     restores format-gated scoring
  abstain            for unanswerable and out-of-scope tasks: the answer
                     declines (an abstention phrase is present) and does not
                     fabricate (no `forbid` pattern matches)
  judge_only         semantic evaluation needed (e.g. current web answers or
                     open-ended explanations); left to the LLM judge

Any check may also carry `judge_if_tools`: a list of tools whose results come
from online sources. A run that called one of them is left to the judge like a
judge_only task, because what it saw is not fixed: an "unanswerable" question
is only unanswerable if the search really returned nothing relevant. Runs that
did not call those tools keep the deterministic check.

A run that did not complete (step_limit, truncated, ...) is never a success,
whatever its partial answer happens to contain.
"""

from __future__ import annotations

import json
import math
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
    # Independent format compliance; None where no format check applies.
    format_ok: bool | None = None
    semantic_correct: bool | None = None


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


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _unfence(answer: str) -> str:
    """Allow one whole-answer code fence, never extract JSON from prose."""
    stripped = answer.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n\s*```", stripped, re.S | re.I)
    return match.group(1).strip() if match else stripped


def _string_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_contract(spec: dict[str, Any]) -> None:
    """Reject invalid authored checks instead of blaming the model's answer.

    This is a small answer-contract language, not a general JSON Schema engine.
    Objects have exact fields; arrays have exact element counts. Set-like arrays
    opt into unordered matching. All scalar leaves have authored gold values.
    """
    if not isinstance(spec, dict):
        raise ValueError("structured contract nodes must be objects")
    kind = spec.get("type")
    if kind == "object":
        fields = spec.get("fields")
        if not isinstance(fields, dict) or not fields:
            raise ValueError("structured object needs nonempty fields")
        for child in fields.values():
            _validate_contract(child)
    elif kind == "array":
        if not isinstance(spec.get("elements"), list):
            raise ValueError("structured array needs elements")
        if not isinstance(spec.get("unordered", False), bool):
            raise ValueError("structured array unordered must be boolean")
        for child in spec["elements"]:
            _validate_contract(child)
    elif kind == "number":
        for key, default in (("value", None), ("tolerance", 0)):
            value = spec.get(key, default)
            if not _finite_number(value):
                raise ValueError(f"structured number needs finite {key}")
        if spec.get("tolerance", 0) < 0:
            raise ValueError("structured number tolerance must be nonnegative")
        if not isinstance(spec.get("integer", False), bool):
            raise ValueError("structured integer flag must be boolean")
        if spec.get("integer", False) and spec["value"] != int(spec["value"]):
            raise ValueError("structured integer gold must be integral")
    elif kind == "string":
        if not isinstance(spec.get("aliases", []), list):
            raise ValueError("structured aliases must be an array")
        values = [spec.get("value"), *spec.get("aliases", [])]
        if not all(isinstance(v, str) and v.strip() for v in values):
            raise ValueError("structured string needs nonempty value/aliases")
    elif kind == "boolean":
        if not isinstance(spec.get("value"), bool):
            raise ValueError("structured boolean needs boolean value")
    elif kind != "null":
        raise ValueError(f"unknown structured type: {kind}")


def _contract_error(value: Any, spec: dict[str, Any], path: str = "$", *, content_only: bool = False) -> str | None:
    kind = spec["type"]
    if kind == "object":
        if not isinstance(value, dict):
            return f"{path}: expected object"
        fields = spec["fields"]
        missing, extra = set(fields) - set(value), set(value) - set(fields)
        if missing:
            return f"{path}: missing fields {', '.join(sorted(missing))}"
        if extra and not content_only:
            return f"{path}: unexpected fields {', '.join(sorted(extra))}"
        for key, child in fields.items():
            error = _contract_error(value[key], child, f"{path}.{key}", content_only=content_only)
            if error:
                return error
    elif kind == "array":
        if not isinstance(value, list):
            return f"{path}: expected array"
        elements = spec["elements"]
        if len(value) != len(elements):
            return f"{path}: expected {len(elements)} elements, found {len(value)}"
        if spec.get("unordered", False):
            # Find a one-to-one matching. Greedy matching would reject valid
            # sets when aliases/tolerance intervals overlap.
            edges = [[j for j, child in enumerate(elements)
                      if _contract_error(item, child, content_only=content_only) is None] for item in value]
            assigned: dict[int, int] = {}

            def assign(i: int, seen: set[int]) -> bool:
                for j in edges[i]:
                    if j in seen:
                        continue
                    seen.add(j)
                    if j not in assigned or assign(assigned[j], seen):
                        assigned[j] = i
                        return True
                return False

            if not all(assign(i, set()) for i in range(len(value))):
                return f"{path}: set members differ (including duplicates or unsupported members)"
        else:
            for i, (item, child) in enumerate(zip(value, elements)):
                error = _contract_error(item, child, f"{path}[{i}]", content_only=content_only)
                if error:
                    return error
    elif kind == "number":
        if content_only and isinstance(value, str) and re.fullmatch(NUMERIC_SCALAR, value.strip()):
            value = float(value.strip().replace(",", ""))
        if not _finite_number(value):
            return f"{path}: expected finite number"
        if spec.get("integer", False) and value != int(value):
            return f"{path}: expected integer"
        if not math.isclose(value, spec["value"], rel_tol=0, abs_tol=spec.get("tolerance", 0)):
            return f"{path}: number outside tolerance"
    elif kind == "string":
        if not isinstance(value, str):
            return f"{path}: expected string"
        if _string_key(value) not in {_string_key(v) for v in [spec["value"], *spec.get("aliases", [])]}:
            return f"{path}: incorrect value"
    elif kind == "boolean":
        if not isinstance(value, bool) or value is not spec["value"]:
            return f"{path}: incorrect boolean"
    elif kind == "null" and value is not None:
        return f"{path}: expected null for unavailable value"
    return None


def _load_json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _embedded_object(answer: str) -> Any:
    """The last JSON object that parses anywhere in the answer, or None.

    The last one, because agents that explain first put their final answer
    last. Duplicate keys and non-finite numbers are rejected as in a bare
    answer, so leniency never admits a malformed object.
    """
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    found, i = None, answer.find("{")
    while i != -1:
        try:
            value, end = decoder.raw_decode(answer, i)
        except (ValueError, TypeError, RecursionError):
            # A nested fragment of a malformed/duplicate-key outer object is
            # not an independently stated answer. Do not salvage that fragment.
            return None
        if isinstance(value, dict):
            found = value
        i = answer.find("{", end)
    return found


def _structured_score(answer: str, params: dict[str, Any]) -> Score:
    contract = params.get("contract", {})
    _validate_contract(contract)
    try:
        value, format_ok, note = _load_json(_unfence(answer)), True, ""
    except (ValueError, TypeError, RecursionError) as exc:
        value = None if params.get("strict_format") else _embedded_object(answer)
        if value is None:
            return Score(False, "structured", f"invalid JSON answer: {exc}", format_ok=False)
        format_ok, note = False, "JSON taken from surrounding prose; "
    separate = params.get("separate_format", False)
    if separate:
        format_ok = format_ok and _shape_ok(value, contract)
    error = _contract_error(value, contract, content_only=separate)
    return Score(error is None, "structured", note + (error or "all structured fields correct"),
                 format_ok=format_ok, semantic_correct=error is None)


NUMERIC_SCALAR = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"


def _shape_ok(value: Any, spec: dict[str, Any]) -> bool:
    """Check public types and keys, without consulting gold values or set sizes.

    Public v2 contracts permit null at every scalar; whether null is correct
    belongs to content grading. Ordering and array membership do too.
    """
    kind = spec["type"]
    if kind == "object":
        return (isinstance(value, dict) and set(value) == set(spec["fields"])
                and all(_shape_ok(value[k], s) for k, s in spec["fields"].items()))
    if kind == "array":
        return isinstance(value, list) and all(
            any(_shape_ok(item, child) for child in spec["elements"])
            for item in value
        )
    if value is None:
        return True
    if kind == "number":
        return _finite_number(value) and (not spec.get("integer") or value == int(value))
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    # An unavailable gold value must not disclose which scalar type is missing.
    return isinstance(value, (str, bool)) or _finite_number(value)


def _separate_numeric(answer: str, params: dict[str, Any]) -> Score:
    scalar = _unfence(answer)
    format_ok = bool(re.fullmatch(NUMERIC_SCALAR, scalar))
    if not format_ok:
        # Recover only a unique standalone number. Never choose the gold out
        # of several candidates or repair ambiguous/malformed grouping.
        if re.search(r"\b(?:wrong|incorrect|not|isn't|isnt)\b", scalar, re.I):
            return Score(False, "numeric_tolerance", "no unambiguous numeric answer: negated claim", False)
        matches = list(re.finditer(r"(?<![\w.,])" + NUMERIC_SCALAR + r"(?![\w.,])", scalar))
        if len(matches) != 1:
            return Score(False, "numeric_tolerance", "no unambiguous numeric answer", False)
        scalar = matches[0].group()
    found = float(scalar.replace(",", ""))
    if not math.isfinite(found):
        return Score(False, "numeric_tolerance", "non-finite numeric answer", False, False)
    ok = math.isfinite(found) and abs(found - float(params["value"])) <= float(params.get("tolerance", 0))
    return Score(ok, "numeric_tolerance", f"found {found:g}; format checked separately", format_ok, ok)


# -- GAIA: the benchmark's official quasi-exact-match scorer ---------------------
# Follows GAIA's published question_scorer: numbers compare as floats after
# dropping $, % and thousands separators; comma/semicolon lists compare element
# by element (same length, same order); strings compare after lowercasing and
# removing whitespace and punctuation. The answer is the text after the last
# "FINAL ANSWER:" marker, which the task asks for.
FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*[:：]\s*(.+)", re.I)


def _gaia_number(text: str) -> float | None:
    cleaned = text.replace("$", "").replace("%", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _gaia_str(text: str, remove_punct: bool = True) -> str:
    text = re.sub(r"\s", "", text).lower()
    if remove_punct:
        text = re.sub(r"[^\w]", "", text)
    return text


def _gaia_split(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[,;]", text)]


def gaia_match(model_answer: str, ground_truth: str) -> bool:
    model_answer = model_answer.strip().strip("*").strip()
    if _gaia_number(ground_truth) is not None:
        value = _gaia_number(model_answer)
        return value is not None and value == _gaia_number(ground_truth)
    if any(c in ground_truth for c in ",;"):
        gt, ma = _gaia_split(ground_truth), _gaia_split(model_answer)
        if len(gt) != len(ma):
            return False
        for g, m in zip(gt, ma):
            if _gaia_number(g) is not None:
                if _gaia_number(m) is None or _gaia_number(m) != _gaia_number(g):
                    return False
            elif _gaia_str(m, remove_punct=False) != _gaia_str(g, remove_punct=False):
                return False
        return True
    return _gaia_str(model_answer) == _gaia_str(ground_truth)


def _gaia_score(answer: str, params: dict[str, Any]) -> Score:
    gold = str(params.get("answer", ""))
    found = FINAL_ANSWER_RE.findall(answer)
    if found:
        extracted, format_ok = found[-1].strip().splitlines()[0].strip(), True
    else:
        # No marker: grade the last non-empty line, and report the format miss.
        lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
        extracted, format_ok = (lines[-1] if lines else ""), False
    ok = bool(extracted) and gaia_match(extracted, gold)
    detail = f"extracted {extracted[:80]!r} vs gold {gold[:60]!r}" + ("" if format_ok else "; no FINAL ANSWER line")
    return Score(ok, "gaia_exact", detail, format_ok=format_ok, semantic_correct=ok)


def _score_content(
    answer: str,
    check: dict[str, Any],
    status: str = "completed",
    tools_used: list[str] | None = None,
) -> Score:
    mode = check.get("mode", "key_facts")
    params = check.get("params", {})

    if mode == "judge_only":
        return Score(None, mode, params.get("reason", "no stable ground truth"))
    online = sorted(set(params.get("judge_if_tools", [])) & set(tools_used or []))
    if online:
        return Score(None, "judge_only", f"called {', '.join(online)}: left to the judge")
    if status != "completed":
        return Score(False, mode, f"run did not complete ({status})")

    if mode == "gaia_exact":
        return _gaia_score(answer or "", params)
    if mode == "structured":
        return _structured_score(answer or "", params)

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
        if params.get("separate_format") and params.get("single_number"):
            return _separate_numeric(answer or "", params)
        value, tol = float(params["value"]), float(params.get("tolerance", 0))
        if params.get("single_number"):
            scalar = _unfence(answer or "")
            pattern = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
            if not re.fullmatch(pattern, scalar):
                return Score(False, mode, "expected only one numeric scalar")
            found = float(scalar.replace(",", ""))
            ok = math.isfinite(found) and abs(found - value) <= tol
            return Score(ok, mode, f"want {value:g}±{tol:g}, found {found:g}")
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


def score(answer: str, check: dict[str, Any], status: str = "completed",
          tools_used: list[str] | None = None) -> Score:
    """Assess content and format even when execution did not finish.

    ``success`` remains completion-gated for compatibility. An unparseable
    v2 answer has unknown semantic correctness, rather than a reasoning fail.
    No extra model judge is invoked for malformed deterministic answers.
    """
    result = _score_content(answer, check, "completed", tools_used)
    if result.semantic_correct is None and result.success is not None:
        if not (check.get("params", {}).get("separate_format") and result.format_ok is False
                and (result.detail.startswith("invalid JSON") or result.detail.startswith("no unambiguous"))):
            result.semantic_correct = result.success
    if status != "completed":
        result.success = False
        result.detail = f"run did not complete ({status}); " + result.detail
    return result


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
