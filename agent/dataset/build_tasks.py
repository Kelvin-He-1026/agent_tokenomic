"""Build the Phase 1 task set from the reference ground truth plus authored items.

The reference Agentic RAG files carry question, expected_tool, reference_context
and reference_answer. Phase 1 additionally needs acceptable_tool_sets (a list of
sets, so a different valid route is not punished), ideal_llm_calls (the
denominator for Step Efficiency), and a deterministic answer_check.

Two things this script does and one it refuses to do:

- It merges the four reference files, dropping duplicate questions.
- It proposes key_facts for the deterministic check, from the reference answer.
- It does not treat those proposals as ground truth. Every item carries
  needs_review until a human confirms it, and `--report` shows what is left.

    python -m agent.dataset.build_tasks \\
        --reference ../agentic_ai/dataset/*.json \\
        --authored dataset/authored_tasks.json \\
        --out dataset/tasks/phase1.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

TOOL_ALIASES = {
    "retrieve": "retrieve",
    "rag": "retrieve",
    "vector_db": "retrieve",
    "vectordb": "retrieve",
    "web_search": "web_search",
    "tavily": "web_search",
    "web": "web_search",
    "fred_releases": "fred_releases",
    "fred": "fred_releases",
    "none": "",
}

STOPWORDS = {
    "The", "This", "That", "These", "Those", "It", "They", "There", "A", "An",
    "In", "On", "For", "With", "And", "But", "Its", "Their", "His", "Her",
}


def normalize_tools(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [t.strip() for t in raw.replace("+", ",").split(",") if t.strip()]
    out = [TOOL_ALIASES.get(str(t).lower().strip(), str(t).lower().strip()) for t in raw]
    return [t for t in out if t]


def propose_key_facts(answer: str, limit: int = 4) -> list[str]:
    """Candidate strings a correct answer should contain.

    Heuristic on purpose: quoted phrases, numbers with units, and capitalised
    multi-word terms are the parts of an answer that are hard to get right by
    accident. A human keeps, edits or replaces them.
    """
    if not answer:
        return []
    candidates: list[str] = []
    candidates += re.findall(r"[\"'“]([^\"'”]{3,40})[\"'”]", answer)
    candidates += re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|percent|billion|million|tokens?|layers?)\b", answer)
    for match in re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,})\b", answer):
        if match not in STOPWORDS and not match.split()[0] in STOPWORDS:
            candidates.append(match)

    seen: set[str] = set()
    out: list[str] = []
    for c in (c.strip() for c in candidates):
        key = c.lower()
        if c and key not in seen:
            seen.add(key)
            out.append(c)
    return out[:limit]


def convert_reference(item: dict[str, Any], idx: int, source: str) -> dict[str, Any]:
    tools = normalize_tools(item.get("expected_tool") or item.get("tool"))
    answer = item.get("reference_answer") or item.get("answer_reference") or ""
    context = item.get("reference_context") or item.get("context_reference") or ""

    if not tools:
        task_type = "no_tool"
    elif len(tools) > 1:
        task_type = "multi_tool"
    else:
        task_type = "single_tool_lookup"

    raw_id = item.get("id", idx + 1)
    facts = propose_key_facts(answer)
    return {
        "task_id": f"{Path(source).stem}-{raw_id:03d}" if isinstance(raw_id, int) else f"{Path(source).stem}-{raw_id}",
        "question": item["question"],
        "task_type": task_type,
        "difficulty": item.get("difficulty", "medium"),
        "reference_answer": answer,
        "reference_context": context,
        "answer_check": {
            "mode": "key_facts",
            "params": {"key_facts": facts, "min_hits": max(1, len(facts) // 2)},
        },
        "acceptable_tool_sets": [tools] if tools else [[]],
        "expected_args": {},
        "ideal_llm_calls": (len(tools) + 1) if tools else 1,
        "required_no_tool": not tools,
        "replay_fixture_id": "",
        "source": source,
        "notes": (item.get("metadata") or {}).get("title", "") if isinstance(item.get("metadata"), dict) else "",
        "key_facts_proposed": bool(facts),
        "needs_review": True,
    }


def normalize_authored(item: dict[str, Any]) -> dict[str, Any]:
    out = {
        "task_id": item["task_id"],
        "question": item["question"],
        "task_type": item["task_type"],
        "difficulty": item.get("difficulty", "medium"),
        "reference_answer": item.get("reference_answer", ""),
        "reference_context": "",
        "answer_check": item["answer_check"],
        "acceptable_tool_sets": item.get("acceptable_tool_sets", [[]]),
        "expected_args": item.get("expected_args", {}),
        "ideal_llm_calls": item.get("ideal_llm_calls", 1),
        "required_no_tool": item.get("required_no_tool", False),
        "replay_fixture_id": item.get("replay_fixture_id", ""),
        "source": "authored",
        "notes": item.get("notes", ""),
        "key_facts_proposed": False,
        "needs_review": True,
    }
    return out


def apply_reviewed(tasks: list[dict[str, Any]], path: str) -> None:
    """Overlay reviewed answer checks, then refuse any fact the question gives away.

    A key fact already present in the question is satisfied by any answer that
    restates the question, so it measures nothing. That is exactly how the
    first-pass proposals failed ("ACE", "RAG", "HOTPOTQA"), so the builder now
    rejects them unless a task opts in explicitly, with a note saying why.
    """
    from ..scoring import trivial_facts

    reviewed = json.loads(Path(path).read_text())
    meta = reviewed.get("_review", {})
    by_id = {t["task_id"]: t for t in tasks}
    unknown = [k for k in reviewed if not k.startswith("_") and k not in by_id]
    if unknown:
        raise SystemExit(f"reviewed checks for unknown tasks: {unknown}")

    for task_id, patch in reviewed.items():
        if task_id.startswith("_"):
            continue
        t = by_id[task_id]
        for key in ("reference_answer", "answer_check", "acceptable_tool_sets"):
            if key in patch:
                t[key] = patch[key]
        if patch.get("notes"):
            t["notes"] = patch["notes"]
        t["needs_review"] = False
        t["key_facts_proposed"] = False
        t["review"] = meta

    problems = []
    for t in tasks:
        params = t["answer_check"].get("params", {})
        leaked = trivial_facts(t["question"], t["answer_check"])
        if leaked and not params.get("allow_question_terms"):
            problems.append(f"{t['task_id']}: key fact(s) {leaked} appear in the question")
    if problems:
        raise SystemExit("trivial key facts:\n  " + "\n  ".join(problems))


def report(tasks: list[dict[str, Any]]) -> None:
    by_type = Counter(t["task_type"] for t in tasks)
    by_source = Counter("authored" if t["source"] == "authored" else "reference" for t in tasks)
    total = len(tasks)

    print(f"\n{total} tasks")
    print("\n  by type" + " " * 18 + "count   share   target")
    targets = {
        "single_tool_lookup": 0.30,
        "multi_hop": 0.20,
        "multi_tool": 0.25,
        "no_tool": 0.15,
        "tool_error": 0.10,
        "unanswerable": 0.10,
    }
    for name in sorted(set(by_type) | set(targets)):
        n = by_type.get(name, 0)
        tgt = targets.get(name)
        print(f"  {name:24s} {n:5d}  {n / total:6.0%}  {f'{tgt:.0%}' if tgt else '-'}")
    print(f"\n  by source: {dict(by_source)}")

    modes = Counter(t["answer_check"]["mode"] for t in tasks)
    print(f"\n  answer checks: {dict(modes)}")
    missing_facts = [
        t["task_id"]
        for t in tasks
        if t["answer_check"]["mode"] == "key_facts"
        and not t["answer_check"]["params"].get("key_facts")
    ]
    review_needed = [t["task_id"] for t in tasks if t["needs_review"]]
    print(f"\n  blocking: {len(missing_facts)} task(s) have no key_facts, so success cannot be scored")
    if missing_facts:
        print("    " + ", ".join(missing_facts[:8]) + (" ..." if len(missing_facts) > 8 else ""))
    print(f"  review:   {len(review_needed)} task(s) still marked needs_review")
    print("\n  Proposed key_facts are candidates, not ground truth. Confirm each one,")
    print("  then clear needs_review before the pilot runs.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", nargs="*", default=[])
    ap.add_argument("--authored", default="")
    ap.add_argument("--reviewed", default="",
                    help="reviewed answer checks to overlay (dataset/reviewed_checks.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks: list[dict[str, Any]] = []
    seen_questions: dict[str, str] = {}

    for path in args.reference:
        items = json.loads(Path(path).read_text())
        for i, item in enumerate(items):
            if "question" not in item:
                continue
            key = re.sub(r"\W+", " ", item["question"].lower()).strip()
            if key in seen_questions:
                continue
            seen_questions[key] = path
            tasks.append(convert_reference(item, i, path))

    if args.authored:
        for item in json.loads(Path(args.authored).read_text()):
            tasks.append(normalize_authored(item))

    if args.reviewed:
        apply_reviewed(tasks, args.reviewed)

    ids = Counter(t["task_id"] for t in tasks)
    dupes = [k for k, v in ids.items() if v > 1]
    if dupes:
        raise SystemExit(f"duplicate task_ids: {dupes}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, indent=2))
    print(f"wrote {out_path}")
    report(tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
