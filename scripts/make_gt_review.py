"""Build the ground-truth review sheet for a task set: one row per task.

Automated checks can show that each reference answer passes its own check and
that no question passes on its own; they cannot show that the reference answer
is right. This sheet is for the human sign-off: it puts the question, the
reference answer, what the check accepts, the judge rubric and the evidence
pointers side by side, with empty reviewer / approved / notes columns.

    python scripts/make_gt_review.py --tasks dataset/tasks/phase1_v2.json \\
        --out dataset/reviews/phase1_v2_gt_review.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.scoring import score, trivial_facts  # noqa: E402


def check_summary(check: dict) -> str:
    mode, params = check.get("mode"), check.get("params", {})
    if mode == "key_facts":
        facts = [f if isinstance(f, str) else " | ".join(f) for f in params.get("key_facts", [])]
        return f"key_facts (need {params.get('min_hits', len(facts))}): " + "; ".join(facts)
    if mode == "numeric_tolerance":
        return f"number {params.get('value')} ± {params.get('tolerance', 0)}"
    if mode == "structured":
        return "structured JSON contract: " + json.dumps(params.get("contract", {}))[:1500]
    if mode == "abstain":
        return "must decline; forbid: " + "; ".join(params.get("forbid", []))
    if mode == "judge_only":
        return "semantic judge"
    return mode or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="dataset/tasks/phase1_v2.json")
    ap.add_argument("--out", default="dataset/reviews/phase1_v2_gt_review.csv")
    args = ap.parse_args()

    tasks = json.loads((ROOT / args.tasks).read_text())
    cols = ["task_id", "difficulty", "task_type", "check_mode", "question", "reference_answer",
            "check", "judge_rubric", "judge_threshold", "evidence", "self_test",
            "reviewer", "approved", "notes"]
    problems = 0
    with open(ROOT / args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for t in tasks:
            check = t["answer_check"]
            if check.get("mode") == "judge_only":
                self_test = "n/a (judge)"
            else:
                ref_ok = score(t.get("reference_answer", ""), check).success
                q_passes = score(t["question"], check).success
                leaked = trivial_facts(t["question"], check)
                issues = ([] if ref_ok else ["reference fails its check"]) + \
                         (["question alone passes"] if q_passes else []) + \
                         ([f"facts in question: {leaked}"] if leaked else [])
                self_test = "; ".join(issues) or "ok"
                problems += bool(issues)
            evidence = t.get("evidence") or t.get("reference_context") or ""
            w.writerow({
                "task_id": t["task_id"], "difficulty": t.get("difficulty", ""),
                "task_type": t.get("task_type", ""), "check_mode": check.get("mode", ""),
                "question": t["question"], "reference_answer": t.get("reference_answer", ""),
                "check": check_summary(check),
                "judge_rubric": " / ".join(t.get("grading_rubric") or []) if check.get("mode") == "judge_only" else "",
                "judge_threshold": (t.get("evaluation") or {}).get("semantic_threshold", "")
                if check.get("mode") == "judge_only" else "",
                "evidence": evidence if isinstance(evidence, str) else json.dumps(evidence, ensure_ascii=False)[:2000],
                "self_test": self_test, "reviewer": "", "approved": "", "notes": "",
            })
    print(f"wrote {args.out}: {len(tasks)} tasks, {problems} with automated-check problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
