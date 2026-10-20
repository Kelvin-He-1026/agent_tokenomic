"""Rebuild the GAIA task files and attachments from Hugging Face.

GAIA asks that its questions and answers not be reshared in a crawlable form,
so the repo commits only the task ids (GAIA_stack/tasks/gaia_task_ids.json:
25 per level, drawn from the 2023 validation split). This script downloads that
split at the pinned revision, rebuilds GAIA_stack/tasks/phase1-GAIA.json in the
same order, copies the attachments into GAIA_stack/documents/, and then runs
prepare_gaia.py (scoring, tool routes, phase1-GAIA-text.json).

The dataset is gated: accept its terms on huggingface.co/datasets/gaia-benchmark/GAIA
and log in once (`huggingface-cli login`) or set HF_TOKEN.

    .venv/bin/python GAIA_stack/scripts/fetch_gaia.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

STACK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STACK / "scripts"))

import prepare_gaia  # noqa: E402

REPO = "gaia-benchmark/GAIA"
REVISION = "682dd723ee1e1697e00360edccf2366dc8418dd9"
SPLIT = "2023/validation"
IDS = STACK / "tasks/gaia_task_ids.json"
DIFFICULTY = {1: "easy", 2: "medium", 3: "hard"}
RUBRIC = [
    "Compare the response with the official GAIA reference answer; accept equivalent wording.",
    "Require all parts requested by the question and reject unsupported or contradictory claims.",
]


def main() -> int:
    snap = Path(snapshot_download(REPO, repo_type="dataset", revision=REVISION,
                                  allow_patterns=[f"{SPLIT}/*"]))
    rows = {}
    for level in (1, 2, 3):
        name = f"metadata.level{level}.parquet"
        for r in pd.read_parquet(snap / SPLIT / name).to_dict("records"):
            rows[r["task_id"]] = (r, f"{REPO}:{SPLIT}/{name}")

    prepare_gaia.DOCS.mkdir(parents=True, exist_ok=True)
    items = []
    for gaia_id in json.loads(IDS.read_text())["task_ids"]:
        r, source = rows[gaia_id]
        t = {
            "task_id": f"gaia-{gaia_id}",
            "question": r["Question"],
            "task_type": "gaia_agent_task",
            "difficulty": DIFFICULTY[int(r["Level"])],
            "gaia_level": int(r["Level"]),
            "reference_answer": str(r["Final answer"]),
            "reference_context": "",
            "answer_check": {},  # filled by prepare_gaia
            "grading_rubric": RUBRIC,
            "acceptable_tool_sets": [],  # filled by prepare_gaia
            "source": source,
            "source_revision": REVISION,
            "source_split": SPLIT,
            "requires_attachment": bool(r["file_name"]),
        }
        # Placeholders keep the field order of the original file; prepare_gaia
        # fills them in place.
        if r["file_name"]:
            t.update(file_name=r["file_name"], file_path="", notes="", answer_format="",
                     attachment_readable=True)
            shutil.copyfile(snap / SPLIT / r["file_name"], prepare_gaia.DOCS / r["file_name"])
        else:
            t["answer_format"] = ""
        items.append(t)
    prepare_gaia.SRC.parent.mkdir(parents=True, exist_ok=True)
    prepare_gaia.SRC.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
    print(f"fetched {len(items)} tasks and {sum(t['requires_attachment'] for t in items)} attachments")
    return prepare_gaia.main()


if __name__ == "__main__":
    raise SystemExit(main())
