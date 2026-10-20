"""Make GAIA_stack/tasks/phase1-GAIA.json runnable with the GAIA tools.

  * answer_check -> gaia_exact (GAIA's official quasi-exact match against the
    reference answer); the rubric stays for an optional judge pass
  * answer_format -> gaia_final_answer: the task asks for a "FINAL ANSWER:" line
  * acceptable_tool_sets -> every route over web_search, fetch_url, read_file
    (GAIA does not prescribe tools)
  * attachments: file_path under GAIA_stack/documents/, and attachment_readable
    false for images and audio, which the harness cannot read
  * writes phase1-GAIA-text.json: the tasks a text-only harness can attempt
    (no attachment, or a readable one)

    .venv/bin/python GAIA_stack/scripts/prepare_gaia.py   (fetch_gaia.py runs it too)
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

STACK = Path(__file__).resolve().parent.parent
ROOT = STACK.parent
sys.path.insert(0, str(ROOT))

from agent.tools.impl import UNREADABLE  # noqa: E402

SRC = STACK / "tasks/phase1-GAIA.json"
TEXT = STACK / "tasks/phase1-GAIA-text.json"
DOCS = STACK / "documents"
TOOLS = ["web_search", "fetch_url", "read_file"]


def main() -> int:
    items = json.loads(SRC.read_text())
    routes = [list(c) for n in range(len(TOOLS) + 1) for c in itertools.combinations(TOOLS, n)]
    for t in items:
        t["answer_check"] = {"mode": "gaia_exact", "params": {"answer": str(t["reference_answer"])}}
        t["answer_format"] = "gaia_final_answer"
        t["acceptable_tool_sets"] = routes
        if t.get("file_name"):
            t["file_path"] = str((DOCS / t["file_name"]).relative_to(ROOT))
            suffix = Path(t["file_name"]).suffix.lower()
            t["attachment_readable"] = suffix not in UNREADABLE
            t["notes"] = ("Attachment in GAIA_stack/documents/, opened with the read_file tool."
                          if t["attachment_readable"] else
                          f"Attachment is {UNREADABLE[suffix]}; the harness has no {UNREADABLE[suffix]} "
                          "understanding, so this task is excluded from phase1-GAIA-text.json.")
            if not (ROOT / t["file_path"]).exists():
                raise SystemExit(f"missing attachment {t['file_path']}")
    SRC.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
    text = [t for t in items if t.get("attachment_readable", True)]
    TEXT.write_text(json.dumps(text, indent=2, ensure_ascii=False) + "\n")
    levels = {}
    for t in text:
        levels[t["difficulty"]] = levels.get(t["difficulty"], 0) + 1
    print(f"updated {SRC.name}: {len(items)} tasks; wrote {TEXT.name}: {len(text)} tasks {levels} "
          f"({len(items) - len(text)} image/audio tasks left out)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
