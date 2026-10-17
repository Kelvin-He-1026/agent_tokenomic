"""Record-once, replay-forever storage for non-deterministic tools.

The hard case is free-text search. A fixture keyed on the exact argument only
replays when the model issues a byte-identical query, and different models word
searches differently — so a miss would look like a model or framework
difference when it is really a gap in the fixture set.

The lookup therefore degrades in stages, and **records which stage matched**:

    exact       arguments identical             trustworthy
    normalized  same tokens, different wording  trustworthy
    nearest     token overlap above threshold   approximate, must be reported
    miss        nothing close enough            becomes a tool error

`fixture_match` and `fixture_similarity` land in every tool_call row, so an
analysis can exclude approximate matches, and the paper can state what share of
tool calls were served exactly. Hiding a fuzzy match behind a clean result is
what this design is built to prevent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Only free-text arguments are normalized. An enum like fred_releases.window
# has no near-misses: a different value is a different question.
FREE_TEXT_ARGS = {"query"}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to", "what",
    "when", "which", "who", "why", "with",
}

DEFAULT_THRESHOLD = 0.6


def fixture_id(tool_name: str, args: dict[str, Any]) -> str:
    blob = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords, strip plurals.

    The plural strip is crude ("bias" -> "bia") and deliberately so: both sides
    of a comparison get the same transform, so consistency matters more than
    linguistic correctness. Without it, "transformer" and "transformers" score
    as different tokens and an identical question can fall below threshold.
    """
    words = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    out = set()
    for w in words:
        if w in STOPWORDS or len(w) <= 1:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


def normalized_key(tool_name: str, args: dict[str, Any]) -> str:
    """Canonical form: same tokens in any order collapse to one key."""
    parts: list[str] = [tool_name]
    for key in sorted(args):
        value = args[key]
        if key in FREE_TEXT_ARGS and isinstance(value, str):
            parts.append(f"{key}={' '.join(sorted(tokenize(value)))}")
        else:
            parts.append(f"{key}={value}")
    return "|".join(parts)


def similarity(a: set[str], b: set[str]) -> float:
    """Jaccard overlap; 1.0 when the token sets are identical."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class FixtureHit:
    result: str
    fixture_id: str
    match: str  # exact | normalized | nearest
    similarity: float = 1.0
    matched_args: dict[str, Any] | None = None


@dataclass
class FixtureStore:
    root: Path
    threshold: float = DEFAULT_THRESHOLD

    def path(self, tool_name: str, fid: str) -> Path:
        return self.root / tool_name / f"{fid}.json"

    def _all(self, tool_name: str) -> list[dict[str, Any]]:
        d = self.root / tool_name
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except json.JSONDecodeError:
                continue
        return out

    def load(self, tool_name: str, args: dict[str, Any]) -> FixtureHit | None:
        fid = fixture_id(tool_name, args)
        exact = self.path(tool_name, fid)
        if exact.exists():
            data = json.loads(exact.read_text())
            return FixtureHit(data["result"], fid, "exact", 1.0, data.get("args"))

        entries = self._all(tool_name)
        if not entries:
            return None

        wanted_key = normalized_key(tool_name, args)
        for entry in entries:
            if normalized_key(tool_name, entry["args"]) == wanted_key:
                return FixtureHit(
                    entry["result"], entry["fixture_id"], "normalized", 1.0, entry["args"]
                )

        # Nearest, over free-text arguments only, and only when the non-text
        # arguments match exactly — a different enum is a different question.
        best, best_score = None, 0.0
        wanted_tokens = self._text_tokens(args)
        if wanted_tokens:
            for entry in entries:
                if self._fixed_args(entry["args"]) != self._fixed_args(args):
                    continue
                score = similarity(wanted_tokens, self._text_tokens(entry["args"]))
                if score > best_score:
                    best, best_score = entry, score

        if best is not None and best_score >= self.threshold:
            return FixtureHit(
                best["result"], best["fixture_id"], "nearest", best_score, best["args"]
            )
        return None

    @staticmethod
    def _text_tokens(args: dict[str, Any]) -> set[str]:
        tokens: set[str] = set()
        for key, value in args.items():
            if key in FREE_TEXT_ARGS and isinstance(value, str):
                tokens |= tokenize(value)
        return tokens

    @staticmethod
    def _fixed_args(args: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in args.items() if k not in FREE_TEXT_ARGS}

    def save(self, tool_name: str, args: dict[str, Any], result: str, source: str = "") -> str:
        fid = fixture_id(tool_name, args)
        p = self.path(tool_name, fid)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Written to a temporary file and renamed, so a parallel recorder never
        # reads half a fixture.
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "fixture_id": fid,
                    "tool": tool_name,
                    "args": args,
                    "normalized_key": normalized_key(tool_name, args),
                    "result": result,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "recorded_by": source,  # which model's probe produced this query
                },
                indent=2,
            )
        )
        tmp.replace(p)
        return fid

    def stats(self, tool_name: str) -> dict[str, Any]:
        entries = self._all(tool_name)
        return {
            "count": len(entries),
            "by_source": _count(e.get("recorded_by", "") for e in entries),
        }


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v or "unknown"] = out.get(v or "unknown", 0) + 1
    return out
