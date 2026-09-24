"""Load .env into the process environment.

Deliberately dependency-free and forgiving about format: `.env` files get
hand-edited, and a key that silently fails to load shows up much later as a
tool error in the middle of a benchmark run. Accepts `KEY=value` and
`KEY: value`, strips quotes and whitespace, ignores comments.

Existing environment variables always win, so an explicit export in the shell
overrides the file.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Keys the tools need. Reported by require_keys() so a run fails at the start
# rather than at the first tool call.
TOOL_KEYS = {
    "TAVILY_API_KEY": "web_search",
    "FRED_API_KEY": "fred_releases",
}


def load_env(path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    """Read a .env file into os.environ. Returns the names that were set."""
    env_path = Path(path) if path else REPO_ROOT / ".env"
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]

        # Split on whichever separator comes first, so both styles work.
        eq, colon = line.find("="), line.find(":")
        if eq == -1 and colon == -1:
            continue
        sep = eq if (colon == -1 or (eq != -1 and eq < colon)) else colon

        key = line[:sep].strip()
        value = line[sep + 1 :].strip().strip("\"'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def require_keys(*names: str) -> list[str]:
    """Return the names that are still missing after loading."""
    return [n for n in names if not os.environ.get(n)]


def redact(text: str) -> str:
    """Strip secret values out of text before it is logged or stored.

    Tool errors travel into telemetry, fixtures and the console. An API key in
    a failing request URL would end up committed with the results, so every
    error string is scrubbed at the one place they are all produced.
    """
    for key in TOOL_KEYS:
        value = os.environ.get(key)
        if value and len(value) >= 8:
            text = text.replace(value, f"<{key}>")
    return text


def report(loaded: dict[str, str]) -> None:
    """Print which tool keys are present. Never prints a value."""
    for key, tool in sorted(TOOL_KEYS.items()):
        present = bool(os.environ.get(key))
        mark = "ok     " if present else "MISSING"
        print(f"  {mark} {key:16s} (needed by {tool})")
