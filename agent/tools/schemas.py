"""Tool names, descriptions and JSON schemas: one source for all frameworks.

Every adapter builds its framework-native tool objects from these dicts, so the
tool_schema_hash is identical across frameworks by construction rather than by
careful copying. Descriptions are carried over from the reference agent with
the input format made explicit, since native function calling needs the
argument documented in the schema rather than in prose.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "retrieve",
        "description": (
            "Retrieve relevant passages from the local research-paper knowledge base. "
            "Use for questions about AI, machine learning, RAG, retrieval, attention, "
            "neural networks, or the contents of the indexed papers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A natural-language search query describing the information needed.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the public web for information, news and current events. "
            "Use for anything not covered by the local knowledge base."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to send to the web search engine.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fred_releases",
        "description": (
            "List scheduled US macroeconomic data releases from the FRED calendar, "
            "such as CPI, GDP, FOMC statements and jobless claims."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "window": {
                    "type": "string",
                    "enum": ["this_week", "next_week"],
                    "description": "Which week to list releases for, relative to today's date.",
                }
            },
            "required": ["window"],
            "additionalProperties": False,
        },
    },
]

# Tools for open-ended agent tasks (GAIA). Offered only to a run whose config
# lists them in `tools`, so the Phase 1 task sets keep exactly the three tools
# above and their tool_schema_hash is unchanged.
EXTRA_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "fetch_url",
        "description": (
            "Fetch a web page and return its readable text (scripts, styles and markup "
            "removed). Use it to read a page found with web_search, or a URL named in the task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full http(s) URL of the page to read.",
                }
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file attached to the task and return its contents as text. Spreadsheets "
            "are returned sheet by sheet with cell values (and cell fill colours where set); "
            "PDFs as text; zip archives as a list of members with the contents of the readable "
            "ones. Images and audio cannot be read."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_name": {
                    "type": "string",
                    "description": "The attached file's name exactly as given in the task.",
                }
            },
            "required": ["file_name"],
            "additionalProperties": False,
        },
    },
]

DEFAULT_TOOLS = tuple(t["name"] for t in TOOL_SCHEMAS)
TOOL_NAMES = [t["name"] for t in TOOL_SCHEMAS]
ALL_SCHEMAS = TOOL_SCHEMAS + EXTRA_TOOL_SCHEMAS
BY_NAME = {t["name"]: t for t in ALL_SCHEMAS}
ALL_TOOL_NAMES = [t["name"] for t in ALL_SCHEMAS]


def tool_specs(names: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    """The schemas a run offers, in the order its config lists them."""
    if not names:
        return TOOL_SCHEMAS
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        raise ValueError(f"unknown tools {unknown}; available: {ALL_TOOL_NAMES}")
    return [BY_NAME[n] for n in names]


def tool_schema_hash(names: tuple[str, ...] | list[str] | None = None) -> str:
    """Parity evidence for row 2/3: identical in every framework.

    With the default tools this hashes exactly what it always did, so Phase 1
    config_ids are unchanged.
    """
    blob = json.dumps(tool_specs(names), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def openai_tools(names: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    """OpenAI function-calling wire format, for the proxy and for adapters."""
    return [{"type": "function", "function": dict(t)} for t in tool_specs(names)]


def validate_args(name: str, args: dict[str, Any]) -> tuple[bool, str]:
    """Minimal schema check. Returns (ok, reason).

    Deliberately not jsonschema: we need to distinguish a malformed tool call
    from a tool that errored, and the schemas here are small enough that a
    hand-rolled check keeps the dependency surface (and the failure modes)
    obvious.
    """
    spec = BY_NAME.get(name)
    if spec is None:
        return False, f"unknown tool '{name}'"
    params = spec["parameters"]
    props: dict[str, Any] = params["properties"]

    for key in params.get("required", []):
        if key not in args:
            return False, f"missing required argument '{key}'"
    for key, value in args.items():
        if key not in props:
            return False, f"unexpected argument '{key}'"
        prop = props[key]
        if prop["type"] == "string" and not isinstance(value, str):
            return False, f"argument '{key}' must be a string"
        if "enum" in prop and value not in prop["enum"]:
            return False, f"argument '{key}' must be one of {prop['enum']}"
    return True, ""
