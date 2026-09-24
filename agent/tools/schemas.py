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

TOOL_NAMES = [t["name"] for t in TOOL_SCHEMAS]
BY_NAME = {t["name"]: t for t in TOOL_SCHEMAS}


def tool_schema_hash() -> str:
    """Parity evidence for row 2/3: identical in every framework."""
    blob = json.dumps(TOOL_SCHEMAS, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def openai_tools() -> list[dict[str, Any]]:
    """OpenAI function-calling wire format, for the proxy and for adapters."""
    return [{"type": "function", "function": dict(t)} for t in TOOL_SCHEMAS]


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
