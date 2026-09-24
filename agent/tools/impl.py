"""The three tool implementations, shared by every framework.

Each framework wraps these exact functions, so tool behaviour and tool latency
are identical by construction. Live implementations are adapted from the
reference agent, with two changes required by the methodology:

- The FRED window is computed from the frozen as_of_date, not from today.
- Web and FRED results are recorded once and replayed from fixtures, so a
  framework comparison is not contaminated by API drift or network variance.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..config import RunConfig
from .fixtures import FixtureHit, FixtureStore, fixture_id  # re-exported for callers

# Watchlist of FRED release ids -> (label, tier, time ET), carried over from the
# reference agent so the tool output format stays comparable.
MACRO_WATCHLIST: dict[int, tuple[str, int, str]] = {
    10: ("Consumer Price Index", 1, "8:30 AM"),
    50: ("Employment Situation", 1, "8:30 AM"),
    53: ("Gross Domestic Product", 1, "8:30 AM"),
    54: ("Personal Income and Outlays", 1, "8:30 AM"),
    13: ("G.17 Industrial Production and Capacity Utilization", 2, "9:15 AM"),
    14: ("G.19 Consumer Credit", 3, "3:00 PM"),
    15: ("H.4.1 Factors Affecting Reserve Balances", 3, "4:30 PM"),
    46: ("Producer Price Index", 1, "8:30 AM"),
    97: ("Advance Monthly Sales for Retail and Food Services", 1, "8:30 AM"),
    175: ("Unemployment Insurance Weekly Claims Report", 2, "8:30 AM"),
    180: ("Manufacturer's Shipments, Inventories, and Orders", 2, "10:00 AM"),
    194: ("FOMC Press Release", 1, "2:00 PM"),
    229: ("Chicago Fed National Financial Conditions Index", 3, "8:30 AM"),
}
TIER_LABELS = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}

TOOL_ERROR_PREFIX = "ToolError:"


class ToolFailure(Exception):
    """Raised inside a tool; the registry turns it into a fixed error string."""


# ---------------------------------------------------------------------------
# Live implementations
# ---------------------------------------------------------------------------
def live_fred_releases(window: str, cfg: RunConfig) -> str:
    """Scheduled macro releases for the week relative to cfg.as_of_date."""
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        raise ToolFailure("FRED_API_KEY is not set")

    import requests

    today = datetime.strptime(cfg.as_of_date, "%Y-%m-%d").date()
    weekday = today.weekday()  # Mon=0
    if window == "next_week":
        start = today + timedelta(days=(7 - weekday))
        end = start + timedelta(days=4)
    else:
        start = today
        end = today + timedelta(days=max(4 - weekday, 0))

    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/releases/dates",
            params={
                "api_key": api_key,
                "file_type": "json",
                "realtime_start": start.isoformat(),
                "realtime_end": end.isoformat(),
                "include_release_dates_with_no_data": "true",
                "sort_order": "asc",
                "limit": 1000,
            },
            timeout=15,
        )
        resp.raise_for_status()
        releases = resp.json().get("release_dates", [])
    except Exception as exc:  # network, HTTP, JSON
        raise ToolFailure(f"FRED request failed: {exc}") from exc

    rows = []
    for r in releases:
        meta = MACRO_WATCHLIST.get(r.get("release_id"))
        if meta is None:
            continue
        label, tier, time_et = meta
        rows.append((r["date"], time_et, label, TIER_LABELS[tier]))
    if not rows:
        return f"No major macro releases scheduled between {start} and {end}."

    rows.sort(key=lambda row: (row[0], datetime.strptime(row[1], "%I:%M %p")))
    lines = [
        f"# Macro Releases: {start} to {end}",
        "",
        "| Date | Time (ET) | Release | Tier |",
        "|------|-----------|---------|------|",
    ]
    lines += [f"| {d} | {t} | {label} | {tier} |" for d, t, label, tier in rows]
    return "\n".join(lines)


def live_web_search(query: str, cfg: RunConfig) -> str:
    """Tavily web search, formatted as in the reference agent."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise ToolFailure("TAVILY_API_KEY is not set")

    try:
        from tavily import TavilyClient

        results = TavilyClient(api_key=api_key).search(
            query=query,
            search_depth="basic",
            topic="general",
            max_results=5,
            include_answer=True,
        )
    except Exception as exc:
        raise ToolFailure(f"web search failed: {exc}") from exc

    lines: list[str] = []
    if results.get("answer"):
        lines.append(f"Summary: {results['answer']}\n")
    lines.append("Sources:")
    for i, r in enumerate(results.get("results", []), 1):
        content = r.get("content", "")
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"\n{i}. {r.get('title', 'No title')}")
        lines.append(f"   URL: {r.get('url', '')}")
        lines.append(f"   {content}")
    return "\n".join(lines)


_VECTORSTORE: Any = None


def live_retrieve(query: str, cfg: RunConfig) -> str:
    """Similarity search over the frozen Chroma index."""
    global _VECTORSTORE
    chroma_dir = cfg.resolve(cfg.chroma_dir)
    if not chroma_dir.exists():
        raise ToolFailure(f"no vector database at {chroma_dir}")

    if _VECTORSTORE is None:
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ToolFailure(f"retrieval dependencies missing: {exc}") from exc

        # Local weights only. Passing a Hub id here would silently download on
        # first use, which puts network latency inside a measured tool call.
        embed_path = cfg.embedding_path()
        if not embed_path.exists():
            raise ToolFailure(
                f"embedding model not found at {embed_path}; "
                "fetch it first with scripts/fetch_models.py"
            )
        # Embedding device is a controlled variable, recorded in the config.
        # CPU by default: on GPU it would compete with vLLM for VRAM and put
        # the retrieval model inside the same power envelope being measured.
        _VECTORSTORE = Chroma(
            persist_directory=str(chroma_dir),
            collection_name=cfg.chroma_collection,
            embedding_function=HuggingFaceEmbeddings(
                model_name=str(embed_path),
                model_kwargs={"device": cfg.embedding_device},
            ),
        )

    try:
        docs = _VECTORSTORE.similarity_search(query, k=cfg.retrieval_top_k)
    except Exception as exc:
        raise ToolFailure(f"retrieval failed: {exc}") from exc

    if not docs:
        return f"No relevant documents found for '{query}'."

    out: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        out.append(f"--- Document {i} (source: {source}) ---")
        out.append(doc.page_content)
        out.append("")
    return "\n".join(out)


LIVE_IMPLS = {
    "retrieve": lambda args, cfg: live_retrieve(args["query"], cfg),
    "web_search": lambda args, cfg: live_web_search(args["query"], cfg),
    "fred_releases": lambda args, cfg: live_fred_releases(args["window"], cfg),
}

# retrieve is deterministic given a frozen index, so it runs live even in replay
# mode unless it has a fixture. web_search and fred_releases always replay.
ALWAYS_REPLAY = {"web_search", "fred_releases"}


def execute(
    tool_name: str,
    args: dict[str, Any],
    cfg: RunConfig,
    store: FixtureStore,
    source: str = "",
) -> tuple[str, FixtureHit | None]:
    """Run one tool. Returns (result_text, fixture_hit_or_None).

    Only the non-deterministic tools (web_search, fred_releases) ever touch
    fixtures. retrieve always runs live against the frozen index in every mode,
    so a stray fixture can never change its behaviour or latency between runs.

      replay  fixtures only; a missing one is a tool error
      record  probe run: reuse an exact or reworded fixture, otherwise call the
              live API and save it. Fills gaps, never overwrites.
      live    no fixtures at all

    Raises ToolFailure; the registry catches it. Never returns an error string
    directly, so the error path stays in one place.
    """
    replayable = tool_name in ALWAYS_REPLAY
    if replayable and cfg.tool_mode in ("replay", "record"):
        hit = store.load(tool_name, args)
        # A "nearest" match is approximate. In record mode the model's own
        # query is recorded instead, so the later replay run gets an exact hit.
        if hit is not None and (cfg.tool_mode == "replay" or hit.match in ("exact", "normalized")):
            return hit.result, hit
        if cfg.tool_mode == "replay":
            raise ToolFailure(
                f"no fixture for {tool_name}({json.dumps(args, sort_keys=True)}); "
                "record one with a probe run (--tool-mode record)"
            )

    impl_fn = LIVE_IMPLS.get(tool_name)
    if impl_fn is None:
        raise ToolFailure(f"unknown tool '{tool_name}'")
    result = impl_fn(args, cfg)

    if cfg.tool_mode == "record" and replayable:
        fid = store.save(tool_name, args, result, source=source)
        return result, FixtureHit(result, fid, "recorded", 1.0, args)
    return result, None


def warmup(cfg: RunConfig) -> float:
    """Load the embedding model and run one throwaway query.

    Cold-loading bge-large costs ~6s, which would otherwise be charged to
    whichever run happens to be first — and the run order is shuffled, so that
    penalty would land on an arbitrary framework. Called before timing starts.
    """
    started = time.perf_counter()
    try:
        live_retrieve("warmup", cfg)
    except ToolFailure:
        pass  # no index yet; the real call will report it properly
    return (time.perf_counter() - started) * 1000.0


def replay_latency_ms(tool_name: str) -> float:
    """Fixed, recorded stand-in for real tool latency.

    Replayed tools return instantly, which would flatter every framework
    equally but understate T_Tools. Holding a constant here keeps the latency
    decomposition honest and is reported as a constant in the paper.
    """
    return {"retrieve": 0.0, "web_search": 850.0, "fred_releases": 320.0}.get(tool_name, 0.0)


def sleep_replay(tool_name: str) -> None:
    ms = replay_latency_ms(tool_name)
    if ms:
        time.sleep(ms / 1000.0)
