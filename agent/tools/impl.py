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


# The engine that produced the last live web_search result, saved with the
# fixture ("tavily" or "ddg"; tavily_then_ddg can give either).
LAST_SEARCH_ENGINE = ""


def live_web_search(query: str, cfg: RunConfig) -> str:
    global LAST_SEARCH_ENGINE
    backend = cfg.web_search_backend
    if backend == "tavily_then_ddg":
        return _tavily_then_ddg(query, cfg)
    if backend not in ("tavily", "ddg"):
        raise ToolFailure(f"unknown web_search_backend '{backend}'")
    LAST_SEARCH_ENGINE = backend
    return live_ddg_search(query, cfg) if backend == "ddg" else live_tavily_search(query, cfg)


def live_tavily_search(query: str, cfg: RunConfig) -> str:
    """Tavily web search, formatted as in the reference agent."""
    if not os.getenv("TAVILY_API_KEY", ""):
        raise ToolFailure("TAVILY_API_KEY is not set")
    try:
        return _tavily_search(query)
    except Exception as exc:
        raise ToolFailure(f"web search failed: {exc}") from exc


def _tavily_search(query: str) -> str:
    """One Tavily call; raises the client's own errors."""
    from tavily import TavilyClient

    results = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", "")).search(
        query=query,
        search_depth="basic",
        topic="general",
        max_results=5,
        include_answer=True,
    )
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


def _tavily_marker(cfg: RunConfig) -> Path:
    return cfg.resolve("output/.tavily_exhausted")


def _tavily_exhausted(cfg: RunConfig) -> bool:
    """Set once Tavily refuses for lack of credit; shared by parallel recorders.
    Holds the month it was set in, so it lapses when the monthly credits reset."""
    try:
        return _tavily_marker(cfg).read_text().strip() == datetime.now().strftime("%Y-%m")
    except OSError:
        return False


def _tavily_then_ddg(query: str, cfg: RunConfig) -> str:
    """Tavily while it has credits, then DuckDuckGo.

    Out of credit (HTTP 432/433, or a missing or invalid key) switches every
    recorder to DuckDuckGo for the rest of the month. A rate limit (429) waits
    and retries, then falls back for this one query only. Any other Tavily
    error is a tool failure, as with the plain tavily backend."""
    global LAST_SEARCH_ENGINE
    if os.getenv("TAVILY_API_KEY", "") and not _tavily_exhausted(cfg):
        from tavily.errors import ForbiddenError, InvalidAPIKeyError, UsageLimitExceededError

        for attempt in range(3):
            try:
                result = _tavily_search(query)
                LAST_SEARCH_ENGINE = "tavily"
                return result
            except UsageLimitExceededError:  # 429: too many requests just now
                time.sleep(5 * (attempt + 1))
            except (ForbiddenError, InvalidAPIKeyError):  # 432/433: credits used up
                marker = _tavily_marker(cfg)
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(datetime.now().strftime("%Y-%m"))
                break
            except Exception as exc:
                raise ToolFailure(f"web search failed: {exc}") from exc
    LAST_SEARCH_ENGINE = "ddg"
    return live_ddg_search(query, cfg)


DDG_MIN_INTERVAL_S = 1.5  # between searches, across all parallel recorders
DDG_ATTEMPTS = 4
DDG_SNIPPET = 300  # characters per result, as for Tavily


def _ddg_throttle(cfg: RunConfig) -> None:
    """Keep searches at least DDG_MIN_INTERVAL_S apart across processes.

    Parallel recording workers are separate processes, so the last search time
    lives in a small file guarded by an exclusive lock. DuckDuckGo answers
    "no results" when queried too fast."""
    import fcntl

    stamp = cfg.resolve("output/.ddg_last_search")
    stamp.parent.mkdir(parents=True, exist_ok=True)
    with open(stamp, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        try:
            last = float(fh.read().strip() or 0)
        except ValueError:
            last = 0.0
        wait = last + DDG_MIN_INTERVAL_S - time.time()
        if wait > 0:
            time.sleep(wait)
        fh.seek(0)
        fh.truncate()
        fh.write(f"{time.time():.3f}")
        fcntl.flock(fh, fcntl.LOCK_UN)


def live_ddg_search(query: str, cfg: RunConfig) -> str:
    """DuckDuckGo-led metasearch via ddgs: no API key, no credits.

    ddgs reports "no results" both for a real empty result and when it is being
    rate limited, so each query is retried with backoff, falling back from the
    default engine mix to Bing. A query still empty after that is answered
    "No results found." and recorded like any result, so replay stays
    deterministic. Unlike Tavily there is no generated summary: sources only."""
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise ToolFailure("ddgs is not installed (pip install ddgs)") from exc

    results: list[dict[str, Any]] = []
    for attempt in range(DDG_ATTEMPTS):
        for backend in ("auto", "bing"):
            _ddg_throttle(cfg)
            try:
                results = DDGS(timeout=20).text(query, max_results=5, backend=backend)
            except Exception:  # noqa: BLE001 - "no results", timeouts and rate limits look alike
                results = []
            if results:
                break
        if results:
            break
        time.sleep(3 * (attempt + 1))
    if not results:
        return "No results found."

    lines = ["Sources:"]
    for i, r in enumerate(results, 1):
        body = (r.get("body") or "").strip()
        if len(body) > DDG_SNIPPET:
            body = body[:DDG_SNIPPET] + "..."
        lines.append(f"\n{i}. {r.get('title') or 'No title'}")
        lines.append(f"   URL: {r.get('href', '')}")
        lines.append(f"   {body}")
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


# -- GAIA tools ------------------------------------------------------------------
MAX_TOOL_TEXT = 20000  # characters returned per call; longer output is cut and says so


MAX_PAGE_TEXT = 12000  # characters per fetched page


def _cap(text: str, limit: int = MAX_TOOL_TEXT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... truncated: {len(text) - limit} more characters not shown]"


def _blocked_host(host: str) -> bool:
    """Never fetch the machine itself or a private network (the proxy and vLLM
    listen on localhost)."""
    import ipaddress
    import socket

    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return True
    return False


def html_to_text(html: str) -> str:
    """Readable text from HTML with the standard library: drop scripts, styles
    and navigation, keep headings, paragraphs and table cells on their own lines."""
    from html.parser import HTMLParser

    class _Text(HTMLParser):
        skip = {"script", "style", "noscript", "svg", "head", "nav", "footer"}
        block = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table",
                 "section", "article", "header", "dd", "dt", "caption"}

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []
            self.depth = 0

        def handle_starttag(self, tag, attrs):
            if tag in self.skip:
                self.depth += 1
            elif tag in self.block:
                self.parts.append("\n")
            elif tag in ("td", "th"):
                self.parts.append(" | ")

        def handle_endtag(self, tag):
            if tag in self.skip and self.depth:
                self.depth -= 1

        def handle_data(self, data):
            if not self.depth:
                self.parts.append(data)

    parser = _Text()
    parser.feed(html)
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line)


def live_fetch_url(url: str, cfg: RunConfig) -> str:
    from urllib.parse import urlparse

    import requests

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ToolFailure(f"only http(s) URLs can be fetched, got '{url[:100]}'")
    if _blocked_host(parsed.hostname):
        raise ToolFailure(f"refusing to fetch a local or private address: {parsed.hostname}")
    try:
        resp = requests.get(url.strip(), timeout=20, allow_redirects=True,
                            headers={"User-Agent": "Mozilla/5.0 (agent-tokenomics benchmark)"})
        resp.raise_for_status()
    except Exception as exc:
        raise ToolFailure(f"fetch failed: {exc}") from exc
    kind = resp.headers.get("content-type", "")
    if "pdf" in kind:
        import io

        from pypdf import PdfReader

        text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(resp.content)).pages)
    elif "html" in kind or resp.text.lstrip().startswith("<"):
        text = html_to_text(resp.text)
    else:
        text = resp.text
    # Pages are capped tighter than files: a GAIA run may read many of them, and
    # every page stays in the context for the rest of the run.
    return _cap(f"URL: {resp.url}\n\n{text}", MAX_PAGE_TEXT)


TEXT_SUFFIXES = {".txt", ".csv", ".tsv", ".json", ".jsonld", ".md", ".xml", ".py", ".html", ".htm"}
UNREADABLE = {".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image", ".webp": "image",
              ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".mp4": "video", ".mov": "video"}


def _xlsx_text(data: bytes) -> str:
    import io

    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), data_only=True)
    out = []
    for ws in wb.worksheets:
        out.append(f"## Sheet: {ws.title} ({ws.max_row} rows x {ws.max_column} columns)")
        colours = []
        for row in ws.iter_rows():
            values = ["" if c.value is None else str(c.value) for c in row]
            if any(values):
                out.append(f"row {row[0].row}: " + " | ".join(values))
            for c in row:
                fill = getattr(c.fill, "fgColor", None)
                rgb = getattr(fill, "rgb", None) if fill is not None else None
                if c.fill is not None and c.fill.fill_type and isinstance(rgb, str) and rgb not in ("00000000", "FFFFFFFF"):
                    colours.append(f"{c.coordinate}=#{rgb[-6:]}")
        if colours:
            out.append("Cell fill colours: " + ", ".join(colours))
    return "\n".join(out)


def _convert(name: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in UNREADABLE:
        raise ToolFailure(f"{name} is {UNREADABLE[suffix]}; this harness has no {UNREADABLE[suffix]} "
                          "understanding, so its contents cannot be read")
    if suffix in TEXT_SUFFIXES:
        text = data.decode("utf-8", errors="replace")
        return html_to_text(text) if suffix in (".html", ".htm") else text
    if suffix in (".xlsx", ".xlsm"):
        return _xlsx_text(data)
    if suffix == ".xls":  # pre-2007 Excel
        import xlrd

        book = xlrd.open_workbook(file_contents=data)
        out = []
        for sh in book.sheets():
            out.append(f"## Sheet: {sh.name} ({sh.nrows} rows x {sh.ncols} columns)")
            for r in range(sh.nrows):
                values = ["" if v in (None, "") else str(v) for v in sh.row_values(r)]
                if any(values):
                    out.append(f"row {r + 1}: " + " | ".join(values))
        return "\n".join(out)
    if suffix == ".pdf":
        import io

        from pypdf import PdfReader

        return "\n".join(f"[page {i + 1}]\n{p.extract_text() or ''}" for i, p in enumerate(PdfReader(io.BytesIO(data)).pages))
    if suffix == ".docx":
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return html_to_text(z.read("word/document.xml").decode("utf-8", errors="replace"))
    raise ToolFailure(f"{name}: unsupported file type '{suffix}'")


def live_read_file(file_name: str, cfg: RunConfig) -> str:
    import io
    import zipfile

    name = Path(file_name.strip()).name  # attachments only; no paths
    path = cfg.resolve(cfg.attachments_dir) / name
    if not name or not path.is_file():
        raise ToolFailure(f"no attached file named '{file_name[:100]}'")
    data = path.read_bytes()
    if path.suffix.lower() == ".zip":
        parts = [f"# Archive {name}"]
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            members = [m for m in z.infolist() if not m.is_dir()]
            parts.append("Members: " + ", ".join(f"{m.filename} ({m.file_size} bytes)" for m in members))
            for m in members:
                try:
                    parts.append(f"\n## {m.filename}\n{_convert(m.filename, z.read(m))}")
                except ToolFailure as exc:
                    parts.append(f"\n## {m.filename}\n[not readable: {exc}]")
        return _cap("\n".join(parts))
    return _cap(f"# File {name}\n{_convert(name, data)}")


LIVE_IMPLS = {
    "retrieve": lambda args, cfg: live_retrieve(args["query"], cfg),
    "web_search": lambda args, cfg: live_web_search(args["query"], cfg),
    "fred_releases": lambda args, cfg: live_fred_releases(args["window"], cfg),
    "fetch_url": lambda args, cfg: live_fetch_url(args["url"], cfg),
    "read_file": lambda args, cfg: live_read_file(args["file_name"], cfg),
}

# retrieve is deterministic given a frozen index, so it runs live even in replay
# mode unless it has a fixture; read_file reads a local attachment, the same
# every time. web_search, fred_releases and fetch_url always replay: the web
# changes, so runs see the recorded page, keyed by the exact URL.
ALWAYS_REPLAY = {"web_search", "fred_releases", "fetch_url"}


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
        backend = LAST_SEARCH_ENGINE if tool_name == "web_search" else ""
        fid = store.save(tool_name, args, result, source=source, backend=backend)
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
    return {"retrieve": 0.0, "web_search": 850.0, "fred_releases": 320.0,
            "fetch_url": 900.0}.get(tool_name, 0.0)


def sleep_replay(tool_name: str) -> None:
    ms = replay_latency_ms(tool_name)
    if ms:
        time.sleep(ms / 1000.0)
