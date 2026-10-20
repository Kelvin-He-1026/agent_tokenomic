# GAIA stack

Runs the agent harness on GAIA (2023 validation split) instead of the Phase 1
tasks. Everything GAIA-specific lives in this folder. The parts every framework
shares (the `read_file` and `fetch_url` tools, the `gaia_exact` scorer, per-config
tool lists) are in `agent/`, like the rest of the harness.

```
GAIA_stack/
├── README.md
├── configs/gaia_qwen_tp2.yaml   the GAIA cell: tools, limits, folders below
├── run_gaia_35.py               unattended sweep: record -> clean sweep -> results
├── scripts/
│   ├── fetch_gaia.py            rebuild tasks/ and documents/ from Hugging Face
│   └── prepare_gaia.py          scoring, tool routes, text-only task list
├── tasks/
│   ├── gaia_task_ids.json       the 75 task ids (committed)
│   ├── phase1-GAIA.json         75 tasks              (not committed, see below)
│   └── phase1-GAIA-text.json    68 text-only tasks    (not committed)
├── documents/                   21 task attachments   (not committed)
├── fixtures/                    recorded web_search / fetch_url results (not committed)
└── fixtures_ddg_first_try/      DuckDuckGo searches from a stopped run, set aside (not committed)
```

## Data is not committed

GAIA asks that its questions and answers not be reshared in a crawlable form,
so they do not leak into model training data. The repo therefore holds only
the task ids. To rebuild the task files and attachments, accept the dataset's
terms at huggingface.co/datasets/gaia-benchmark/GAIA, log in once
(`huggingface-cli login` or `HF_TOKEN`), then:

```bash
.venv/bin/python GAIA_stack/scripts/fetch_gaia.py
```

It downloads the split at a pinned revision, rebuilds `tasks/phase1-GAIA.json`
in the committed order and copies the attachments into `documents/`. The
rebuild is identical to the original files. Recorded fixtures are kept out
too: the searches quote GAIA questions and the results often contain the
answers.

## How GAIA runs differ from Phase 1

| | Phase 1 configs | GAIA config |
|---|---|---|
| tools | retrieve, web_search, fred_releases, calendar tools (default) | `tools: [web_search, fetch_url, read_file]` |
| LLM calls per run | 10 | 30 |
| max output tokens per call | 1024 (16384 in thinking sweeps) | 16384 |
| task timeout | 600 s | 1800 s |
| context | 64k | 128k |
| web_search engine | Tavily (API key, 1,000 credits/month) | Tavily, then DuckDuckGo via `ddgs` (free) once credits run out |
| fixtures | `dataset/fixtures/` | `GAIA_stack/fixtures/` |

- `web_search` uses `web_search_backend: tavily_then_ddg`: Tavily while it has
  credits, then DuckDuckGo. A GAIA run makes about 7 searches (up to 30 on L3
  tasks), more than Tavily's free 1,000 credits a month cover.
  - Out of credit (HTTP 432/433, or a missing or invalid key): every recorder
    switches to DuckDuckGo for the rest of the month, through the marker file
    `output/.tavily_exhausted` (holds the month; delete it to retry Tavily).
  - Rate limited (429): wait and retry, then DuckDuckGo for that query only.
  - DuckDuckGo (`ddgs`) returns the top 5 sources with 300-character snippets,
    like Tavily but without Tavily's generated summary. Searches are spaced at
    least 1.5 s apart across parallel recorders and retried with backoff, since
    DuckDuckGo answers "no results" when queried too fast; a query still empty
    after 4 attempts is recorded as "No results found." so replay stays
    deterministic.
  - Each recording notes the engine that produced it (`backend` in the fixture
    file), so results can be split by engine. The engine depends on when credit
    ran out, so in a sweep it is not balanced across modes: check the split
    before comparing modes. `web_search_backend` values other than `tavily`
    enter `config_id`. Plain `tavily` and plain `ddg` are also available.
- `fetch_url` returns a page's text (HTML stripped, PDFs extracted), capped at
  12,000 characters. It is recorded and replayed like `web_search`, by exact
  URL, so a GAIA cell needs a record pass before replay runs. It refuses
  non-HTTP URLs and local or private addresses.
- `read_file` opens the task's attachment by name from `documents/`: text, CSV,
  JSON/JSON-LD, PDF, DOCX, XLSX (values plus cell fill colours), XLS, and ZIP
  archives (each member in turn). It runs live and never reads outside that
  folder.
- Scoring is `gaia_exact`, GAIA's official quasi-exact match: numbers ignore
  units, commas and `$`/`%`; strings ignore case, spaces and punctuation; lists
  compare element by element. The prompt asks for a closing `FINAL ANSWER:`
  line; an answer without it is scored from its last line with
  `format_ok = false`. No LLM judge is needed.
- 7 tasks have image or audio attachments, which the harness cannot read.
  `prepare_gaia.py` marks them and writes `phase1-GAIA-text.json`, the 68 tasks
  it can attempt (24 easy = L1, 20 medium = L2, 24 hard = L3). Rerun it after
  editing `phase1-GAIA.json`.
- GAIA runs skip loading the retrieval model, since `retrieve` is not offered.

The tool list is provenance, not part of `config_id`; it is covered by
`tool_schema_hash`, which differs from Phase 1's. Phase 1 configs set no
`tools`, so they offer exactly the tools they always have.

Known limits: GAIA's answers date from 2023 and the live web has changed since,
so a few may now be stale. A few tasks refer to a video or image in the question
text without an attachment; they fail in every mode.

## Running

A recording pass (live web, timings not clean), then the clean
replay sweep:

```bash
.venv/bin/python scripts/sweep_thinking.py --base-config GAIA_stack/configs/gaia_qwen_tp2.yaml \
    --tasks GAIA_stack/tasks/phase1-GAIA-text.json --modes off --repeats 1 \
    --difficulty-mix easy=10,medium=15,hard=10 --frameworks langgraph,openai_agents \
    --tool-mode record --workers 6
```

`run_gaia_35.py` does the whole thing unattended for 10 L1 / 15 L2 / 10 L3
tasks, thinking off / low / xhigh, LangGraph + OpenAI Agents SDK: it records
each mode (with Tavily it first checks the remaining credits, since a search
that fails for lack of credit would become a tool failure in the clean sweep),
runs the clean sweep, then exports results, energy by phase and `summary.md`.

```bash
nohup .venv/bin/python -u GAIA_stack/run_gaia_35.py > /dev/null 2>&1 &
```

Tests: `tests/test_gaia.py` (skipped when the task files have not been fetched).
