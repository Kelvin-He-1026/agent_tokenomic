# agent_tokenomic

Benchmark harness for the Agent Tokenomics study: how agent runtime, architecture,
model and precision affect quality, behaviour, latency, infrastructure cost and
cost per successful task.

Phase 1 compares **LangGraph / CrewAI / OpenAI Agents SDK** running the same
ReAct agent, over the same tools, against the same vLLM endpoint. Methodology,
including the parity checklist this code implements, is in the Phase 1 doc.

## Layout

```
agent/                   the package
├── config.py            RunConfig: every controlled variable, hashed into config_id
├── prompts.py           the one system prompt all frameworks must send
├── telemetry.py         the six record types + JSONL writer
├── tools/
│   ├── schemas.py       tool names, descriptions, JSON schemas (single source)
│   ├── impl.py          the three tool implementations + fixture record/replay
│   └── registry.py      per-run timing, logging, and the single error path
├── adapters/
│   ├── base.py          runtime-neutral interface + shared client/sampling settings
│   └── __init__.py      the runtime factory
├── langGraph/           LangGraph runtime
│   └── react_agent.py   Phase 1 architecture (Phase 2 adds judge_loop, router_judge)
├── proxy/
│   ├── server.py        OpenAI-compatible logging proxy (source of LLM truth)
│   └── mock_upstream.py scripted LLM server, so everything runs with no GPU
├── harness/
│   ├── runner.py        one run: binds the proxy, assembles the run record
│   └── cli.py           run a cell, frameworks interleaved per task
└── dataset/build_tasks.py   reference GT + authored items -> Phase 1 tasks

configs/     pilot_mock.yaml (no GPU), pilot_qwen.yaml (Qwen3.8-27B cell)
dataset/     tasks/ (annotated), fixtures/ (recorded tool responses), authored_tasks.json
model/       local weights, one dir per model + MANIFEST.json (never fetched at run time)
db/chroma/   the frozen retrieval index
output/runs/ telemetry per cell: config/run/llm_call/tool_call/eval + blobs/
scripts/     start_stack.sh, stop_stack.sh, fetch_models.py, serve_vllm.sh,
             build_index.py, record_fixtures.py, parity_check.py
```

Each runtime gets its own folder (`langGraph/`, and later `crewAI/`,
`openai_agents/`), with one file per architecture inside it. `adapters/` holds
only what is runtime-neutral.

## Models are local

Weights live under `model/<local_dir>` and are never pulled from the Hub at run
time: a download mid-run is neither reproducible nor offline-safe, and it would
land inside a measured first token or tool call. `scripts/fetch_models.py` is
the only code that reaches the network, and it records the resolved upstream
commit in `model/MANIFEST.json`. Set `HF_HUB_OFFLINE=1` for runs so a missing
local copy fails loudly.

## Corpus

The retrieval index is built from five research papers in `documents/`. They
are third-party works and are not redistributed in this repository; obtain
them from their publishers, place them in `documents/`, and rebuild the index
with `scripts/build_index.py`. The checksums below (SHA-256, first 16 hex
characters, as recorded in `db/manifest.json`) confirm you have the identical
files the published results used.

| File | SHA-256 (16) |
| --- | --- |
| `Agentic Context Engineering.pdf` | `880e354fc747d444` |
| `Attention Is All You Need.pdf` | `bdfaa68d8984f0dc` |
| `Dynamic RAG by TsinguaU.pdf` | `02807e9427f7e240` |
| `HOTPOTQA ADataset for Diverse.pdf` | `1ff47fefeb264db4` |
| `ManuRAG.pdf` | `50f87e72945c3bda` |

Index settings: 1000-character chunks, 150 overlap,
top-5, 400 chunks, embeddings from `BAAI/bge-large-en-v1.5`.

## Measurement principle

Nothing a framework reports about itself is used. `llm_call` rows come from the
proxy, `tool_call` rows from the shared tool layer, and the run record joins
them. A framework's own step count is recorded only so a gap between it and the
proxy's view becomes visible, which is itself a finding.

`T_Orch` is a residual: `e2e - t_llm - t_tool - t_queue`. With no replay arm it
absorbs measurement error, which is why the mock upstream exists — it can
measure orchestration overhead directly, with no GPU.

## Running

### One-time setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fetch_models.py --config configs/pilot_qwen.yaml   # ~56 GB
.venv/bin/python scripts/build_index.py --config configs/pilot_qwen.yaml --documents documents
.venv/bin/python scripts/record_fixtures.py --config configs/pilot_qwen.yaml  # needs .env keys
```

`.env` holds `TAVILY_API_KEY` and `FRED_API_KEY` as `KEY=value`. It is gitignored,
and secrets are stripped from every error string before logging.

### Smoke test, no GPU

The mock upstream answers like a scripted model, so the whole pipeline runs
without weights. Use it after any harness change.

```bash
./scripts/start_stack.sh --mock
.venv/bin/python -m agent.harness.cli \
    --config configs/pilot_mock.yaml --tasks dataset/tasks/phase1.json \
    --frameworks langgraph --repeats 1 --limit 3 --out output/runs/smoke
./scripts/stop_stack.sh --all
```

### A real cell

Three processes: vLLM on 8000, the proxy on 8100, the harness. The harness only
ever talks to the proxy.

```bash
# 1. serve the model, and wait for it (2-4 min: weights, JIT, CUDA graphs)
nohup ./scripts/serve_vllm.sh --config configs/pilot_qwen.yaml > /tmp/vllm_serve.log 2>&1 &
./scripts/wait_ready.sh          # blocks; exits 0 when serving, 1 if it failed

# 2. start the proxy pointed at it
ATK_UPSTREAM=http://127.0.0.1:8000 ./scripts/start_stack.sh

# 3. run
.venv/bin/python -m agent.harness.cli \
    --config configs/pilot_qwen.yaml --tasks dataset/tasks/phase1.json \
    --frameworks langgraph --repeats 3 --limit 15 --out output/runs/pilot_qwen

# 4. export
.venv/bin/python scripts/export_csv.py --runs output/runs/pilot_qwen \
    --out output/results.csv --calls
```

### Stopping

```bash
./scripts/stop_stack.sh          # proxy only; leaves vLLM up for the next run
./scripts/stop_stack.sh --all    # proxy + vLLM, and confirms the GPU is free
./scripts/stop_vllm.sh           # vLLM alone
```

Stopping the proxy alone is the normal case between runs: vLLM takes minutes to
reload, so leave it up.

Never stop vLLM with a bare `kill` on the port's pid. The model lives in a
`VLLM::EngineCore` child that holds the VRAM, and it can outlive an abrupt
parent kill, leaving tens of GB reserved so the next serve cannot allocate.
`stop_vllm.sh` terminates the parent, waits, escalates to the child if needed,
and then prints the GPU state:

```
stopping vLLM (pid 4145471, children: 4145893 4145894)
GPU memory in use: 3 MiB 3 MiB
GPU released
```

If it ever reports processes still holding the GPU, check with
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`.

### Useful flags

| Flag | Effect |
| --- | --- |
| `--frameworks a,b,c` | Interleaved per task, never run in blocks |
| `--repeats N` | Repeats per task; 5 is the provisional Phase 1 value |
| `--limit N` | Sample N tasks spread across all task types (0 = all, the default) |
| `--tool-mode record` | Probe run: tools hit live APIs and every call is saved as a fixture |
| `--seed N` | Seeds the run-order shuffle |
| `--no-timestamp` | Use `--out` exactly; by default a `_YYYYmmdd_HHMMSS` suffix is added |

Every run starts with a preflight check: proxy reachable, its upstream
answering, and the upstream serving the model the config names. It fails in a
second with instructions rather than producing one `framework_error` per run.

### Is it up?

```bash
./scripts/status.sh              # one glance: vLLM, proxy, GPU, running benchmark
./scripts/wait_ready.sh          # blocks until vLLM answers, or explains why not
```

`status.sh` reports and exits; use it any time. `wait_ready.sh` is for right
after starting the server, when you want to wait for it.

Nothing answers on the port until startup fully completes, so a quiet terminal
during those first minutes is normal. `wait_ready.sh` also watches the log and
gives up early if the engine has already failed, instead of waiting out the
timeout.

Other entry points: `scripts/parity_check.py` (pilot gate 2, diffs what each
framework actually sent) and `agent/dataset/build_tasks.py` (rebuild the task set).

### Thinking mode

**Set it in the config, nowhere else** — `model.thinking_setting` and
`model.reasoning_effort` in `configs/<cell>.yaml`:

```yaml
model:
  thinking_setting: "off"      # off | on
  reasoning_effort: ""         # "" when off; "low" | "medium" | "xhigh" when on
```

Then restart vLLM from that config, since thinking mode is a server setting:

```bash
./scripts/stop_vllm.sh
nohup ./scripts/serve_vllm.sh --config configs/pilot_qwen.yaml > /tmp/vllm_serve.log 2>&1 &
./scripts/wait_ready.sh
```

Qwen3.8's chat template defaults to thinking **on** at `xhigh`. That default is
never inherited: leaving it alone roughly doubled output tokens and end-to-end
time, and glued the reasoning into the answer.

How the config stays in charge:

1. **Validated on load.** `on` without an effort, `off` with one, or an effort
   the model's template does not accept, are refused before the GPU is touched.
2. **The server is launched from it.** `serve_vllm.sh --config` derives every
   serving setting from the YAML and records what it launched in
   `output/serving_<port>.json`.
3. **Checked before every run.** Preflight compares the config against that
   launch record field by field (parsers, context length, TP, thinking, effort)
   and probes the live server for reasoning. Any mismatch aborts the run:

   ```
   the running server does not match this config:
       thinking_setting: server='off'  config='on'
       reasoning_effort: server=''  config='medium'
   ```
4. **Recorded with the results.** Effort is part of the cell, so each
   combination gets its own `config_id`, and `thinking_setting` and
   `reasoning_effort` are columns in `results.csv`.

`--reasoning-parser` splits reasoning into its own field. vLLM reports thinking
token counts (`usage.completion_tokens_details.reasoning_tokens`) only when a
parser is set. In the CSVs, `output_tokens = reasoning_tokens + answer_tokens`:
thinking tokens are generated and paid for like any other output, so they stay
in the total. `reasoning_tokens_missing` flags rows with reasoning text but no
count — unknown, not zero. Every `llm_call` row also records `reasoning_chars`,
so a framework that overrides the server default per request shows up.

To compare thinking modes, sweep them in one command:

```bash
.venv/bin/python scripts/sweep_thinking.py --base-config configs/pilot_qwen.yaml \
    --tasks dataset/tasks/phase1.json --modes off,low,medium,xhigh \
    --repeats 3 --limit 15            # add --dry-run to validate and print the plan
```

Each mode is its own cell: a generated config, a vLLM restart from it (~95 s),
and its own run directory under `output/runs/thinking_sweep_<stamp>/`, with the
configs and server logs saved beside the results. Every mode is validated before
the first restart, and the sweep stops at the first failure rather than leaving
a partial result that looks complete. It finishes by exporting all modes into
one `results.csv`, summarised per cell. vLLM is left serving the last mode.

### Serving other models

`serve_vllm.sh --config configs/<cell>.yaml`, with `tp_size: 2` in the config for the 70B.
**Confirm the tool parser per model before serving** — it is read off the chat
template, not guessed. Qwen3.8 uses `qwen3_xml` (XML function format), which is
not the same as `hermes` (JSON inside `<tool_call>`). A wrong parser makes every
tool call malformed.

Known environment quirks, already handled in the script: `CUDA_HOME` points at
the pip CUDA wheels, the venv is put on `PATH` for `ninja`,
`--max-num-seqs 256` fits the hybrid Mamba cache blocks, and
`VLLM_USE_FLASHINFER_SAMPLER=0` avoids a JIT toolchain mismatch.

## Results

`export_csv.py` joins the four JSONL tables into one row per run, 62 columns:
identity, outcome, trajectory, tokens, the latency decomposition, resources,
fixture-match counts and provenance hashes. `--calls` adds a second, long-form
CSV with one row per LLM call, for reading trajectories step by step.

Nothing is filtered or corrected on export. Warm-up runs, framework errors and
approximate fixture matches are all present and flagged by their own columns
(`is_first_in_dir`, `status`, `has_approx_fixture`), so the analysis decides what
to exclude rather than the exporter deciding silently. The printed summary
excludes warm-up and non-completed runs, and says which rows it left out.

## Outputs are timestamped

Nothing overwrites an earlier run:

- **Harness**: `--out output/runs/pilot` writes to `output/runs/pilot_20260918_193012/`.
- **Export**: `--out output/results.csv` writes `output/results_20260918_193012.csv`.
- **Sweeps**: `output/runs/thinking_sweep_<model>_<stamp>/`, and probe sweeps
  `output/runs/probe_<model>_<stamp>/`.

`--no-timestamp` opts out; the sweep uses it inside its own stamped directory.

## Scoring: did the agent succeed?

`agent/scoring.py` decides success deterministically from the final answer,
using each task's `answer_check`. It is the primary measure; an LLM judge
(DeepEval) is secondary. Four modes:

| Mode | Success when | Used for |
| --- | --- | --- |
| `key_facts` | at least `min_hits` facts appear; each fact may list alternatives, so paraphrase passes | corpus and FRED tasks |
| `numeric_tolerance` | a number in the answer is within tolerance, matched on number boundaries | arithmetic tasks |
| `abstain` | the answer declines, and no `forbid` pattern (an invented figure) matches | unanswerable and out-of-scope tasks |
| `judge_only` | not scored here: no stable ground truth | time-sensitive web questions |

A run that did not complete (`step_limit`, `truncated`, ...) is never a success.

The checks live in `dataset/reviewed_checks.json`, applied on top of the source
tasks by `agent/dataset/build_tasks.py`. Two rules are enforced when building:

- **No fact may come from the question.** A fact the question already contains
  is satisfied by any answer that restates it. The builder refuses it unless
  the task opts in with `allow_question_terms` and a written reason.
- **Every check is self-tested.** Each reference answer must pass its own check,
  and the question on its own must fail it.

FRED ground truth is built from our recorded fixtures, not the reference repo's
answers, which came from a different FRED snapshot and watchlist.

The export adds `success`, `check_mode` and `success_detail` columns, and its
summary reports success rate, **tokens per successful task** and **seconds per
successful task**. Failed runs still count in the numerator: they cost tokens
too.

Review status: checked against the reference answers, the corpus and the
fixtures, not yet human-verified (`review.human_verified: false` on every task).

Rebuild the task set after editing any source:

```bash
.venv/bin/python -m agent.dataset.build_tasks \
    --reference dataset/reference/*.json --authored dataset/authored_tasks.json \
    --reviewed dataset/reviewed_checks.json --out dataset/tasks/phase1.json
```

## LLM judge: DeepEval, judged by Claude via Claude Code

The deterministic scorer is primary. DeepEval adds what string matching cannot
judge: faithfulness to tool output, task completion, tool and argument choice.
It scores **stored runs offline**. Runs are never re-executed, and each test
case is built from the proxy's request bodies and the tool layer's rows
(`agent/eval/cases.py`), so every framework is judged from identical input.

```bash
.venv/bin/python scripts/judge_runs.py --runs "output/runs/thinking_sweep_*/*" \
    --select judge_only --metrics faithfulness,task_completion   # --dry-run to preview
```

`--select` takes `judge_only`, `all`, or task types. Results go to each run
directory's `eval.jsonl`, and the pass is resumable: already-scored pairs are
skipped, so a run stopped by usage limits continues where it left off. The CSV
export adds `judge_<metric>` and `judge_<metric>_pass` columns, plus what judging
each run cost (`judge_calls`, `judge_input_tokens`, `judge_output_tokens`).

**The judge runs on a Claude subscription**, through Claude Code's headless mode
(`agent/eval/claude_code_judge.py`), with no API key. Each call is isolated
from Claude Code's usual environment: its own system prompt, no tools, no MCP
connectors, no settings, hooks, skills or project memory, and no saved session.
Background model calls are suppressed, so only the judge model runs; about 430
tokens of Claude Code preamble remain around each prompt. `--bare` would remove
that, but it cannot use a subscription login.

Caveats for publication: this is Claude Code's harness rather than a direct
API call to a pinned model, and subscription usage limits apply. Use it for
pilots and development. For the paper's numbers, swap in a direct API judge;
the case builder and runner stay the same. The judge is from a different model
family than every agent under test, as the methodology requires. Disclose that
Claude also helped build the evaluation.

## Web fixtures: the probe run

`web_search` replays recorded results, keyed by the exact query. Different
models word searches differently, so fixtures are recorded by a **probe sweep**
that runs every cell once in record mode:

```bash
.venv/bin/python scripts/sweep_thinking.py --base-config configs/<cell>.yaml \
    --tasks dataset/tasks/phase1.json --modes <modes> --repeats 1 --tool-mode record
```

In record mode, `web_search` and `fred_releases` reuse an existing exact or
reworded fixture, and otherwise call the live API once and save the result.
`retrieve` is never recorded: it always runs live against the frozen index, so
its behaviour and latency match every other run. Probe results are for
recording fixtures only; tool latency is live in them, so do not analyse them.

## Parity, in code

Settings that must match across frameworks live in shared modules, so parity is
structural rather than a matter of remembering to copy values:

| Parity row | Where it is enforced |
| --- | --- |
| System prompt | `prompts.build_system_prompt` |
| Tool schemas and descriptions | `tools/schemas.TOOL_SCHEMAS`, passed through unmodified |
| Endpoint, timeout, retries | `adapters/base.client_kwargs` |
| Sampling, streaming | `adapters/base.sampling_kwargs` |
| Step budget | `adapters/base.recursion_limit_for` and each adapter's own mapping |
| Tool errors | `tools/registry.ToolRegistry.call`, one fixed error string |
| Frozen clock | `RunConfig.as_of_date`, never `datetime.now()` |

`scripts/parity_check.py` diffs what each framework actually sent.

## Status

- Working: config, telemetry, tool layer with record/replay, logging proxy,
  mock upstream, LangGraph ReAct agent, runner, CLI, dataset builder, CSV export.
  First real cell ran against Qwen3.8-27B on vLLM.
- Next: CrewAI and OpenAI Agents SDK adapters; the hardware sampler (NVML +
  RAPL); DeepEval scoring over stored traces; the analysis notebook.
- Dataset: 34 tasks in `dataset/tasks/phase1.json` — 20 deduplicated from the
  reference ground truth, 14 authored (no-tool, unanswerable, tool-error,
  multi-hop, multi-tool). Proposed `key_facts` are candidates, not ground
  truth: every item is `needs_review: true` until a human confirms it.
