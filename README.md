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
├── openai_agents/       OpenAI Agents SDK runtime
│   └── react_agent.py   Phase 1 architecture
├── crewAI/              CrewAI runtime
│   └── react_agent.py   Phase 1 architecture
├── proxy/
│   ├── server.py        OpenAI-compatible logging proxy (source of LLM truth)
│   └── mock_upstream.py scripted LLM server, so everything runs with no GPU
├── harness/
│   ├── runner.py        one run: binds the proxy, assembles the run record
│   └── cli.py           run a cell, frameworks interleaved per task
└── dataset/build_tasks.py   reference GT + authored items -> Phase 1 tasks

configs/     one YAML per cell: pilot_mock (no GPU), pilot_qwen, pilot_nemotron,
             pilot_llama8b, pilot_llama70b
dataset/     tasks/phase1.json (the 50 tasks that run), its sources (reference/,
             authored_tasks.json, reviewed_checks.json), fixtures/ (recorded tool responses)
model/       local weights, one dir per model + MANIFEST.json (never fetched at run time)
db/chroma/   the frozen retrieval index
output/runs/ telemetry per cell: config/run/llm_call/tool_call/eval + blobs/
scripts/     start_stack.sh, stop_stack.sh, fetch_models.py, serve_vllm.sh,
             build_index.py, record_fixtures.py, parity_check.py
```

Each runtime gets its own folder (`langGraph/`, `openai_agents/`, `crewAI/`),
with one file per architecture inside it. `adapters/` holds only
what is runtime-neutral.

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
    --frameworks langgraph,openai_agents --repeats 1 --limit 3 --out output/runs/smoke
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
    --frameworks langgraph,openai_agents --repeats 3 --limit 15 --out output/runs/pilot_qwen

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
| `--run-code LABEL` | Names what this execution is for (e.g. `openAISDK-v1`). Stored on every run row and exported as the `run_code` column; not part of `config_id` |
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

### Tensor parallelism (TP)

TP is a serving setting, so it lives in the config: `model.tp_size`, part of
the cell and its `config_id`, launched by `serve_vllm.sh` and checked by
preflight like the parsers. To make a TP variant of a cell:

```bash
.venv/bin/python scripts/make_cell.py --base configs/pilot_qwen.yaml --tp-size 2
# -> configs/pilot_qwen_tp2.yaml; checks GPU count and that head counts divide evenly
nohup ./scripts/serve_vllm.sh --config configs/pilot_qwen_tp2.yaml > /tmp/vllm_serve.log 2>&1 &
```

Then run, judge and export it like any other cell. A sweep takes
`--tp-size 2`, applied to every mode. TP=2 uses both GPUs, so it runs with no
stack and nothing alongside it (both the sweep and `serve_vllm.sh` refuse
otherwise).

The GPUs are on different sockets without NVLink, so each layer's all-reduce
crosses PCIe and the socket link. Even so, one request at a time is limited by
memory bandwidth, and TP=2 doubles it: a first check with Qwen3.8-27B on three
phase1-v2 tasks gave 21.0 ms per output token against 37.5 ms at TP=1, and runs
finished in 17.3 s against 29.7 s (medians; a small sample). TP=2 changes the
cell, so compare it with TP=1 as a variable, not as a drop-in.

### Two stacks at once (one per GPU)

The two GPUs sit on different sockets with no NVLink, so tensor parallelism
across them slows each token and changes the measured cell. Instead each GPU
can carry its own independent stack, so two cells run at the same time while
each stays at concurrency 1. Defined in `agent/stacks.py`:

| Stack | GPU | NUMA node (cores) | vLLM | Proxy |
| --- | --- | --- | --- | --- |
| `a` | 0 | 1 (43-85) | 8000 | 8100 |
| `b` | 1 | 3 (129-171) | 8001 | 8101 |
| none | all | unpinned | 8000 | 8100 |

Set `ATK_STACK=a` or `b` for the shell scripts, and pass `--stack a|b` to the
harness and the sweep. vLLM, the proxy and the harness (with its CPU
embedding model) are then pinned to the GPU's NUMA node.

```bash
# stack b: serve, proxy, run
ATK_STACK=b nohup ./scripts/serve_vllm.sh --config configs/pilot_llama8b.yaml > /tmp/vllm_serve_b.log 2>&1 &
ATK_STACK=b LOG=/tmp/vllm_serve_b.log ./scripts/wait_ready.sh
ATK_STACK=b ./scripts/start_stack.sh
.venv/bin/python -m agent.harness.cli --stack b --config configs/pilot_llama8b.yaml ...
ATK_STACK=b ./scripts/stop_stack.sh --all     # stops stack b only

# or a whole sweep on stack b, alongside one on stack a
.venv/bin/python scripts/sweep_thinking.py --stack b --base-config configs/pilot_qwen.yaml ...
```

- A config with `tp_size: 2` (the 70B) needs both GPUs and is refused on a
  stack; run it with no stack, alone.
- Stack `a` and no stack use the same GPU and ports, so do not mix them.
- Each run records `hardware_id` (`gpu1`) and `numa_pinning` (`node3`). The
  proxy address is provenance too, so the same cell has the same `config_id`
  on either stack.
- `./scripts/status.sh` shows both stacks. Two sweeps hold separate locks
  (`output/.sweep.lock`, `output/.sweep_b.lock`).

## Power and hardware use

`scripts/power_sampler.py` records GPU power and energy (NVML: the driver's own
energy counter, so energy between two moments is exact), SM utilisation, VRAM,
and CPU busy fraction per NUMA node, every 100 ms, on the same wall clock as the
runs. `scripts/attribute_energy.py` then splits each run into LLM time (a
request with vLLM), tool time (retrieval, replayed web results) and
orchestration, with energy and utilisation per phase:

```bash
setsid nohup .venv/bin/python -u scripts/power_sampler.py --until-pid <driver pid> \
    > /dev/null 2>&1 < /dev/null &          # -> output/power/power_<stamp>.jsonl
.venv/bin/python scripts/attribute_energy.py --runs "output/runs/<sweep>/*" \
    --idle-w 88 --out output/energy.csv     # idle: per-GPU W with the model loaded, no requests
```

The harness now starts the sampler itself for every benchmark run (not in
record mode), writing `<run folder>/power.jsonl`, and the export fills
`energy_gpu_j` and `energy_cpu_j` per run from it (or from a trace in
`output/power/`). Pass `--no-power` to skip it. Energy is left blank for runs
that overlapped another run, since a shared window cannot be split per run.

CPU power needs Intel RAPL, which is root-only here. To let the sampler read it
until the next reboot: `sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj`.
Without it, CPU energy columns stay empty and only CPU utilisation is recorded.

Per-run splits are only meaningful when runs do not overlap: parallel recording
(`--workers`) runs several at once, so `overlap_runs` > 0 there and only totals
are usable. Tool calls carry a `start_ts` from 2026-10-16 on; older runs have
tool time folded into orchestration.

## Results

`export_csv.py` joins the four JSONL tables into one row per run, 62 columns:
identity, outcome, trajectory, tokens, the latency decomposition, resources,
fixture-match counts and provenance hashes. `--calls` adds a second, long-form
CSV with one row per LLM call, for reading trajectories step by step.

Each call row also splits its prompt by source (`agent/prompt_breakdown.py`),
counted with the served model's own tokenizer from the saved request:
`system_prompt_tokens`, `conversation_history_tokens` (earlier assistant and
user turns, including tool-call arguments), `tool_response_tokens` (every tool
result in the request) and `current_user_prompt_tokens` (the latest user
message). These count message text; `other_prompt_tokens` is the rest of the
server's `prompt_tokens`, i.e. chat-template markup and the tool definitions,
so the five add up to `prompt_tokens`. Counting is done at export, not in the
proxy, so it never touches measured latency, and it works on older runs too as
long as their request blobs were saved. The main CSV carries the same five
columns summed over each run's calls.

The export also writes each run's text (`agent/transcripts.py`), rebuilt from
the saved requests and responses: `<out>_texts.csv` has, per run, the task
question, the system prompt and user prompt as the framework sent them, the
model's reasoning from every call, and the full final answer (cells are cut at
32,000 characters for Excel, flagged in `truncated_in_csv`);
`<out>_transcripts.jsonl` has the full text plus a call-by-call trace (the
messages added since the previous call, such as tool results, and the model's
text, reasoning and tool calls). `--no-texts` skips both.

Nothing is filtered or corrected on export. Warm-up runs, framework errors and
approximate fixture matches are all present and flagged by their own columns
(`is_first_in_dir`, `status`, `has_approx_fixture`), so the analysis decides what
to exclude rather than the exporter deciding silently. The printed summary
excludes warm-up and non-completed runs, and says which rows it left out.

## Outputs are timestamped

Nothing overwrites an earlier run. Stamps are US Eastern time (following
daylight saving) whatever the machine's clock is set to, from
`telemetry.file_stamp`; timestamps inside the records stay epoch milliseconds.

- **Harness**: `--out output/runs/pilot` writes to `output/runs/pilot_20260918_193012/`.
- **Export**: `--out output/results.csv` writes `output/results_20260918_193012.csv`.
- **Sweeps**: `output/runs/thinking_sweep_<model>_<stamp>/`, and probe sweeps
  `output/runs/probe_<model>_<stamp>/`.

`--no-timestamp` opts out; the sweep uses it inside its own stamped directory.

## Scoring: did the agent succeed?

`agent/scoring.py` supplies deterministic content checks. `agent/evaluation.py`
combines them with the dedicated semantic-correctness judge into one quality
result for all tasks. Five answer-check modes:

| Mode | Success when | Used for |
| --- | --- | --- |
| `key_facts` | at least `min_hits` facts appear; each fact may list alternatives, so paraphrase passes | corpus and FRED tasks |
| `numeric_tolerance` | a number in the answer is within tolerance, matched on number boundaries | arithmetic tasks |
| `structured` | every typed JSON field matches its gold value, tolerance, alias, and collection ordering | Phase 1 v2 selections, calculations, and evidence-sufficiency decisions |
| `abstain` | the answer declines, and no `forbid` pattern (an invented figure) matches | unanswerable and out-of-scope tasks |
| `judge_only` | scored by the semantic judge | time-sensitive web questions and open-ended explanations |

Phase 1 v2 retains **25 easy / 50 medium / 25 hard**, with **80 deterministic / 20
semantic** checks. All tasks carry the `phase1-v2-p0-1` evaluation policy.

- `semantic_correct` measures the requested conclusions, independently of format
  and termination. `format_ok` measures JSON shape or numeric serialization.
- `success` requires correct content and completed execution. `format_success`
  additionally requires format compliance. Under outcome policy `phase1-v2-p0-2`,
  `strict_success` also requires natural completion; forced finalization, step
  limits and truncation never count as strict successes.
- Correct structured fields in surrounding prose, unambiguous quoted numbers,
  and numeric strings can receive content credit while failing format checks.
  Wrong values still fail content. Unparseable or ambiguous deterministic content
  is `unscorable_format`: content unknown and an explicit unsuccessful outcome,
  without adding paid judge calls. Legacy task checks retain their behavior.
- Semantic tasks use `semantic_correctness`, with a 0.8 threshold and instructions
  to ignore presentation. Missing/stale judgments remain `pending_judge`.
  Faithfulness and task completion remain diagnostic metrics.

Judge-only runs always include `semantic_correctness`, even when an older
command explicitly requests only diagnostic metrics; `--diagnostics-only` is
the explicit opt-out. The judge records `semantic-correctness-v1` and validates
the normalized 0–1 score, threshold and pass flag. DeepEval's raw 0–10 score is
normalized before applying the inclusive 0.8 threshold. Legacy task-completion
scores cannot fill a semantic slot.

`output_cap_reached` is the union of `per_call_output_cap_reached` and
`total_output_cap_reached` (plus unknown length terminations). Export reconstructs
these from call telemetry, so old total-only flags do not hide per-call
truncation. `completion_type` separates natural completion, forced final,
step limit and truncation; `hit_step_limit` now denotes step-limit status only.

The shared proxy enforces the frozen config's call cap and per-call output cap.
`limits.max_total_output_tokens` optionally supplies a tighter run ceiling; zero
means `max_llm_calls * sampling.max_tokens`. Thinking tokens count as output.
Every admitted generation request consumes a call, including failed requests
and forced finalization. Rejections are recorded separately. A valid final stop
on the last permitted call may complete; `length` is truncation. CrewAI's forced
final response is subject to the same rule and retains a separate marker.

Restart the proxy before new runs: the runner requires a `proxy-v1` budget
acknowledgment. New telemetry records framework status, canonical termination,
budget limits, cap flags, missing usage, and blocked requests. Config hashes
include the accounting policy. An elapsed task timeout is classified consistently
and blocks subsequent model requests; in-progress tools/requests are not forcibly
preempted. Different budgets or accounting policies are separate comparison cells.

The harness writes `plan.jsonl` before execution and fingerprints task definitions.
Export writes a companion `.quality.json` report with planned/observed/missing
runs, dataset coverage and pending judgments. Failed and pending runs retain their
agent costs. An interrupted or partially judged run set cannot be labeled a
complete 100-task benchmark. See
[`dataset/reviews/phase1_v2_p0.md`](dataset/reviews/phase1_v2_p0.md) for the full
contract and commands. Original `phase1.json` is unchanged.

One rule keeps live web results out of the deterministic score. `tool_error`
and `unanswerable` tasks carry `judge_if_tools: [web_search]`: a run that
searched the web is left to the judge, like a `judge_only` task, because a
question is only unanswerable if the search really returned nothing relevant.
Runs that did not search keep the `abstain` check, so `check_mode` is decided
per run, and `judge_runs.py --select judge_only` picks up exactly those runs.

Calendar tasks stay deterministic. `fred_releases` computes its week from the
frozen `as_of_date` and is replayed from two recorded responses, so "this week"
is always 2026-04-06 to 04-10 whatever day the benchmark runs. Do not re-record
those two fixtures: FRED revises past schedules, and the reference answers are
written from this recording.

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

DeepEval supplies semantic correctness for judge-only tasks, plus optional
diagnostics for faithfulness, task completion, tool and argument choice.
It scores **stored runs offline**. Runs are never re-executed, and each test
case is built from the proxy's request bodies and the tool layer's rows
(`agent/eval/cases.py`), so every framework is judged from identical input.

```bash
.venv/bin/python scripts/judge_runs.py --runs "output/runs/thinking_sweep_*/*" \
    --select judge_only --metrics semantic_correctness,faithfulness   # --dry-run to preview
```

`--select` takes `judge_only`, `all`, or task types. Results go to each run
directory's `eval.jsonl`, and the pass is resumable: already-scored run/metric/judge/answer-fingerprint entries are
skipped, so a run stopped by usage limits continues where it left off. The CSV
export adds `judge_<metric>` and `judge_<metric>_pass` columns, plus what judging
each run cost (`judge_calls`, `judge_input_tokens`, `judge_output_tokens`).
Use `--tasks dataset/tasks/phase1_v2.json` at judging and export. Old
`task_completion` scores do not substitute for the new semantic metric.
With multiple stored semantic judges, export requires `--judge-id` to select
the primary judge rather than mixing judges across runs.

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

```bash
.venv/bin/python scripts/parity_check.py --config configs/pilot_mock.yaml \
    --tasks dataset/tasks/phase1.json --frameworks langgraph,openai_agents
```

### Known residuals: LangGraph vs OpenAI Agents SDK

Checked against the mock upstream. The system prompt, sampling settings and
message history are identical; these differences remain and are kept rather
than patched around, so the SDK runs as shipped:

| Residual | Effect |
| --- | --- |
| The SDK adds `"strict": false` to every tool definition, with no setting to omit it | `parity_check.py` reports the `tools` field as differing. A chat template may render it into the prompt; the input-token cost is not yet measured on a real model |
| At the step limit, the SDK runs the tool the last call asked for, and LangGraph does not | Both make exactly `max_llm_calls` LLM calls, but a `step_limit` run has one more tool call, and its tool time, on the SDK side |

The SDK names the output limit `max_tokens` where LangGraph sends
`max_completion_tokens`; the adapter sends the latter, so that field matches.

### Known residuals: CrewAI

CrewAI builds its own prompt and offers no supported way to replace it: custom
templates merge everything into a single user message, and `prompt_file` no
longer reaches agent prompts (1.15). The adapter runs one agent through
`Agent.kickoff`, the entry point closest to the other runtimes, and carries the
shared system prompt in the role and backstory fields. Sampling settings and
tool schemas match; these differences remain, checked against the mock upstream:

| Residual | Effect |
| --- | --- |
| System message: one extra space after the first sentence, and a closing line `Your personal goal is: Answer the user's question.` | 52 more characters than the shared prompt; `parity_check.py` reports `system_prompt` as differing |
| The question is sent as `\nCurrent Task: <question>` | A fixed prefix on the user message |
| Every tool carries `"strict": true`, hard-coded | vLLM's default `--tool-strict-level auto` constrains the arguments of strict tools while decoding, so malformed tool calls may be rarer here for a reason that is not the framework's loop. Not yet measured |
| Requests carry `tool_choice: "auto"` | The default, stated explicitly |
| At its iteration budget CrewAI requests a forced final answer | `max_iter = max_llm_calls - 1`; the proxy enforces the same total cap. A valid stop within budget can complete; forced finalization remains visible in `forced_final_answer`, `completion_type` and `termination_reason`, and is excluded from `strict_success`. |
| A call to a tool that does not exist gets CrewAI's own reply, `Tool not found` | LangGraph also answers the model; the SDK aborts the run |
| Tool arguments that are not valid JSON are answered by CrewAI before the tool layer sees them | Not counted in `n_malformed_tool_calls` |

A budget of one LLM call cannot be expressed in CrewAI and is reported as a
`framework_error`. Switched off because the harness could not see them: tool
result caching, agent retries, context-window summarising, console panels and
telemetry.

A Crew with a Task was not used: it appends "This is the expected criteria for
your final answer ... you MUST return the actual complete content as the final
answer, not a summary" to every question, which contradicts the shared prompt's
instruction to answer concisely.

## Status

- Working: config, telemetry, tool layer with record/replay, logging proxy,
  mock upstream, LangGraph ReAct agent, runner, CLI, dataset builder, CSV export.
  First real cell ran against Qwen3.8-27B on vLLM.
- OpenAI Agents SDK ReAct agent: implemented and smoke-tested against the mock
  upstream (matching LangGraph on status, LLM calls, tool calls and tokens),
  then run on the original 34 tasks against Qwen3.8-27B, Nemotron, Llama-3.1-8B
  and Llama-3.3-70B (run code `openAISDK-v1`).
- CrewAI ReAct agent: implemented and smoke-tested against the mock upstream
  with all three frameworks (150 runs, all completed).
- Next: a cell with all three frameworks interleaved; the
  hardware sampler (NVML + RAPL); the analysis notebook.
- Dataset: 50 tasks in `dataset/tasks/phase1.json` — 20 deduplicated from the
  reference ground truth, 30 authored (no-tool, unanswerable, tool-error,
  multi-hop, multi-tool). The last 16, added 2026-10-13, lean hard: arithmetic
  over paper tables, cross-paper questions, false premises and calendar
  reasoning. They have not been run against a model yet. Every task has a reviewed answer check
  (`needs_review: false`), but none is human-verified yet
  (`review.human_verified: false`).
