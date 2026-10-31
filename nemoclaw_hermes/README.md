# nemoclaw_hermes

Agent-tokenomics cell for **NemoClaw Hermes**: the Hermes agent running in its
NemoClaw/OpenShell sandbox, measured on the same model, server and GAIA tasks as
the framework cells in the parent folder, and through the same logging proxy.

Hermes is measured **as shipped**: its own system prompt, tools and loop. The
frameworks share one prompt and one tool layer; Hermes cannot, so it is a
separate arm rather than a fourth framework under parity. What it shares with
them is the endpoint, the model, the call and token budget, the sampling and the
scorer, which is what makes tokens, latency and cost per success comparable.

```
configs/gaia_hermes_qwen_tp2.yaml   the cell: model block = GAIA cell, plus a hermes: block
configs/*_think_low_load.yaml       the concurrency-test cell (3600 s task deadline)
hermes_plugins/bench-spans/         Hermes plugin: timestamps tool calls for the harness
scripts/setup_sandbox.sh            once per sandbox: bench profile, plugin, attachments, snapshot, manifest
scripts/run_cell.sh                 serve, proxy, run, export
scripts/run_load.sh                 serve, proxy, concurrency sweep per framework, export
manifests/hermes_sandbox.json       Hermes/NemoClaw/OpenShell versions, image digest, policy presets
manifests/install_nemoclaw.sha256   checksum of scripts/install_nemoclaw.sh
output/                             run directories, logs, CSVs
```

The harness code lives in the parent folder with the other runtimes:
`agent/hermes/oneshot_agent.py` (the adapter), stack `h` in `agent/stacks.py`,
and the `hermes:` config block in `agent/config.py`.

## How Hermes's traffic reaches the proxy

The sandbox's recorded inference route is `http://host.openshell.internal:8000/v1`,
and OpenShell delivers it to `172.18.0.1:8000` (the `openshell-docker` bridge
gateway). Stack `h` puts the proxy exactly there and moves vLLM to loopback:

| | address |
|---|---|
| vLLM | `127.0.0.1:8002` (unreachable from the sandbox) |
| proxy | `172.18.0.1:8000` (sandbox and harness) |

So the sandbox's route, provider attachment and network policy stay as NemoClaw
set them, and no Hermes call can bypass the proxy. The proxy is not exposed on
the LAN.

## One run

`HermesOneShotAgent.run` executes, in the sandbox via `openshell sandbox exec`
on the sandbox's recorded gateway:

1. kill any Hermes process a timed-out run of the same slot left, restore the
   `bench` profile from `/sandbox/.bench/bench.tar`, and write the run id into
   its `X-Run-Id` header;
2. `cd /sandbox/gaia_docs && hermes -p bench -t web,file,code_execution --yolo -z "<question>"`;
3. print the answer, Hermes's `--usage-file`, and the run's timestamps.

The exec goes to OpenShell directly rather than through `nemohermes <sandbox>
exec`, which holds a host-wide lock while the command runs (no two runs could
be in flight) and adds ~1 s per run. Same user, directory, environment and
policy. Runs before 2026-10-28 (hermes-v1, -v2) went through `nemohermes`, so
their `T_Orch` is ~1 s higher.

**Timing.** The `bench-spans` plugin records each tool call's start and end
inside the sandbox; they become `tool_call` rows (`fixture_match` = `agent`), so
`t_tool_ms` is measured rather than folded into `T_Orch` (it was, up to v2).
The rest of each run is in `span.jsonl` as named orchestration: `exec_launch`,
`profile_restore`, `hermes_boot`, `agent_setup` (plugins loaded to first model
request), `loop` (between calls), `hermes_exit`, `exec_return`.

**Concurrent runs** (the load test) each get a slot: their own copy of the
profile (`bench-s<k>`), usage file and kill pattern. The run id travels in the
`X-Run-Id` header because the sandbox's L7 policy only permits the plain `/v1`
paths.

## The bench profile

`setup_sandbox.sh` clones the sandbox's default profile into `bench` (the
NemoClaw-managed default config is never edited) and changes:

| Setting | Why |
|---|---|
| `memory.memory_enabled`, `memory.user_profile_enabled` = false | Memory would carry answers between repeats |
| `curator.enabled` = false, `skills.creation_nudge_interval` = 0 | No skill creation or maintenance |
| `agent.max_turns` = `max_llm_calls - 1` | Hermes's final call fits inside the proxy's cap |
| `auxiliary.title_generation.enabled` = false | The title call runs **concurrently** with the first agent call, which the proxy refuses as `concurrency_violation`, ending the run |
| provider `extra_body` = cell sampling (`pin_sampling: true`) | Hermes sends no temperature/top_p/seed; without this vLLM uses Qwen's generation defaults while the frameworks send temperature 0 |
| `web.search_backend`/`extract_backend` = `hermes.web_backend` (keenable), `web.provider_tier.firecrawl` = paid (`hermes.web_disable`) | See **Web access** below |
| provider `extra_headers` = `X-Run-Id: __ATK_RUN_ID__` | Placeholder each run replaces with its run id, so the proxy files concurrent runs' calls correctly |
| `plugins/bench-spans`, `plugins.enabled` += bench-spans | Tool-call timing (see **One run**); inert unless `HERMES_BENCH_SPANS` is set |

The profile is snapshotted after setup and restored before every run, so no run
sees another's sessions or state. Toolsets per run come from `hermes.toolsets`;
memory, session_search, skills, delegation and cron are left out.

## Running

```bash
./scripts/setup_sandbox.sh                       # once, and after any config change
# hermes-v2: thinking on at low effort, on the 30 tasks hermes-v1 sampled
./scripts/run_cell.sh --name gaia_hermes_v2 --config configs/gaia_hermes_qwen_tp2_think_low.yaml \
    --tasks "$PWD/configs/gaia30_tasks.json" --frameworks hermes,langgraph --repeats 2 \
    --tool-mode record --run-code hermes-v2
```

`configs/gaia30_tasks.json` holds the 30 GAIA tasks (10 easy, 10 medium,
10 hard) that `--difficulty-mix easy=10,medium=10,hard=10` drew for hermes-v1,
so later cells run the same questions. A config with a different path makes
`run_cell.sh` restart vLLM; the thinking mode is a server setting.

Stop afterwards from the parent folder: `ATK_STACK=h ./scripts/stop_stack.sh --all`.

### Concurrency test

```bash
./scripts/setup_sandbox.sh configs/gaia_hermes_qwen_tp2_think_low.yaml   # after pulling these changes
nohup ./scripts/run_load.sh --name load_v1 --frameworks hermes,langgraph --levels 1,4,8,16,32 \
    --exclude gaia-384d0dd8,gaia-5d0080cb,gaia-676e5e31,gaia-72e110e7,gaia-872bfbb1,gaia-ad2b4d70 \
    --tool-mode record --run-code load-v1 > output/load_v1.driver.log 2>&1 &
```

One closed-loop sweep per framework (`agent/harness/load.py` in the
parent folder), stopping at the first level that breaks a rule against
the single-user level: P95 latency over 2x, success more than 5 points lower,
or infrastructure errors and timeouts over 5%. The `--exclude` list is the six
tasks whose GAIA answers Hermes found online in v2. The load config's 3600 s
task deadline keeps the latency rule usable: at one user, Hermes's slowest runs
already reach 1800 s. Per framework: `output/<name>_<fw>_<stamp>_levels.csv`
(one row per level), the run CSVs, and `_stages.csv` with time, energy and CPU
per run stage for the single-user level.

The host has 2x Xeon 6740P since 2026-10-29: 48 cores per socket with SMT (192
CPUs), GPU0 on NUMA node 0 and GPU1 on node 1, so the test uses every CPU by
default. `--cpus-per-socket N` still pins it to N CPUs per socket, GPU-local node
first: vLLM (restarted if it runs on other CPUs), the proxy, the harness, the
OpenShell gateway and the sandbox container (`docker update --cpuset-cpus`; the
gateway and the container get their previous CPUs back on exit). The power
sampler keeps to CPU 0. Levels record the set in `cpuset`.

## Known residuals and limits

| Residual | Effect |
|---|---|
| Hermes's own system prompt (~9.5k chars) + 7 tool schemas + NemoClaw runtime context appended to the user message | ~5.1k prompt tokens before the question, resent every call. This is the main finding to measure (`system_prompt_tokens`, `other_prompt_tokens` in the calls CSV) |
| `config` rows record the shared prompt's hash | Hermes's real prompt is in the request blobs and `_texts.csv` |
| Tools run inside the sandbox, not the shared tool layer | `n_tool_calls` is counted from the proxy (calls the model requested); tool time is in `T_Orch` |
| Hermes's web tools hit the live web through OpenShell policy | Not replayable; the frameworks use recorded fixtures in replay mode, or live in record mode |
| Context compression is left on (Hermes default, 50% of 128k) | Its summarization calls are real cost and appear as extra LLM calls |
| `--yolo` | No approval prompts; the OpenShell policy is the boundary |
| Hermes can still reach the call cap (`agent.max_turns` 29 gave 30 calls, then a 31st) | The proxy refuses it with 429 and the run ends `step_limit`/`call_cap` like any framework's; Hermes retries the 429 three times, which adds a little wall time, not tokens |
| Blocked web: the model probes the sandbox instead (ports, `auth.json`) | Seen in the smoke test; one such run ended when OpenShell dropped a request carrying credential placeholders (`framework_error`, "Connection error") |

**Web access:** Hermes's web toolset uses keyless third-party services. The
custom preset `configs/policies/hermes-keyless-web.yaml` opens only those hosts
(`api.keenable.ai` search/fetch, `mcp.exa.ai`, `search.parallel.ai`), for Hermes's
Python only; keyless Firecrawl stays blocked and Hermes falls back past it.
GAIA questions and extracted URLs are sent to these services. Applied with:

```bash
nemohermes gaia-hermes policy add --from-file configs/policies/hermes-keyless-web.yaml
# remove: nemohermes gaia-hermes policy remove hermes-keyless-web
```

Left unpinned, Hermes's keyless tier round-robins each search across Exa,
Parallel, Firecrawl and Keenable, and moves to the next vendor only on a
rate-limit error. In the `hermes-v1` run (unpinned, since deleted)
18% of searches failed outright:

- `Keyless Firecrawl search failed: 403 Forbidden` (280): the sandbox policy
  denies `api.firecrawl.dev`, and a 403 is not a rate limit, so there was no
  failover.
- `Keyless Exa search failed: Unrecognized MCP response shape` (27): an
  occasional Exa reply without a text result. It doesn't reproduce on the same
  queries.

The bench profile therefore pins search and extract to Keenable
(`hermes.web_backend`), with Exa and Parallel as rate-limit failover, and takes
Firecrawl out of the ring (`hermes.web_disable`). Both settings are in the
`hermes:` block, so they are part of `config_id`.

The frameworks search through Tavily/DuckDuckGo instead, so search engine is a
difference between the arms, not only the agent.
