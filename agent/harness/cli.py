"""Run a benchmark cell.

    python -m agent.harness.cli \
        --config configs/pilot_mock.yaml \
        --tasks dataset/tasks/pilot.json \
        --frameworks langgraph \
        --repeats 1 \
        --out output/runs/smoke

Frameworks are interleaved per task rather than run in blocks, so slow drift in
thermals or cache state cannot favour whichever framework ran first.
"""

from __future__ import annotations

import argparse
import random
import uuid
from dataclasses import replace
from pathlib import Path

from ..adapters import FRAMEWORKS, get_adapter
from ..env import load_env
from ..config import RunConfig
from ..telemetry import PlanRecord, file_stamp
from .runner import Runner, Task


def preflight(cfg) -> None:
    """Fail before the plan starts if the serving stack is not reachable.

    Without this, an unreachable model server produces a framework_error per
    run: the whole plan burns through in seconds and every row is useless. One
    clear message beats 45 identical connection errors.
    """
    import httpx

    control = cfg.base_url.rsplit("/v1", 1)[0] + "/control/health"
    try:
        health = httpx.get(control, timeout=5.0).json()
    except Exception as exc:
        raise SystemExit(
            f"proxy not reachable at {control} ({type(exc).__name__}).\n"
            "  start it with: ATK_UPSTREAM=http://127.0.0.1:8000 ./scripts/start_stack.sh"
        )

    upstream = health.get("upstream", "")
    try:
        resp = httpx.get(f"{upstream}/v1/models", timeout=10.0)
        served = [m["id"] for m in resp.json().get("data", [])]
    except Exception as exc:
        raise SystemExit(
            f"proxy is up but its upstream {upstream} is not answering "
            f"({type(exc).__name__}).\n"
            "  start the model server with:\n"
            "    nohup ./scripts/serve_vllm.sh --config <cfg> "
            "> /tmp/vllm_serve.log 2>&1 &\n"
            "  it takes 2-4 min to load; wait for /v1/models to answer."
        )

    wanted = cfg.model.served_name
    if served and wanted not in served:
        raise SystemExit(
            f"upstream serves {served}, but the config asks for '{wanted}'.\n"
            "  fix model.served_name in the config, or serve the right model."
        )
    if cfg.model.repo != "mock-model":
        _check_serving(cfg, upstream)
        _check_tool_parser(cfg, upstream, wanted)
        _check_thinking(cfg, upstream, wanted)
    effort = f", effort={cfg.model.reasoning_effort}" if cfg.model.reasoning_effort else ""
    print(f"preflight ok: proxy -> {upstream}, serving {wanted}, "
          f"thinking={cfg.model.thinking_setting}{effort}")


# Fields where the running server must match the config exactly.
_SERVING_FIELDS = (
    "served_name", "tp_size", "max_model_len", "tool_call_parser",
    "reasoning_parser", "thinking_setting", "reasoning_effort",
)


def _check_serving(cfg, upstream: str) -> None:
    """Compare the config with what serve_vllm.sh recorded when it launched.

    The thinking probe can tell on from off, but not medium from xhigh, and it
    cannot see the parsers or context length at all. The launch record can.
    """
    import json

    port = upstream.rsplit(":", 1)[-1].strip("/")
    record = cfg.resolve(f"output/serving_{port}.json")
    if not record.exists():
        raise SystemExit(
            f"no launch record at {record}; the server was not started with\n"
            "  serve_vllm.sh --config, so its settings cannot be checked.\n"
            "  restart it: ./scripts/stop_vllm.sh && "
            "nohup ./scripts/serve_vllm.sh --config <cfg> > /tmp/vllm_serve.log 2>&1 &"
        )
    served = json.loads(record.read_text())
    wanted = {f: getattr(cfg.model, f) for f in _SERVING_FIELDS}
    diffs = [
        f"    {f}: server={served.get(f)!r}  config={wanted[f]!r}"
        for f in _SERVING_FIELDS
        if served.get(f) != wanted[f]
    ]
    if diffs:
        raise SystemExit(
            f"the running server does not match this config "
            f"(launched from {served.get('config')}):\n" + "\n".join(diffs) + "\n"
            "  restart vLLM from this config:\n"
            "    ./scripts/stop_vllm.sh && "
            "nohup ./scripts/serve_vllm.sh --config <cfg> > /tmp/vllm_serve.log 2>&1 &"
        )


# Text that means the model emitted a tool call the parser failed to extract.
_RAW_TOOL_MARKERS = ("<tool_call>", "<function=", "<|python_tag|>", '"parameters"', '"arguments"')


def _check_tool_parser(cfg, upstream: str, model: str) -> None:
    """Pilot gate 1, automated: is the configured tool parser right for this model?

    Each model family formats tool calls differently (Qwen and Nemotron use XML,
    Llama 3 uses JSON). With the wrong parser vLLM returns the call as plain
    text, the framework sees an answer instead of a tool call, and every run in
    the cell is quietly wrong. One question that clearly needs a tool settles it.
    """
    import httpx

    from ..tools import schemas

    try:
        resp = httpx.post(
            f"{upstream}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": "Which US economic data releases are scheduled this week? Use the tool.",
                }],
                "tools": schemas.openai_tools(),
                "tool_choice": "auto",
                "max_tokens": 1024,
                "temperature": 0,
            },
            timeout=180.0,
        ).json()
        message = resp["choices"][0]["message"]
    except Exception as exc:
        raise SystemExit(f"tool-parser probe failed ({type(exc).__name__}: {exc})")

    calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    if calls:
        name = calls[0].get("function", {}).get("name", "")
        if name not in schemas.TOOL_NAMES:
            raise SystemExit(f"tool-parser probe: model called unknown tool '{name}'")
        return
    leaked = [m for m in _RAW_TOOL_MARKERS if m in content]
    if leaked:
        raise SystemExit(
            f"tool parser '{cfg.model.tool_call_parser}' did not extract the model's tool call; "
            f"it came back as text containing {leaked}.\n"
            "  read the tool-call format off the model's chat template and set "
            "model.tool_call_parser to match."
        )
    print("  note: tool-parser probe got a direct answer, no tool call; parser unverified")


def _check_thinking(cfg, upstream: str, model: str) -> None:
    """Verify the server's thinking mode matches the config.

    The config only declares a thinking setting; the server's chat-template
    defaults decide what actually happens. A mismatch here inflates every token
    count and contaminates every answer, silently. So one tiny request goes
    straight to vLLM — not through the proxy, to keep it out of telemetry — and
    the response is checked for a reasoning block.
    """
    import httpx

    try:
        resp = httpx.post(
            f"{upstream}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
                "max_tokens": 256,
                "temperature": 0,
            },
            timeout=120.0,
        ).json()
        message = resp["choices"][0]["message"]
    except Exception as exc:
        raise SystemExit(f"thinking probe failed ({type(exc).__name__}: {exc})")

    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    thinking_on = bool(reasoning.strip())
    declared = cfg.model.thinking_setting

    if declared == "off" and thinking_on:
        raise SystemExit(
            "config says thinking_setting=off, but the server is producing reasoning.\n"
            "  restart vLLM from this config:\n"
            "    ./scripts/stop_vllm.sh && nohup ./scripts/serve_vllm.sh --config <cfg> "
            "> /tmp/vllm_serve.log 2>&1 &"
        )
    if declared != "off" and not thinking_on and cfg.model.reasoning_parser:
        raise SystemExit(
            f"config says thinking_setting={declared}, but the server produced no reasoning.\n"
            "  restart vLLM from this config: ./scripts/serve_vllm.sh --config <cfg>"
        )
    if not cfg.model.reasoning_parser and "</think>" not in str(message.get("content", "")):
        # Without a parser, reasoning is indistinguishable from the answer.
        print("  note: no reasoning parser configured; reasoning cannot be separated from content")


def select_tasks(tasks: list[Task], limit: int) -> list[Task]:
    """All tasks, or a sample of `limit` that covers every task type.

    Taking the first N was a trap: the file lists the reference items first,
    and they are all single-tool lookups, so a 15-task pilot never saw a
    no-tool, multi-hop, multi-tool, unanswerable or tool-error task. Round-robin
    across types keeps small samples representative; within a type, file order
    is kept so the same limit always picks the same tasks.
    """
    if not limit or limit >= len(tasks):
        return list(tasks)
    by_type: dict[str, list[Task]] = {}
    for t in tasks:
        by_type.setdefault(t.raw.get("task_type", "unknown"), []).append(t)
    picked: list[Task] = []
    queues = [list(q) for q in by_type.values()]
    while len(picked) < limit and any(queues):
        for q in queues:
            if q and len(picked) < limit:
                picked.append(q.pop(0))
    order = {id(t): i for i, t in enumerate(tasks)}
    return sorted(picked, key=lambda t: order[id(t)])


def parse_mix(text: str) -> dict[str, int]:
    """'easy=10,medium=15,hard=10' -> {'easy': 10, 'medium': 15, 'hard': 10}."""
    out: dict[str, int] = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        level, _, n = part.partition("=")
        if not n.strip().isdigit():
            raise SystemExit(f"--difficulty-mix: expected level=N, got '{part}'")
        out[level.strip()] = int(n)
    return out


def select_difficulty_mix(tasks: list[Task], mix: dict[str, int]) -> list[Task]:
    """A fixed number of tasks per difficulty level, e.g. 10 easy / 15 medium /
    10 hard, chosen like select_per_difficulty (round-robin over task types)."""
    by_level: dict[str, list[Task]] = {}
    for t in tasks:
        by_level.setdefault(t.raw.get("difficulty", "unknown"), []).append(t)
    unknown = set(mix) - set(by_level)
    if unknown:
        raise SystemExit(f"--difficulty-mix: no tasks at level(s) {sorted(unknown)}; have {sorted(by_level)}")
    picked = [t for level, n in mix.items() for t in select_tasks(by_level[level], n)]
    order = {id(t): i for i, t in enumerate(tasks)}
    return sorted(picked, key=lambda t: order[id(t)])


def select_per_difficulty(tasks: list[Task], per_level: int) -> list[Task]:
    """`per_level` tasks from each difficulty level, for a trial run.

    Within a level, the same round-robin across task types as select_tasks,
    so 5 easy / 5 medium / 5 hard still covers the task types. A level with
    fewer tasks contributes all of them.
    """
    by_level: dict[str, list[Task]] = {}
    for t in tasks:
        by_level.setdefault(t.raw.get("difficulty", "unknown"), []).append(t)
    picked = [t for level in by_level.values() for t in select_tasks(level, per_level)]
    order = {id(t): i for i, t in enumerate(tasks)}
    return sorted(picked, key=lambda t: order[id(t)])


def build_plan(tasks: list[Task], frameworks: list[str], repeats: int, seed: int):
    """(task, repeat, framework) triples, interleaved and shuffled within task."""
    plan = []
    rng = random.Random(seed)
    for task in tasks:
        for repeat in range(repeats):
            order = list(frameworks)
            rng.shuffle(order)
            for framework in order:
                plan.append((task, repeat, framework))
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description="Agent tokenomics Phase 1 runner")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--frameworks", default="langgraph")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", required=True,
                    help="run directory; a _YYYYmmdd_HHMMSS suffix is added unless --no-timestamp")
    ap.add_argument("--no-timestamp", action="store_true",
                    help="use --out exactly as given (the sweep does this inside its own stamped dir)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run-code", default="",
                    help="label recorded on every run and in the CSV, saying what "
                         "this execution is for (e.g. openAISDK-v1)")
    ap.add_argument(
        "--append",
        action="store_true",
        help="allow writing into a directory that already holds runs (off by "
             "default: mixing two executions silently corrupts the joins)",
    )
    ap.add_argument("--limit", type=int, default=0,
                    help="sample N tasks, spread across task types (0 = all tasks)")
    ap.add_argument("--stack", default=None, choices=["a", "b"],
                    help="serving stack (agent/stacks.py): its proxy port, and this process "
                         "pinned to its GPU's NUMA node; default $ATK_STACK, else the single stack")
    ap.add_argument("--no-power", action="store_true",
                    help="do not record GPU/CPU power alongside the runs (on by default, "
                         "except in record mode)")
    ap.add_argument("--base-url", default="",
                    help="proxy to use instead of the config's (or the stack's); provenance only, "
                         "not part of config_id. Used by parallel recording workers")
    ap.add_argument("--difficulty-mix", default="",
                    help="tasks per difficulty level, e.g. easy=10,medium=15,hard=10; "
                         "overrides --per-difficulty and --limit")
    ap.add_argument("--per-difficulty", type=int, default=0,
                    help="trial run: N tasks from each difficulty level (e.g. 5 -> 5 easy, "
                         "5 medium, 5 hard), spread across task types; overrides --limit")
    ap.add_argument(
        "--tool-mode",
        choices=["replay", "record", "live"],
        default=None,
        help="override the config. 'record' is a probe run: tools hit live APIs "
             "and every call is saved as a fixture for later replay.",
    )
    args = ap.parse_args()
    if not args.no_timestamp:
        # Every run gets its own directory, so repeated invocations never
        # overwrite or mix with each other.
        args.out = f"{args.out.rstrip('/')}_{file_stamp()}"

    load_env()  # tool API keys, before any tool can need them
    cfg = RunConfig.from_yaml(args.config)
    if args.tool_mode:
        cfg = replace(cfg, tool_mode=args.tool_mode)
    from .. import stacks

    stack = stacks.get(args.stack)
    if stack.name:
        # Pinned before the embedding model loads, so its threads stay on the
        # node next to this stack's GPU.
        stacks.pin_current_process(stack)
        cfg = replace(cfg, base_url=stack.proxy_url, hardware_id=stack.hardware_id,
                      numa_pinning=stack.numa_pinning)
        print(f"stack {stack.name}: proxy {stack.proxy_url}, {stack.hardware_id}, "
              f"pinned to NUMA {stack.numa_pinning}")
    if args.base_url:
        cfg = replace(cfg, base_url=args.base_url)
    if cfg.tool_mode != "replay":
        print(f"!! tool_mode={cfg.tool_mode}: tools hit live APIs, so tool latency "
              f"and content are NOT comparable across runs")
    all_tasks = Task.load_all(args.tasks)
    if args.difficulty_mix:
        tasks = select_difficulty_mix(all_tasks, parse_mix(args.difficulty_mix))
    elif args.per_difficulty:
        tasks = select_per_difficulty(all_tasks, args.per_difficulty)
    else:
        tasks = select_tasks(all_tasks, args.limit)
    mix: dict[str, int] = {}
    for t in tasks:
        mix[t.raw.get("task_type", "unknown")] = mix.get(t.raw.get("task_type", "unknown"), 0) + 1
    levels: dict[str, int] = {}
    for t in tasks:
        levels[t.raw.get("difficulty", "unknown")] = levels.get(t.raw.get("difficulty", "unknown"), 0) + 1
    print("by difficulty: " + ", ".join(f"{k}={v}" for k, v in levels.items()))
    print(f"{len(tasks)} task(s): " + ", ".join(f"{k}={v}" for k, v in sorted(mix.items())))

    frameworks = [f.strip() for f in args.frameworks.split(",") if f.strip()]
    unknown = set(frameworks) - set(FRAMEWORKS)
    if unknown:
        raise SystemExit(f"unknown frameworks: {sorted(unknown)}")

    session = f"sess_{uuid.uuid4().hex[:10]}"
    runners = {
        f: Runner(
            replace(cfg, framework=f),
            get_adapter(f, replace(cfg, framework=f)),
            args.out,
            session=session,
            run_code=args.run_code,
        )
        for f in frameworks
    }

    out_dir = Path(args.out)
    if (out_dir / "run.jsonl").exists() and not args.append:
        n = sum(1 for _ in open(out_dir / "run.jsonl"))
        raise SystemExit(
            f"{out_dir} already holds {n} run(s).\n"
            "  Two executions in one directory interleave their rows and produce\n"
            "  duplicate (task, repeat) pairs that look like real data.\n"
            "  Use a new --out, delete that directory, or pass --append if you mean it."
        )

    preflight(cfg)

    missing = cfg.require_local_models()
    if missing and cfg.model.repo != "mock-model":
        raise SystemExit("missing local weights:\n  " + "\n  ".join(missing))

    # Cold-loading the embedding model costs seconds and would be charged to
    # whichever run the shuffle puts first. Warm it before any timing starts.
    from ..tools import impl

    warm_ms = impl.warmup(cfg)
    print(f"warmed retrieval in {warm_ms:.0f}ms")

    plan = build_plan(tasks, frameworks, args.repeats, args.seed)
    for idx, (task, repeat, framework) in enumerate(plan):
        runner = runners[framework]
        runner.writer.write(PlanRecord(session, runner.config_id, task.task_id, repeat, idx))
    print(f"{len(plan)} runs -> {args.out}"
          + (f"  (run code: {args.run_code})" if args.run_code else ""))
    print(f"prompt_hash={list(runners.values())[0].prompt_hash} "
          f"tool_schema_hash={list(runners.values())[0].schema_hash}")

    sampler = _start_power_sampler(args, cfg)
    fixture_totals: dict[str, int] = {}
    for idx, (task, repeat, framework) in enumerate(plan):
        rec = runners[framework].run_task(task, repeat, idx)
        for kind, n in runners[framework].last_fixture_matches.items():
            fixture_totals[kind] = fixture_totals.get(kind, 0) + n
        print(
            f"[{idx + 1}/{len(plan)}] {framework:14s} {task.task_id:12s} "
            f"r{repeat} {rec.status:14s} "
            f"llm={rec.n_llm_calls} tools={rec.n_tool_calls} "
            f"tok={rec.input_tokens}/{rec.output_tokens} "
            f"e2e={rec.e2e_ms:7.0f}ms orch={rec.t_orch_ms:6.0f}ms"
            + (f" err={rec.error[:60]}" if rec.error else "")
        )

    if fixture_totals:
        total = sum(fixture_totals.values())
        print("\nfixture matches: " + "  ".join(
            f"{k}={v} ({v / total:.0%})" for k, v in sorted(fixture_totals.items())
        ))
        approx = fixture_totals.get("nearest", 0)
        if approx:
            print(f"  {approx} call(s) matched approximately; exclude or report them separately")

    if sampler is not None:
        # A few more samples, so the trace extends past the last run's end.
        import time

        time.sleep(0.5)
        sampler.terminate()
        sampler.wait(timeout=10)
    _release_proxy(cfg, session)
    print(f"\nwrote {Path(args.out).resolve()}")
    return 0


def _start_power_sampler(args, cfg):
    """Record GPU and CPU power into <out>/power.jsonl while the runs execute.

    The export reads it to fill energy_gpu_j / energy_cpu_j, and
    scripts/attribute_energy.py splits it into LLM, tool and orchestration
    time. Skipped in record mode, where runs overlap and timings do not count.
    """
    import os
    import subprocess
    import sys

    if args.no_power or cfg.tool_mode == "record":
        return None
    root = Path(__file__).resolve().parent.parent.parent
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(root / "scripts" / "power_sampler.py"),
             "--until-pid", str(os.getpid()), "--out", str(Path(args.out) / "power.jsonl")],
            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
    except OSError as exc:
        print(f"power sampler not started: {exc}")
        return None
    # Wait for the first samples, so the trace covers the first run too.
    import time

    trace = Path(args.out) / "power.jsonl"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if trace.exists() and trace.read_text().count("\n") >= 2:
            break
        time.sleep(0.1)
    print(f"recording power -> {trace}")
    return proc


def _release_proxy(cfg, session: str) -> None:
    """Drop the proxy session lock so the next run can bind."""
    import httpx

    try:
        httpx.post(
            cfg.base_url.rsplit("/v1", 1)[0] + "/control/release",
            json={"session": session},
            timeout=5.0,
        )
    except Exception:
        pass  # lock expires on its own; not worth failing a finished run


if __name__ == "__main__":
    raise SystemExit(main())
