# Phase 1 v2 P0 evaluation contract

Revision: `phase1-v2-p0-1`. The dataset remains 100 tasks: 25 easy, 50 medium,
25 hard; 80 deterministic checks and 20 semantic checks. Each task now declares
its correctness source, output format and accounting policy in `evaluation`.
The machine-readable policy is [../policies/phase1_v2.json](../policies/phase1_v2.json).

Outcome accounting revision: `phase1-v2-p0-2`; semantic metric revision:
`semantic-correctness-v1`. These revise scoring/accounting without changing the
frozen task definitions or invalidating their stored trajectories.

## One quality result for all tasks

`agent/evaluation.py` combines deterministic checks and the dedicated
`semantic_correctness` judgment. Both sources populate the same columns:

| Column | Meaning |
| --- | --- |
| `semantic_correct` | Required conclusions are correct, incorrect, or unknown. Assessed independently of termination and formatting. |
| `format_ok` | Required JSON keys/types or scalar-only numeric format are satisfied; blank when no machine format is specified. |
| `execution_completed` | Canonical run status is `completed`. |
| `success` | Content correct and execution completed. Format alone does not gate this value. |
| `format_success` | `success` plus format compliance, where applicable. |
| `strict_success` | `format_success` and natural completion. Forced finalization, step limit and truncation are excluded for every framework. |
| `completion_type` | Natural completion, forced final, step limit, truncated, timeout, error, or unknown completion. |
| `score_source` | Deterministic check, semantic judge, or zero-call empty-answer decision. |
| `score_state` | `scored`, `pending_judge`, `ambiguous_judge`, `unscorable_format`, or `task_mismatch`. |

For the example gold result 2250:

| Answer / run | Content | Format | Success | Strict success |
| --- | --- | --- | --- | --- |
| `2250`, naturally completed | true | true | true | true |
| `2250`, forced final | true | true | true | false |
| `2250 runs`, completed | true | false | true | false |
| `2251`, completed | false | true | false | false |
| `2250`, truncated | true | true | false | false |
| `2250 or 2200`, completed | unknown | false | false | false |

Structured answers preserve exact value/alias/tolerance, relation, membership
and ordering checks. Formatting checks do not use gold values or expected
array lengths. Missing subanswers and wrong relations remain content errors.
Unambiguous numeric strings may preserve content credit while failing JSON
types. Extra keys fail format while requested contract fields are scored on
their values. Whole-answer JSON code fences are accepted. Arbitrary prose is
not reconstructed into JSON; malformed outer objects, duplicate keys and
ambiguous numbers cannot be salvaged by selecting a favorable fragment.

Unknown content due to malformed formatting remains an explicit coverage gap.
It is not reported as evidence of incorrect reasoning, and receives no verified
success credit. This keeps all tasks in the benchmark without increasing paid
semantic evaluation beyond the existing 20 tasks. Deterministic field checks
do not independently validate arbitrary surrounding prose, source faithfulness
or an internal reasoning process; use the separate faithfulness diagnostic
where needed. Legacy answer checks retain their existing opt-in behavior.

## Semantic judging and provenance

The 20 semantic tasks remain medium 011–020, medium 042–050 and hard 021.
The new GEval metric assesses substantive correctness against the reference
answer, frozen reference context and task rubric. Its explicit evaluation steps
ignore presentation, avoid a separate step-generation call, and use a recorded
0.8 pass threshold. This threshold is authored and still needs empirical
calibration. `faithfulness`, `task_completion` and tool metrics remain
diagnostics; they cannot silently fill a semantic-correctness slot.

DeepEval emits a raw 0–10 score and normalizes it to 0–1. Threshold checks are
inclusive: 7.9/10 fails and 8/10 passes at 0.8. Records must contain a valid
normalized score, matching threshold, metric revision and consistent pass flag;
otherwise they remain pending and are eligible for rejudging. Older commands
that request only diagnostics now automatically include semantic correctness
for judge-only tasks. Use `--diagnostics-only` to explicitly opt out.

The runner records `judge_id`, `evaluation_version`, `metric_version`, threshold and `answer_fingerprint`.
Resume checks distinguish judge configurations and changed answers/rubrics.
Old task-completion verdicts are not reinterpreted as the new metric. Missing
or stale judgments stay pending. If multiple judges have current semantic
verdicts, export requires `--judge-id` to choose the primary result. Empty
answers are failures without a model call. A correct partial answer may retain
content credit while its truncated/limited execution fails overall success.

## Shared resource accounting

Caps come from one frozen `RunConfig` shared across frameworks in a comparison:

- `limits.max_llm_calls`: maximum admitted generation requests. Upstream errors,
  retries and forced finalization consume requests; rejected requests are
  counted separately and do not fabricate generated tokens.
- `sampling.max_tokens`: per-call completion ceiling, including reasoning.
  The proxy normalizes request spelling/value across frameworks and records
  requested and effective caps.
- `limits.max_total_output_tokens`: optional tighter total completion ceiling.
  Zero derives `max_llm_calls * sampling.max_tokens`. Each subsequent request
  gets the smaller of the per-call ceiling and the remaining run budget.
- `limits.task_timeout_s`: reject new generation after the deadline and classify
  elapsed overruns as timeout. In-progress tools/model requests are not forcibly
  preempted; request timeout remains independently configured.

`output_cap_reached` now summarizes any output-cap hit, with separate
`per_call_output_cap_reached` and `total_output_cap_reached` flags. A single
8192-token `length` termination below an81920-token total sets the per-call
and summary flags, not the total flag. Export re-derives these from stored calls
so historical total-only flags cannot conceal per-call truncation.

The supplied model pilot configurations retain 10 calls and 1024 output tokens
per call, giving a default ceiling of 10,240 output tokens. These values are an
experimental condition, not a guarantee of adequate room for every thinking
mode. Larger-cap experiments must use the same caps across compared frameworks;
the exporter separates differing caps, timeouts and accounting policies.

All new runs require the proxy's `proxy-v1` budget acknowledgment. Missing usage,
cap violations and missing telemetry are explicit accounting failures. The
runner retains both framework status and canonical termination. A valid `stop`
on the last admitted call may complete; `length` is truncation; an outstanding
tool request is a call-cap outcome only at the cap. CrewAI's forced final
response consumes the same budget and uses the same completion rule, with
`forced_final_answer` retained separately. It is classified as `forced_final`
and excluded from `strict_success`, while content correctness remains visible. No framework receives a free final
generation request.

## Complete benchmark reporting

The harness writes `plan.jsonl` before execution. New run records fingerprint
their task definitions. Export does not silently rescore a changed task against
its old trajectory, and writes a companion `.quality.json` report containing
planned and observed counts, missing/unexpected/duplicate runs, task coverage,
pending judgments, semantic coverage and format failures. A complete quality
result requires all loaded dataset tasks, a matching completed plan and resolved
judgments. Resource-accounting completeness is a separate flag.

CSV/console costs retain failed and pending agent-run tokens and elapsed time;
judge costs stay separate. Confirmed success rates are provisional while
judgments are pending. Historical runs without plans or accounting provenance
remain inspectable but cannot be labeled a verified complete benchmark by this
report. Existing outputs are not rewritten.

Restart the proxy with the updated code, then run, judge and export with the
same dataset path:

```bash
.venv/bin/python -m agent.harness.cli \
  --config configs/pilot_qwen.yaml --tasks dataset/tasks/phase1_v2.json \
  --frameworks langgraph,crewai,openai_agents --repeats 1 \
  --out output/runs/phase1_v2_p0

.venv/bin/python scripts/judge_runs.py \
  --runs 'output/runs/phase1_v2_p0_*' --tasks dataset/tasks/phase1_v2.json \
  --select judge_only --metrics semantic_correctness,faithfulness

.venv/bin/python scripts/export_csv.py \
  --runs 'output/runs/phase1_v2_p0_*' --tasks dataset/tasks/phase1_v2.json \
  --out output/phase1_v2_p0.csv --calls
```

Offline checks: `.venv/bin/python -m unittest discover -s tests -v`. Tests use
synthetic judgments and a scripted upstream, not evidence of actual model
quality or judge accuracy. Human review, judge calibration and a full real-model
100-task benchmark remain necessary before publishing comparative results.
