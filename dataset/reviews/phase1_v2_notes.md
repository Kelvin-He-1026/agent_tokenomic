# Phase 1 v2: 100 tasks

`dataset/tasks/phase1_v2.json` is a runnable list of task objects for the current harness. It contains **25 easy, 50 medium, and 25 hard** tasks, with reference answers, evidence summaries, PDF page/fixture references and SHA-256 hashes, solution steps, capability tags, and structural difficulty profiles.

The set retains 25 baseline designs and reworks or adds 75 more demanding tasks. There are 16 tasks targeting dependent retrieval, 51 requiring multiple source files (46 involving multiple papers), 57 requiring calculation, 29 involving conditional decisions, and 43 tagged for incomplete-evidence handling. Tags overlap. Thirty tasks require a supported partial answer rather than either inventing a complete answer or refusing the answerable portion.

Medium tasks target level 3. Hard tasks target level 4 through constrained selection, feasibility checks, Pareto comparisons, allocation, amortization, and evidence audits. No task is labeled as level-5 recovery: the current tools do not implement controlled per-task failures. Repeated calls to one tool count as one distinct tool. Independently named-source lookups are not labeled dependent retrieval merely because they require multiple calls.

The five local PDFs and the two existing calendar fixtures supply the evidence; no new web fixtures or live APIs are required. Hypothetical workloads, thresholds, and trace values are explicitly supplied in the questions and are not presented as reported experiments. All calendar questions use the existing frozen **2026-04-06** configuration and watchlist, rather than current real-world schedules.

## Grading

**80 tasks are deterministic and 20 need the semantic judge.** The 25 easy / 50 medium / 25 hard mix is unchanged.

| Check | Tasks | Behavior |
| --- | ---: | --- |
| `structured` | 59 | Match all typed JSON fields, numeric tolerances, named relationships, exact set membership, and ordered results. |
| `numeric_tolerance` | 8 | Content and scalar-only format are separate; recover only an unambiguous standalone number. |
| `key_facts` | 13 | Retain direct-lookup baseline checks. |
| `judge_only` | 20 | Keep open-ended explanations and design comparisons for semantic evaluation. |

All 100 questions now carry the `phase1-v2-p0-1` evaluation contract, documented
in [phase1_v2_p0.md](phase1_v2_p0.md). Content correctness, format compliance,
and execution completion are distinct dimensions. Deterministic contracts
check required fields and relations; format checks assess keys and public types
without consulting gold values. Numeric strings and unique standalone numbers
can preserve content credit while failing format. Ambiguous or malformed content
is explicitly unscorable rather than silently classified as a reasoning error.
These deterministic checks do not prove that arbitrary surrounding prose is
faithful; their content claim is limited to the requested contract fields.

The semantic subset remains `medium-011` through `medium-020`, `medium-042`
through `medium-050`, and `hard-021` (all with the `phase1-v2-` prefix). Use the
new `semantic_correctness` metric to fill their unified quality results;
`faithfulness` and `task_completion` remain separate diagnostics. No additional
semantic tasks were introduced. Missing judgments, stale task definitions and
multiple unresolved judge choices remain visible in coverage reports.

Original natural-language questions/reference explanations are retained as
metadata on converted tasks. Re-run edited questions for a new benchmark;
new telemetry fingerprints task definitions and refuses to silently score a
changed definition. Previous benchmark outputs are not modified.

## Running, judging, and exporting

Use the new task path at **all three stages**; those commands otherwise default to the original 50-task file. With the normal serving stack already available:

```bash
.venv/bin/python -m agent.harness.cli \
  --config configs/pilot_qwen.yaml \
  --tasks dataset/tasks/phase1_v2.json \
  --repeats 1 --out output/runs/phase1_v2

.venv/bin/python scripts/judge_runs.py \
  --runs 'output/runs/phase1_v2_*' \
  --tasks dataset/tasks/phase1_v2.json \
  --select judge_only --metrics semantic_correctness,faithfulness

.venv/bin/python scripts/export_csv.py \
  --runs 'output/runs/phase1_v2_*' \
  --tasks dataset/tasks/phase1_v2.json \
  --out output/phase1_v2_results.csv
```

Run all 100 tasks to obtain the exact 25/50/25 mix. The existing `--limit` selector samples by task type, not difficulty, so a limited pilot will not necessarily preserve these proportions. Framework/config selection remains the normal harness behavior.

## Validation and remaining calibration

[phase1_v2_validation.json](phase1_v2_validation.json) records counts, hashes, and checks. Validation includes unique IDs/questions, compatibility with the actual loader/sampler, all 80 deterministic reference-answer self-checks, absence of trivial key-fact leakage, valid dependency ordering, arithmetic checks, and independent parsing of the source tables. In particular, ManuRAG's context-table columns were explicitly reordered from **MCQ, MathQ, RQ** when joining them to category-based results.

Run the offline grading regressions with `.venv/bin/python -m unittest discover -s tests -v`. They exercise swapped relations, missing subanswers, extra or duplicated set members, numeric units and rounding, unknown evidence, aliases, duplicate JSON keys, contradictory prose, non-finite numbers, and original-dataset compatibility.

[phase1_v2_retrieval_checks.json](phase1_v2_retrieval_checks.json) records 12 CPU-only queries against the unchanged frozen index. All returned the expected source paper. These are source-presence checks, not proof that every returned chunk contains all needed facts or that the intended retrieval depth is unavoidable.

The difficulty labels are authored structural targets. Human review, an end-to-end model/framework pilot, and empirical calibration remain to be done. Related table/paper families intentionally support different reasoning operations; keep related examples together if dividing development and held-out calibration sets. The original `dataset/tasks/phase1.json`, tool implementations, supplied YAML configurations, and prior benchmark outputs are unchanged. The shared runtime now enforces and records the P0 accounting contract; restart the proxy before a new benchmark.

## Web search (revised 2026-10-16)

Web search is now a valid tool on every task that uses a tool. The 83 corpus
tasks accept `[retrieve]`, `[web_search]` or `[retrieve, web_search]`; the 10
calendar tasks accept `[fred_releases]` or `[fred_releases, web_search]` (the
calendar is frozen at 2026-04-06, so the web can add context but not replace
it); the 7 no-tool tasks are unchanged. Questions and answer checks are
unchanged: the gold values are the papers' own figures, whichever tool found
them. This is a deliberate choice, and it changes what the corpus tasks
measure: facts about these well-known papers can be found online, so a task
may be solved by web lookup instead of retrieval. `tools_used` in the export
shows which route each run took.

Web results are replayed from recorded fixtures like everywhere else, so each
model / framework / thinking-mode cell needs a record-mode probe run
(`--tool-mode record`) before its benchmark run; a query without a recording
fails as a tool error in replay. The dataset hash in
`phase1_v2_validation.json` predates this change.
