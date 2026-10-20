# Phase 1 v2: metric definitions and checks (2026-10-19)

Answers to the P0–P2 review list. Definitions live in
`dataset/policies/phase1_v2.json`; this note records what was checked and how.

## P0 — Unified task success

Every run carries four independent outcome fields (export columns):

| Field | Meaning |
| --- | --- |
| `semantic_correct` | The answer's content is right (deterministic check, or the semantic judge for judge-only tasks), whatever its format or how the run ended |
| `format_ok` | The reply followed the requested format (bare JSON / single number); blank where no format is specified |
| `success` | Run completed **and** `semantic_correct`. Format does not gate it |
| `strict_success` | `success` **and** `format_ok` **and** a natural finish (no forced final answer, step limit or truncation) |

Rank models and efforts on `success`; report `format_ok` and `strict_success`
beside it. A semantically right answer in the wrong format counts toward
`success` but not `strict_success`. Structured answers are graded on the last
JSON object in the reply (`agent/scoring.py`), so a right answer wrapped in
prose is not lost.

## P0 — Ground truth and judge

* All 80 deterministic reference answers pass their own checks, and no question
  passes on its own (`scripts/make_gt_review.py`).
* Human sign-off is still needed: `dataset/reviews/phase1_v2_gt_review.csv`
  lists all 100 tasks with question, reference answer, what the check accepts,
  judge rubric and evidence, and empty `reviewer` / `approved` / `notes` columns.
* The judge is pinned (`judge` in the policy file): Claude Code, model
  `claude-opus-5-5`, effort `high`, DeepEval GEval `semantic_correctness`
  (version `semantic-correctness-v1`), each task's `grading_rubric` verbatim,
  threshold `evaluation.semantic_threshold` = 0.8 for all 20 judge-only tasks.
  `scripts/judge_runs.py` defaults to this model. The export uses exactly one
  `judge_id`; faithfulness and task completion are diagnostics only.

## P0 — GPU energy method

`energy_gpu_j` is the driver's cumulative energy counter
(`nvmlDeviceGetTotalEnergyConsumption`) at run end minus run start, per GPU,
summed over both GPUs; it is not power × time. The export now also has
`energy_gpu0_j`, `energy_gpu1_j` and `energy_method`. Checked on the 1,200-run
Qwen TP=2 sweep (`scripts/verify_energy.py`, `output/energy_verification_qwen.json`):

| Check | Result |
| --- | --- |
| Counter vs independent integral of sampled power (runs ≥ 2 s) | median 0.5 % apart, p95 2.6 % |
| Counter vs fixed median power × time | median 2.1 %, p95 24 % apart |
| Both GPUs counted | GPU0 share median 49 %; no run with a zero GPU |

The fixed-power estimate is close for most runs only because both GPUs sit
near a constant ~363 W under TP=2; it misses by up to a quarter on others.
`energy_gpu_j` is gross (idle draw included); `scripts/attribute_energy.py
--idle-w` gives energy above idle.

## P1 — CPU energy

RAPL package counters (both sockets) are now readable, and the harness records
power for every benchmark run (`<run folder>/power.jsonl`). The export fills
`energy_cpu_j` and `energy_total_j` (= GPU + CPU) from it. Runs before
2026-10-19 have no CPU energy; it cannot be recovered. The read permission
resets at reboot; see the README for the udev rule that keeps it.

## P1 — Failure classification

New column `failure_category`, one label per run, by precedence:
`step_limit` > `output_truncation` > `timeout` > `framework_error` >
`pending_judge` > `success` > `format_failure` (content unscorable because of
format) > `tool_failure` (wrong, with a failed tool call) > `wrong_answer`.

## P1 — Reasoning effort

The effort reaches the model. Each mode's vLLM launch records its template
kwargs (`low`/`medium`/`xhigh`), and Qwen3.8's chat template turns them into
instructions: `low` adds "Keep your thinking brief…" (+26 prompt tokens on the
first call), `xhigh` adds "Please think carefully…" (+38), and `medium` adds
nothing (the template's neutral setting). The model's response to `low` is
weak: median reasoning tokens per run 986 (low) vs 1,021 (medium), not
significantly different (Mann-Whitney p = 0.70); `xhigh` is clearly higher
(1,400). This is model behaviour, not a configuration error.

## P2 — Prompt composition

`system_prompt_tokens`, `conversation_history_tokens`, `tool_response_tokens`
and `current_user_prompt_tokens` count message text with the served model's
tokenizer. `other_prompt_tokens` is the rest of the server's `prompt_tokens`:
the chat template's role markers and special tokens, the tool definitions
(names, descriptions, JSON schemas) the template renders, and any
template-added instruction (such as the effort line above). The five sum to
`prompt_tokens` exactly.

## P2 — Execution traces

`<out>_transcripts.jsonl` keeps every run's full prompt, reasoning, answer and
call-by-call trace, never cut. `<out>_texts.csv` is the same per run for
reading, with cells cut at 32,000 characters (Excel's limit) and flagged in
`truncated_in_csv`. Analyse from the CSVs; go to the JSONL for full text.
