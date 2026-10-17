# Phase 1 difficulty review

Reviewed 15 October 2026. This is a structural assessment of all 50 current questions, their reference answers/checks, tool schemas, frozen calendar responses, fixture matching, task loading and sampling. Selected arithmetic and variant claims were also checked against the local PDFs. No model benchmark or exhaustive retrieval experiment was run.

**Finding: the current set has no convincing level-4 analysis or level-5 replanning task.** Several hard questions are direct extractions, one-table calculations, or unsupported requests. The current labels are 5 easy / 30 medium / 15 hard. Under the explicit mapping below, the reviewed labels are **46 easy / 4 medium / 0 hard**. This measures task structure, not how often a particular model succeeds.

| Score | Interpretation | Current questions | Proposed v2 slots |
| --- | --- | ---: | ---: |
| 1 | Direct extraction, common knowledge, rewriting | 22 | 10 |
| 2 | Independent aggregation, simple calculation/filtering/premise checks | 24 | 5 |
| 3 | Dependent evidence lookup or cross-source/table synthesis | 4 | 20 |
| 4 | Constrained analysis across several evidence stages | 0 | 10 |
| 5 | Level-4 work plus controlled recovery and a changed plan | 0 | 5 |

Tier mapping: **easy = 1–2; medium = 3; hard = 4–5**. This mapping is a proposed convention, not part of the supplied table. If easy is mapped to 1 and medium to 2–3 instead, the same scores give 22 easy / 28 medium / 0 hard. The conclusion that there is no structurally hard coverage is unchanged. Mean ordinal score: 1.64/5; it is descriptive, not a calibrated interval-scale statistic.

## How the rubric was applied

Your table supplies anchor profiles, not a unique formula. I used profile-based ordinal judgments rather than inventing weights and summing unlike quantities. A task can use one tool repeatedly and still require serious synthesis; three unrelated tools can still do independent lookups. The numeric dimensions support each judgment, while the rationale records evidence layout and synthesis demands that six counts alone cannot represent.

| Field | Operational definition |
| --- | --- |
| Dependencies (D) | Result-dependent transitions between evidence acquisitions. Discovering an entity and then querying it counts as one. Two named papers looked up independently count as zero. Arithmetic is recorded separately. |
| Tools (T) | Distinct tool names in the selected reasonable route. Two calendar windows are one tool and two calls. Alternative routes are not added together. |
| Branches (B) | Evidence-dependent filtering, availability/premise decisions, or choices affecting the next action or final answer. Ordinary tool routing is excluded. |
| Distractors (X) | Specific misleading premises or confusable evidence classes such as dev/test values or similar variant names. Incidental retrieval noise is not assumed. |
| Calculation (C) | Binary: arithmetic, unit conversion or date-window calculation is necessary. Set operations belong under branches. |
| Recovery (R) | Binary: a controlled failure or unusable result forces a changed plan. A false premise, refusal, unsupported schema value or accidental fixture miss is insufficient. |

Level 1 is direct work without a meaningful decision or calculation. Level 2 adds simple processing or independent subtasks. Level 3 adds a genuine discovered-entity lookup, a dependent two-tool comparison, or quantitative synthesis across sources/tables. Level 4 needs several interdependent evidence stages and substantive conditional decisions. Level 5 also requires reproducible recovery. The user anchors are retained verbatim as vectors in the JSON review; they are examples, not tool-count quotas.

Counts and levels below are estimates of a reasonable evidence-grounded route. They are not observed traces or proven lower bounds. In particular, a broad top-5 retrieval can shortcut an intended dependency; all four level-3 assessments have medium confidence until that is checked. Different tasks with the same six-count vector may receive different levels because a single table and a cross-table join impose different evidence work.

## Scores for all 50 current questions

The spreadsheet includes the full question, all dimensions, estimated call ranges, confidence, rationale and grading flags for every row. IDs below are the exact IDs in the runnable dataset.

| Task ID | Current label | D | T | B | X | C | R | Score | Proposed tier | Reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `agentic_rag_gt-001` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | Extract failure modes and the update process from ACE. Multiple answer fields do not create an evidence dependency. |
| `agentic_rag_gt-002` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | Definition and cited methods are direct extraction; the question does not name its paper. |
| `agentic_rag_gt-003` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | The architecture and reported gains are stated together in the Transformer abstract. |
| `agentic_rag_gt-004` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | Extract a computational advantage and its stated tradeoff from one discussion. |
| `agentic_rag_gt-005` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | Read the two retrieval decisions and the limitation of using only the initial input. |
| `agentic_rag_gt-006` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | Read DRAGIN shortcomings and the proposed information-needs framing. |
| `agentic_rag_gt-007` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | Extract the collection method and dataset features from HotpotQA. |
| `agentic_rag_gt-008` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | Describe reasoning types already explained in the paper; the agent does not have to perform that multi-hop reasoning. |
| `agentic_rag_gt-009` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | List manufacturing data modalities and summarize the stated retrieval problem. |
| `agentic_rag_gt-010` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | Extract two multimodal strategies and their domain limitation. |
| `agentic_rag_gt-011` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | One this_week calendar call returns the schedule. |
| `agentic_rag_gt-012` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | One next_week calendar call returns the three releases. |
| `agentic_rag_gt-013` | medium | 0 | 1 | 1 | 0 | 0 | 0 | 2 | easy | Retrieve one schedule and filter rows by Tier 1. |
| `agentic_rag_gt-014` | medium | 0 | 1 | 1 | 1 | 0 | 0 | 2 | easy | Filter the calendar by report category and handle absent employment reports; publisher classification may need corroboration. |
| `agentic_rag_gt-015` | medium | 0 | 1 | 1 | 0 | 0 | 0 | 2 | easy | Filter one calendar by inflation/GDP relevance. |
| `agentic_rag_gt-016` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | One web lookup, with no requested reconciliation or analysis. |
| `agentic_rag_gt-017` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | One web lookup for a market snapshot. |
| `agentic_rag_gt-018` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | An open-ended summary can be copied from one search response; novelty is not operationalized. |
| `agentic_rag_gt-019` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | One web lookup and a short summary of prices/news. |
| `agentic_rag_gt-020` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | One web lookup for a single statistic. |
| `authored-notool-001` | easy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | easy | Expand a common acronym without external evidence. |
| `authored-notool-002` | easy | 0 | 0 | 0 | 0 | 1 | 0 | 2 | easy | Multiply three supplied quantities: 150 * 3 * 5. |
| `authored-notool-003` | easy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | easy | Rewrite the supplied sentence. |
| `authored-notool-004` | easy | 0 | 0 | 0 | 0 | 0 | 0 | 1 | easy | State a standard conceptual distinction. |
| `authored-notool-005` | easy | 0 | 0 | 0 | 0 | 1 | 0 | 2 | easy | Multiply the supplied throughput by the supplied success rate. |
| `authored-unanswerable-001` | medium | 0 | 1 | 1 | 1 | 0 | 0 | 2 | easy | Reject an unsupported benchmark premise after searching the corpus. |
| `authored-unanswerable-002` | medium | 0 | 1 | 1 | 1 | 0 | 0 | 2 | easy | Distinguish reported hardware/time from an unreported exact dollar cost. |
| `authored-unanswerable-003` | medium | 0 | 1 | 1 | 0 | 0 | 0 | 2 | easy | Search for an unsupported metric and abstain. One top-k miss cannot establish absence across all five papers. |
| `authored-multihop-001` | medium | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | The paper identity and self-attention rationale can be retrieved together; no result-dependent query is inherently needed. |
| `authored-multihop-002` | hard | 0 | 1 | 0 | 0 | 0 | 0 | 2 | easy | Independently collect problems from two papers and attribute them; neither lookup requires the other result. |
| `authored-multitool-001` | medium | 0 | 2 | 0 | 0 | 0 | 0 | 2 | easy | Two unrelated subtasks using retrieve and fred_releases; no dependency joins them. |
| `authored-multitool-002` | hard | 1 | 2 | 1 | 1 | 0 | 0 | 3 | medium | Use the paper definition to shape a web query, distinguish unrelated meanings of context collapse, and compare recency. |
| `authored-toolerror-001` | medium | 0 | 0 | 1 | 0 | 0 | 0 | 2 | easy | The schema already rules out week-after-next. A correct no-call limitation explanation is accepted; no failure is injected. |
| `authored-toolerror-002` | medium | 0 | 1 | 1 | 0 | 0 | 0 | 2 | easy | Search the corpus and report missing financial data or ask which company; the company is unspecified. |
| `authored-multihop-003` | hard | 0 | 1 | 0 | 1 | 1 | 0 | 2 | easy | Read three training rows from one table and add them; exclude dev/test rows. |
| `authored-multihop-004` | hard | 0 | 1 | 0 | 1 | 1 | 0 | 2 | easy | Use one split table to form the training denominator and calculate train-hard percentage. |
| `authored-multihop-005` | hard | 0 | 1 | 0 | 1 | 1 | 0 | 3 | medium | Join parameter counts from Table 3 with matched test BLEU scores from Table 2, then compute a ratio and a difference. |
| `authored-multihop-006` | hard | 0 | 1 | 0 | 1 | 1 | 0 | 2 | easy | Select the correct model/language cells in one comparison table and divide scientific-notation values. |
| `authored-multihop-007` | hard | 0 | 1 | 0 | 1 | 1 | 0 | 2 | easy | Use the hardware/schedule paragraph to convert days to hours and divide; ignore the rounded per-step time. |
| `authored-multihop-008` | hard | 1 | 1 | 0 | 0 | 0 | 0 | 3 | medium | Discover the benchmark in DRAGIN, then use that identity to retrieve its original paper and dataset size. |
| `authored-multihop-009` | hard | 0 | 1 | 0 | 1 | 1 | 0 | 3 | medium | Retrieve independent totals from two papers, disambiguate exact versus rounded counts, and divide. |
| `authored-multihop-010` | hard | 0 | 1 | 0 | 0 | 0 | 0 | 1 | easy | All four facts are stated in one ACE passage; answer length does not create multi-hop reasoning. |
| `authored-multihop-011` | hard | 0 | 1 | 0 | 1 | 0 | 0 | 2 | easy | Distinguish similar variants in one methods section and map each to its property. |
| `authored-multihop-012` | medium | 0 | 1 | 1 | 0 | 0 | 0 | 2 | easy | Call the same tool with two known windows and union Tier 1 rows. The calls are independent. |
| `authored-multihop-013` | medium | 0 | 1 | 1 | 0 | 0 | 0 | 2 | easy | Call both known windows and intersect release identities; associate each date with the shared release. |
| `authored-lookup-001` | hard | 0 | 1 | 1 | 0 | 1 | 0 | 2 | easy | Map an explicit date to next_week using the frozen date, then filter the returned rows. |
| `authored-lookup-002` | hard | 0 | 1 | 1 | 1 | 0 | 0 | 2 | easy | Retrieve the training configuration and BLEU, reject the supplied false hardware/time premise, and correct it. |
| `authored-toolerror-003` | hard | 0 | 0 | 1 | 0 | 1 | 0 | 2 | easy | Map the explicit dates to an unsupported window and explain the schema limit without a tool failure. |
| `authored-unanswerable-004` | hard | 0 | 1 | 1 | 1 | 0 | 0 | 2 | easy | Check a complete calendar response and reject the premise that the jobs report appears this week. |
| `authored-notool-006` | medium | 0 | 0 | 0 | 0 | 1 | 0 | 2 | easy | Add supplied input/output tokens, then divide by success rate. |

## Why the current structure is shallow

- 22 tasks are labeled single_tool_lookup and 13 multi_hop, but only 2 use distinct tools in their intended routes: authored-multitool-001 and authored-multitool-002.
- Only two intended routes have a result-dependent evidence transition: authored-multihop-008 (discover a dataset, then retrieve its paper) and authored-multitool-002 (use a paper definition to shape/disambiguate a web search). This excludes optional detours that a different route might take.
- authored-multihop-010 asks for four facts stated together in one ACE passage: level 1. authored-multihop-003/004/006/007 are mostly one-table or one-paragraph arithmetic: level 2. authored-multihop-012/013 call known calendar windows independently: level 2.
- The existing tool_error tasks do not install a controlled failure. The week-after-next request can be declined from the schema; the unspecified-company request yields missing information rather than a guaranteed tool error. Every current task therefore has R=0.
- Several calendar questions reuse the same answer-bearing rows. The split-sum/percentage questions reuse the same table, and the two unsupported-calendar tasks are near duplicates. Keep these as deliberate within-family contrasts if useful, but do not count them as independent topic coverage.
- Difficulty can come from PDF extraction failure or weak retrieval rather than intended reasoning. Record retrieval sufficiency and table readability separately; do not promote a task to hard just because a model cannot retrieve the relevant number.

## Answer-check problems verified locally

I called the existing `agent.scoring.score` with targeted answers. These are deterministic answer-check results; separate ToolCorrectness/faithfulness judge metrics may still detect the mistakes. The full answers, outputs and source hashes are in [phase1_scoring_probes.json](phase1_scoring_probes.json).

| Task | Probe | Current answer check | Needed correction |
| --- | --- | --- | --- |
| authored-multitool-002 | Gives the corpus summary and omits all web discussion | Pass | Score both subtasks, source recency and semantic match. |
| authored-multihop-011 | Swaps the two variants and their properties | Pass | Check variant-to-property relations, not the presence of both names. |
| authored-multihop-013 | Places the second date next to a different release | Pass | Require the same release identity mapped to both dates. |
| agentic_rag_gt-014 | Adds a fabricated jobs release to industrial production | Pass | Check supported entries and explicit absence within the watchlist. |
| authored-multihop-009 | Gives 74.5 despite the integer-rounding instruction | Pass | Specify exact counts and validate an integer final answer. |
| authored-multihop-003 | Rejects 90,564 and asserts 90,000 | Pass | Evaluate the final claimed value, not any number appearing anywhere. |
| authored-toolerror-001 | Correctly declines and repeats the requested April 20–24 dates | Fail | Do not forbid the requested dates when they appear in a limitation explanation. |
| authored-unanswerable-003 | Generic refusal with no tool evidence | Pass | Validate justified uncertainty/evidence coverage separately from refusal wording. |

For the ratio task, the exact 112,779 / 1,515 is approximately 74.4416 and rounds to **74**. Using the abstract’s rounded 113k gives approximately 74.5875 and rounds to **75**. The current question/check tolerate both inputs, plus noninteger answers. The proposed question fixes the input precision.

These checks show that adding reasoning steps to a question will not ensure the benchmark measures them. A harder task needs complete output checks and evidence/trajectory checks appropriate to its capability.

## Reproducibility and harness constraints

- The system date is frozen to 2026-04-06. Calendar ground truth is based on the existing two recorded watchlist responses, not the real-world schedule at run time. Describe absence as absence from this watchlist; it is not proof that no such real-world release exists.
- Web fixtures were recorded later and contain different time references. For example, `fixtures/web_search/a76b0027d5c2747a.json` describes an April 29 decision, after the frozen April 6 date; `fixtures/web_search/d74fc9b0c01801c1.json` answers a relative-date question with an October/November schedule. These are observations about fixture text, not verified claims about real events. “As of” tasks must align evidence availability with the system date. A recording timestamp is not the source publication date.
- Free-text fixture matching uses exact, normalized, then approximate token overlap. Arbitrary misses become tool errors. Unplanned misses/approximate hits can confound difficulty and recovery; report their rates and validate intended routes before scoring.
- The current CLI samples by `task_type`, not difficulty. The builder carries authored difficulty labels through, but does not compute structural profiles. A final restructured set needs joint difficulty/capability sampling and difficulty fields in analysis exports.
- `phase1.json` is generated from reference sources, authored tasks and reviewed checks. Make future task changes in those inputs and build a versioned dataset. Existing export code defaults to looking up task definitions from the current `dataset/tasks/phase1.json`; overwriting that file can change how historical stored runs are rescored. Preserve the original set and record the exact task-file hash for new runs.

## Recommended restructuring

Use **15 easy / 20 medium / 15 hard**, with 10 direct baselines, 5 simple processing tasks, 20 genuine evidence-synthesis tasks, 10 constrained-analysis tasks and 5 controlled-recovery tasks. Retain the useful no-tool/routing/premise baselines; replace redundant lookups and unrelated two-part questions. Keep capability tags (lookup, dependent hop, quantitative synthesis, negative evidence, recovery) separate from difficulty.

The complete [50-slot authoring blueprint](../proposals/phase1_v2_blueprint.md) contains actual question drafts, intended routes, required checks, evidence families and missing fixtures. Its [JSON version](../proposals/phase1_v2_blueprint.json) is deliberately an object outside `dataset/tasks/`, so it cannot be mistaken for a runnable list of validated tasks. Levels in the blueprint are targets; a candidate must be downgraded or redesigned if the actual tool environment permits a shortcut.

Do not force all three current tools into hard tasks. The local research corpus and economic calendar serve different domains; connecting unrelated questions adds calls without adding coherent reasoning. One-tool analysis and two-tool recovery are valid if the evidence dependency graph warrants the level.

Promotion order: repair relation/completeness checks; pin temporal evidence; validate medium tasks against real retrieval; author constrained analyses with gold intermediate values; implement shared deterministic recovery scenarios; then pilot and calibrate the difficulty labels. Use identical task scenarios, tools, budgets and grading across frameworks. Keep failure/answer correctness and tool/trajectory correctness as separate reported metrics, and combine them explicitly if reporting a stricter task-success measure.

For calibration, inspect success rate, minimum sufficient tool calls, observed dependency depth, recovery completion and tokens per successful task. Choose empirical cutoffs only after the pilot; a structurally easy hallucination check can still have a low model pass rate. Preserve structural and empirical labels separately.

## Validation performed

All 50 source IDs and question hashes match the JSON review and CSV, and the active task-file hash is unchanged. The blueprint has 50 unique slots with the stated 15/20/15 target mix. All 45 deterministically checked reference answers pass their existing checks; the remaining five are `judge_only`. No question by itself passes its own deterministic answer check. The eight recorded scoring probes reproduce the outcomes above. No model benchmark or new retrieval/failure environment was run.

## Files and scope

- [phase1_difficulty.csv](phase1_difficulty.csv): all 50 scored questions for spreadsheet review.
- [phase1_difficulty.json](phase1_difficulty.json): rubric, dimensions, confidence, per-task rationales and source hashes.
- [phase1_scoring_probes.json](phase1_scoring_probes.json): observed scorer counterexamples.
- [phase1_v2_blueprint.md](../proposals/phase1_v2_blueprint.md) and [JSON](../proposals/phase1_v2_blueprint.json): 50 draft authoring specifications, not validated benchmark tasks.

Reviewed source SHA-256: `b9682c65e3cc2278e3b5710c57f5c744e4ebb3e9815c13a011c4f1ebec29191d`. The active dataset, builder sources, scorer, tool implementations, configurations and existing outputs were left unchanged. The draft hard/recovery tasks are not ready to run until their listed evidence, checks and environment support are implemented and validated.
