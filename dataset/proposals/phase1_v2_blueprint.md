# Phase 1 v2: 50-slot authoring blueprint

**Design proposal only.** These are question drafts and validation specifications, not a runnable replacement dataset. The current benchmark is preserved. Target mix: **15 easy / 20 medium / 15 hard**, with scores **10 × 1, 5 × 2, 20 × 3, 10 × 4, 5 × 5**.

All relative dates refer to the frozen 2026-04-06 configuration. Future scheduled dates may appear in a schedule available by that date; future observed data or later commentary may not. Archived sources must support the actual claim. Keep related evidence families together in development/evaluation partitions.

Target levels are conditional on the real retrieval environment. Every draft needs exact gold answers, source evidence and validated checks before promotion; higher levels also need an audited dependency graph. A longer question or a requirement to make unnecessary calls does not establish difficulty.

E01–E15 retain baseline designs (with wording/check repairs where needed). M01–M04 develop the four strongest current tasks. Other slots introduce new questions. H01–H10 target constrained analysis; R01–R05 additionally require new controlled failure scenarios. Shared failure infrastructure is not implemented by this proposal.

## Before promoting any slot

1. Author ground truth from exact local PDF passages/tables or dated official archives, including units, entity relations and formulae. Never copy an unverified web summary into ground truth.
2. Validate the minimum sufficient route with the actual tools and top-k=5. If one response answers the whole task, downgrade or redesign rather than prescribing unnecessary calls.
3. Draw the evidence dependency graph, annotate each real branch and distractor, and re-score with the same rubric. Target levels are not validated scores.
4. Pin all relative dates to config.as_of_date; use only evidence published/available by that cutoff. A fixture recording timestamp is not the source publication date.
5. For R01-R05 implement controlled shared failure scenarios and sufficient archived fallbacks; verify framework parity and fresh reset per run. This blueprint does not implement them.
6. Require final-answer correctness and evidence/trajectory correctness separately. Recoveries need trajectory checks, not refusal phrases.
7. Update authored task sources and reviewed checks before building a versioned new tasks file. Do not directly edit the generated phase1.json or silently change historical run scoring.
8. Pilot every candidate across the chosen models/frameworks. Use pass rate, dependent depth, tool calls, recovery success and tokens per success to calibrate structural labels.
9. Keep related evidence families together in development/test partitions and report family-level coverage. Do not count near-duplicate arithmetic questions as independent evidence of breadth.
10. Sample by difficulty and capability jointly; the current CLI samples by task_type only. Ensure any new selector/export preserves framework parity and reports exact dataset version/hash.

## Easy: direct baselines and simple processing

### E01 · target 1/5

What does the acronym RAG stand for in the context of language models? Answer in one short sentence.

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `basic_concepts`.

Based on: `authored-notool-001`.

Authoring requirements: Repair the answer check and verify evidence before promotion.

### E02 · target 1/5

Rewrite this sentence in plain English, without adding information: 'The orchestration layer imposes non-trivial overhead on end-to-end agent execution latency.'

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `rewriting`.

Based on: `authored-notool-003`.

Authoring requirements: Repair the answer check and verify evidence before promotion.

### E03 · target 1/5

In one sentence, what is the difference between a model's weights and its activations?

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `basic_concepts`.

Based on: `authored-notool-004`.

Authoring requirements: Repair the answer check and verify evidence before promotion.

### E04 · target 1/5

ACE says prior context adaptation methods often suffer from two specific failure modes. What are they, and what modular process does ACE use to avoid them?

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `ace_failure_modes`.

Based on: `agentic_rag_gt-001`.

Authoring requirements: Repair the answer check and verify evidence before promotion.

### E05 · target 1/5

What core architectural change does the Transformer make relative to earlier sequence transduction models, and what training and quality gains does the abstract report?

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `transformer_abstract`.

Based on: `agentic_rag_gt-003`.

Authoring requirements: Repair the answer check and verify evidence before promotion.

### E06 · target 1/5

According to the paper, what two decisions define dynamic RAG, and why is single-round retrieval not enough for complex multi-step or long-form tasks?

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `dragin_motivation`.

Based on: `agentic_rag_gt-005`.

Authoring requirements: Repair the answer check and verify evidence before promotion.

### E07 · target 1/5

How is HOTPOTQA collected, and what two additions does the paper highlight beyond simply asking multi-document questions?

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `hotpot_collection`.

Based on: `agentic_rag_gt-007`.

Authoring requirements: Repair the answer check and verify evidence before promotion.

### E08 · target 1/5

What kinds of data make manufacturing QA difficult according to the paper, and why does that make information retrieval harder?

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `manurag_modalities`.

Based on: `agentic_rag_gt-009`.

Authoring requirements: Repair the answer check and verify evidence before promotion.

### E09 · target 1/5

What major economic data releases are scheduled for this week?

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `calendar_week_apr06`.

Based on: `agentic_rag_gt-011`.

Authoring requirements: Repair the answer check and verify evidence before promotion.

### E10 · target 1/5

Using the supplied archived web snapshot valid as of 6 April 2026, report the latest published US unemployment rate and its reference month. Include the release date and source.

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `unemployment_snapshot`.

Based on: `agentic_rag_gt-020`.

Authoring requirements: Repair the answer check and verify evidence before promotion. Create a dated archive whose contents were knowable by the as-of date.

### E11 · target 2/5

A benchmark runs 150 tasks on each of 3 frameworks with 5 repeats per task. How many runs is that in total? Give just the number.

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `benchmark_arithmetic`.

Based on: `authored-notool-002`.

Authoring requirements: Repair the answer check and verify evidence before promotion.

### E12 · target 2/5

An agent run uses 3,000 input tokens and 300 output tokens, and 80% of runs succeed. Charging the failed runs to the successful ones, how many tokens does one successful task cost on average? Give just the number.

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `benchmark_arithmetic`.

Based on: `authored-notool-006`.

Authoring requirements: Repair the answer check and verify evidence before promotion.

### E13 · target 2/5

Which economic releases are scheduled for 16 April 2026?

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `calendar_week_apr13`.

Based on: `authored-lookup-001`.

Authoring requirements: Repair the answer check and verify evidence before promotion.

### E14 · target 2/5

The Attention Is All You Need paper trained its big model on 16 GPUs for a week. What BLEU score did that model reach on WMT 2014 English-to-German?

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `transformer_training`.

Based on: `authored-lookup-002`.

Authoring requirements: Repair the answer check and verify evidence before promotion.

### E15 · target 2/5

In the supplied calendar watchlist for 6-10 April 2026, is an Employment Situation release listed? If so, give its day; otherwise state that it is absent from this watchlist.

Route: Keep a short baseline route; use only the question when no external evidence is required, otherwise one source/tool response.

Required checks: Check every requested fact and its relation; for a no-tool task check both answer correctness and tool abstention.

Evidence family: `calendar_week_apr06`.

Based on: `authored-unanswerable-004`.

Authoring requirements: Repair the answer check and verify evidence before promotion.


## Medium: dependent evidence and synthesis

### M01 · target 3/5

In the Transformer paper, how many times more parameters does the big model have than the base model, and how many BLEU points higher does the big model score on WMT 2014 English-to-German? Give the ratio to one decimal place.

Route: Retrieve Transformer Table 3 parameters and Table 2 test BLEU; join the same base/big models; compute ratio and difference.

Required checks: Require model identifiers, 65M/213M parameters, 27.3/28.4 test BLEU, ratio 3.3, and difference 1.1 with units; reject dev scores.

Evidence family: `transformer_tables`.

Based on: `authored-multihop-005`.

### M02 · target 3/5

One of the datasets the DRAGIN paper uses in its experiments is introduced by another paper in the knowledge base. Which dataset is it, and how many question-answer pairs does that paper say it contains?

Route: Retrieve DRAGIN datasets; discover HotpotQA; retrieve the introducing paper and its exact total.

Required checks: Require benchmark identity, evidence from both papers, and distinguish 112,779 exact from 113k rounded.

Evidence family: `dragin_hotpot`.

Based on: `authored-multihop-008`.

Authoring requirements: Check whether a single broad retrieval shortcuts the intended dependency.

### M03 · target 3/5

Using the exact valid-example total in HotpotQA and the total QA count across all three ManuRAG datasets, compute the HotpotQA-to-ManuRAG ratio. Show both counts and round the ratio to the nearest whole number.

Route: Retrieve two paper totals and divide 112779 by 1515.

Required checks: Require counts 112779 and 1515 plus an integer result of 74; do not accept 74.5 or use the rounded abstract count.

Evidence family: `hotpot_manurag_counts`.

Based on: `authored-multihop-009`.

### M04 · target 3/5

Explain context collapse as defined by ACE. Then find one archived public discussion published after the paper and no later than 6 April 2026 that addresses the same iterative-context problem. Compare its mechanism with ACE, and exclude unrelated social-media meanings. If no eligible source exists in the archive, say so.

Route: Retrieve paper definition and version date; use these to search dated web evidence; disambiguate the phrase and compare.

Required checks: Score corpus and web halves, publication cutoff, semantic match, source citation, and justified absence separately.

Evidence family: `ace_context_collapse`.

Based on: `authored-multitool-002`.

Authoring requirements: Archive eligible and ineligible dated web sources; pin the ACE version date.

### M05 · target 3/5

For the base and big Transformers, calculate GPU-hours per million parameters using the hardware and schedule section and the parameter-count table. Which configuration uses fewer GPU-hours per million parameters, and by what factor? Show the inputs and formula.

Route: Join hardware/time with parameter counts, convert days/hours and normalize by parameters.

Required checks: Check model-to-value mappings, eight GPUs, both durations, normalized units, and the final comparison.

Evidence family: `transformer_training_tables`.

### M06 · target 3/5

Identify the DRAGIN evaluation dataset whose introducing paper supplies sentence-level supporting facts. From that paper, calculate the percentage of its exact valid examples assigned to all training splits combined. Show the three training counts and exact total.

Route: Discover dataset in DRAGIN; retrieve HotpotQA split table; select training rows and compute their share.

Required checks: Require evidence from both papers, 90564/112779 denominator selection, formula and percentage with stated rounding.

Evidence family: `dragin_hotpot`.

### M07 · target 3/5

Using ManuRAG's factual-correctness results, rank its four variants by their unweighted mean across MathQ, MCQ and RQ. For the best variant, retrieve and explain how visual content enters the index.

Route: Retrieve results table, calculate means and rank, then use the winning identity to query the methods section.

Required checks: Require all four means, exact metric/row mappings and the winner-to-mechanism relation.

Evidence family: `manurag_results_methods`.

### M08 · target 3/5

From the Transformer paper, select the base or big configuration that achieves at least 27.0 test BLEU on WMT 2014 English-to-German with at most 100 million parameters. What is its reported training FLOP cost, and how much lower is that cost than the other configuration's? Show the constraints and ratio.

Route: Join test BLEU/FLOPs and parameter tables; apply two constraints; calculate the cost factor.

Required checks: Check test versus dev scores, parameter limit, selected model, FLOP units, and ratio direction.

Evidence family: `transformer_tables`.

### M09 · target 3/5

Find the production agent ACE compares with on AppWorld and the model powering it. Then check the archived official agent documentation available by 6 April 2026. Does that archive corroborate the same model version? Cite both sources or explain the mismatch.

Route: Discover agent/model in ACE, then use the identity to retrieve official dated web documentation.

Required checks: Check identity, exact version, as-of cutoff, and agreement/disagreement supported by two sources.

Evidence family: `ace_appworld_archive`.

Authoring requirements: Create and verify a dated official-documentation snapshot.

### M10 · target 3/5

Identify one named multimodal embedding approach cited by ManuRAG. Using its archived official model card available by 6 April 2026, determine whether it jointly embeds text and images, and explain how that matches or differs from ManuRAG's caption-before-indexing variant.

Route: Discover an embedding model in the paper; query its official model card; retrieve the relevant ManuRAG variant and compare representations.

Required checks: Require source-backed model identity, capability, exact variant mapping and comparison of indexing inputs.

Evidence family: `manurag_embedding_archive`.

Authoring requirements: Freeze official model-card evidence and define acceptable cited models.

### M11 · target 3/5

Across both archived calendar weeks, identify the only release that repeats. For that release, find its issuing agency in the archived official documentation and report the two scheduled dates and the number of calendar days between them.

Route: Retrieve two known calendar windows; intersect identities; use the discovered release to find its official agency; calculate interval.

Required checks: Require one repeated release, both date-to-release relations, agency attribution, and interval.

Evidence family: `calendar_repeat_agency`.

Authoring requirements: Archive official release/agency documentation.

### M12 · target 3/5

Find the earliest Tier 1 release day across the two archived calendar weeks. For every Tier 1 release on that day, use archived official documentation to name the publishing agency. Give the shared date and time, and distinguish the different release identities.

Route: Retrieve both schedules, select earliest eligible day, and search agencies using discovered release identities.

Required checks: Require both releases, correct agency mapping, common date/time and no later Tier 1 releases.

Evidence family: `calendar_earliest_agency`.

Authoring requirements: Archive agency documentation for the selected releases.

### M13 · target 3/5

Using the archived calendar, identify the next-week price-index release. From the archived official release available as of 6 April 2026, report its latest already-published reference period and distinguish that from the future calendar release date.

Route: Discover release identity in next_week; query the matching dated official data release; separate scheduled future publication from available observation.

Required checks: Check index identity, reference period, actual publication cutoff, scheduled date and absence of future observations.

Evidence family: `calendar_price_index_archive`.

Authoring requirements: Freeze official pre-cutoff release evidence and the future schedule.

### M14 · target 3/5

Identify the model ACE used in its AppWorld comparison, then consult the archived official model card to determine whether the version is described as open weight or open source. Quote the relevant license designation and distinguish it from ACE's characterization.

Route: Discover exact model version in paper; search its official archived card/license; reconcile terminology.

Required checks: Require model version, source license designation and evidence-supported wording; do not infer a license from the phrase open source alone.

Evidence family: `ace_model_archive`.

Authoring requirements: Archive the exact model-card/license version; author the terminology check.

### M15 · target 3/5

For ManuRAG's MathQ evaluation, calculate the factual-correctness improvement from standard RAG to the best ManuRAG variant, both in percentage points and relative percent. Then identify that variant's visual-processing method from the methods section.

Route: Find MathQ metric winner, compute two different improvements and retrieve the winning variant mechanism.

Required checks: Check MathQ rows, metric type, percentage points versus relative percent, and mechanism relation.

Evidence family: `manurag_results_methods`.

### M16 · target 3/5

Among the non-Transformer systems in the Transformer's WMT 2014 English-to-German comparison, identify the highest-BLEU system. Compare its FLOP cost with both Transformer configurations, and use the parameter table to state which Transformer is below 100 million parameters.

Route: Select the baseline by test score; compare matched cost cells; join parameters from another table.

Required checks: Require baseline identity, correct language column, both cost ratios and parameter-to-model mapping.

Evidence family: `transformer_tables`.

### M17 · target 3/5

The Transformer paper says a particular design helps counteract the reduced resolution from attention averaging. Identify that design, then find its base-model setting in the architecture table and compare it with the big-model setting. Give the numerical ratio and explain which setting is being compared.

Route: Discover multi-head attention from explanatory passage; retrieve the corresponding architecture variable; compare base/big values.

Required checks: Require h rather than another dimension, both values, ratio and link to the stated drawback.

Evidence family: `transformer_architecture`.

### M18 · target 3/5

Find which benchmark DRAGIN shares with an indexed paper that introduces comparison questions. In that benchmark paper, distinguish the two test settings and calculate the total number of test examples across them, stating whether they are the same test questions.

Route: Discover HotpotQA, retrieve split table and benchmark-setting explanation, calculate total and resolve split identity.

Required checks: Check both settings, row counts, sum, and the paper-supported distinction between test sets.

Evidence family: `dragin_hotpot`.

### M19 · target 3/5

Identify the paper that describes generation, reflection and curation as context-update roles. Find its AppWorld production-agent comparator, then use the dated official comparator archive to verify the model version. Report any source disagreement rather than choosing silently.

Route: Discover ACE from the process; discover comparator/model; retrieve the matching dated official archive.

Required checks: Require paper, comparator, exact model version and a supported agreement/disagreement statement.

Evidence family: `ace_appworld_archive`.

Authoring requirements: Freeze official comparator documentation and verify the chain cannot be answered from one retrieved passage.

### M20 · target 3/5

For each next-week calendar release, classify its publishing organization using the archived official release pages. Which releases are published by a Federal Reserve organization rather than a federal statistical agency, and what are their scheduled dates and times? Include the evidence for each classification.

Route: Retrieve next_week identities; search official publisher pages; classify and map selected releases back to schedule rows.

Required checks: Check all publisher identities, classification, selected release/date/time relations and no fabricated employment release.

Evidence family: `calendar_publishers`.

Authoring requirements: Archive official publisher metadata for all next-week releases.


## Hard: constrained analysis

### H01 · target 4/5

Build a constrained shortlist from the Transformer's English-to-German results. Keep systems with test BLEU at least 26, training cost at most 1e20 FLOPs, and an explicitly reported parameter count at most 100 million. Treat missing parameters as unknown rather than zero. Choose the lowest-cost eligible system, calculate its BLEU gap and FLOP savings versus the highest-BLEU system, and explain every exclusion.

Route: Retrieve results and parameter tables, join system identities, branch on missing versus reported values, apply constraints, select and calculate comparisons.

Required checks: Validate full eligibility table, unknown values, language/test split, chosen system and two derived comparisons.

Evidence family: `transformer_constrained_selection`.

Authoring requirements: Author gold table joins and verify at least two substantive decisions; downgrade if one response exposes the whole result.

### H02 · target 4/5

Choose a ManuRAG variant for a workload containing 50% MathQ, 30% MCQ and 20% RQ. Exclude variants with context recall below 30% in either MathQ or MCQ. Among the survivors, maximize weighted factual correctness; break ties by mean semantic similarity. Explain the winner's indexing mechanism and show why each other variant was excluded or ranked lower.

Route: Retrieve recall table, filter candidates, join correctness/similarity table, compute weighted objective and conditional tie-break, retrieve winner mechanism.

Required checks: Check row/metric alignment, eligible set, weighted calculations, actual tie handling and winner mechanism.

Evidence family: `manurag_constrained_selection`.

Authoring requirements: Verify thresholds exercise both inclusion and exclusion and author the full gold calculation.

### H03 · target 4/5

Identify the DRAGIN benchmark introduced by the indexed paper with sentence-level supporting facts. Use that paper's splits to create a 10,000-question evaluation pool containing test examples only, allocated proportionally between its two test settings. Apply largest-remainder rounding, break fractional ties alphabetically by split name, respect each split's capacity, and explain how the pool avoids training and development leakage.

Route: Discover benchmark, retrieve split usage and setting identity, filter test rows, allocate under rounding/capacity rules and assess leakage.

Required checks: Check benchmark chain, eligible split identities, integer allocation summing to 10000, capacity and exclusion of train/dev.

Evidence family: `hotpot_constrained_sampling`.

Authoring requirements: Verify the two test sets and exact capacities; record decision points and allocation gold.

### H04 · target 4/5

Compare the base and big Transformer under a one-machine budget of 100 GPU-hours and a minimum WMT 2014 English-to-German test BLEU of 27. Use the paper's reported full training durations. Which configuration is feasible? For each rejected configuration, calculate the GPU-hour shortfall or the BLEU shortfall as applicable; explain why wall-clock hours alone cannot determine feasibility.

Route: Join hardware/durations with test results, convert cost units, apply two feasibility conditions and produce reason-specific shortfalls.

Required checks: Check GPU count, time conversions, feasibility, test scores and conditional rejection arithmetic.

Evidence family: `transformer_constrained_training`.

Authoring requirements: Validate the intended evidence stages and branch count; downgrade if it remains only a single-table arithmetic task.

### H05 · target 4/5

Using the two archived calendar weeks, plan one 90-minute monitoring session per week starting exactly at a listed release time. First maximize the number of Tier 1 releases covered, then all listed releases, then choose the earliest start. A release at the end boundary is excluded. Verify the selected releases' publishing agencies from the official archive and report which simultaneous releases require separate entries.

Route: Retrieve both schedules, construct candidate time intervals, apply lexicographic constraints, resolve simultaneous identities, then retrieve and join agency evidence.

Required checks: Check every candidate score, end-boundary rule, selected dates/times, release completeness and publisher relations.

Evidence family: `calendar_monitoring_optimization`.

Authoring requirements: Freeze publisher archive and gold candidate intervals; verify the boundaries create meaningful alternatives.

### H06 · target 4/5

Audit a claim that every Tier 1 calendar entry in the two archived weeks represents an already-published observation available on 6 April 2026. Identify each relevant release, retrieve its latest official pre-cutoff publication, and build a table distinguishing scheduled release date, publication date and reference period. Count supported versus future-information claims and explain every mismatch.

Route: Discover Tier 1 identities, retrieve their official dated histories, apply publication/reference-period tests and aggregate findings.

Required checks: Check all release identities, three distinct temporal fields, cutoff eligibility and supported/future claim counts.

Evidence family: `calendar_vintage_audit`.

Authoring requirements: Author dated official release archives; no current web data may substitute for pre-cutoff evidence.

### H07 · target 4/5

Assess whether the higher-parameter Transformer delivers the larger relative gain in test BLEU or the larger relative increase in training FLOPs. Join the appropriate tables, calculate all ratios, and recompute BLEU gain using the development scores as a sensitivity analysis. Explain which conclusion changes and why development and test results cannot be mixed in one comparison.

Route: Join model identities across parameter/test/dev tables, calculate normalized changes, branch on evaluation split and compare conclusions.

Required checks: Check source/table provenance, ratio baselines, separate dev/test calculations and the final sensitivity conclusion.

Evidence family: `transformer_metric_sensitivity`.

Authoring requirements: Verify the conclusions and retain this as level 4 only if the validated evidence/decision profile warrants it.

### H08 · target 4/5

For the four ManuRAG variants, build the Pareto frontier using mean factual correctness and minimum per-category context recall across MathQ, MCQ and RQ. From the nondominated variants, choose one whose minimum recall is at least 30%; if several qualify, choose the best mean correctness. Explain each dominance comparison and connect the winner to its visual-indexing strategy.

Route: Retrieve two metric tables, join by variant/category, compute aggregate and worst-case metrics, test pairwise dominance, apply constraint and retrieve methods.

Required checks: Check all metrics, nondominated set, dominance direction, final constraint and mechanism mapping.

Evidence family: `manurag_pareto`.

Authoring requirements: Author exact gold frontier and confirm the source values support nontrivial choices.

### H09 · target 4/5

Identify ACE's AppWorld comparator and both model versions from the paper. Check archived official model cards and leaderboard records available by 6 April 2026. Determine whether the claim that ACE surpassed the comparator holds for the same split, date and evaluation setting; report unknown metadata, reconcile disagreements and calculate a score gap only when the scores are comparable.

Route: Discover comparator/model/split, query exact archived records, branch on date and evaluation comparability, reconcile claims and conditionally calculate gap.

Required checks: Check exact versions, source dates, split identity, comparability decision and justified numeric result or abstention.

Evidence family: `ace_leaderboard_comparability`.

Authoring requirements: Create official dated leaderboard/model snapshots including a plausible but incomparable distractor.

### H10 · target 4/5

Audit a draft benchmark that labels every repeated retrieval as multi-hop and every missing answer as tool recovery. Use the indexed HotpotQA, DRAGIN and ACE papers to define the evidence needed for each label. Apply those definitions to the supplied archived trace bundle, classify each trace, calculate the corrected proportions, and explain any case that cannot be classified from its logged evidence.

Route: Retrieve definitions from multiple papers, inspect trace evidence, branch by dependency/failure/unknown cases and calculate corrected proportions.

Required checks: Require evidence-backed definitions, per-trace labels, justified unknowns and denominators; do not accept label restatement.

Evidence family: `benchmark_trace_audit`.

Authoring requirements: Author a small trace bundle with explicit tool arguments/results, both independent and dependent paths, an actual failure and incomplete evidence; attach it through a shared reproducible tool response.


## Hard: controlled recovery

### R01 · target 5/5

Across the two archived calendar weeks, identify the release that repeats, verify its publisher, and calculate the interval between its two dates. Use the available archived sources to complete both weeks even if a requested calendar window is unavailable. State which evidence supplied each date and publisher.

Route: Request both windows; controlled calendar-window failure occurs; locate the corresponding official archived schedule via web_search; reconcile release identities and time zones; verify publisher and interval.

Required checks: Require observed injected failure, changed route, fallback provenance, both correct dates, publisher and interval; unchanged retry fails recovery.

Evidence family: `calendar_recovery`.

Authoring requirements: Requires a new deterministic per-task/per-run failure scenario in the shared tool registry, reset between runs and identical across frameworks. Archive a valid fallback with the same as-of semantics; a missing fixture or arbitrary search miss is not a controlled failure. Verify the no-failure companion meets level 4; the failure must force a changed plan, not just a repeated call. Score the failure observation, changed tool/query/route, retained constraints and complete final result from the trajectory. Expand the no-failure analysis with comparability and schedule constraints until it validates at level 4.

### R02 · target 5/5

Select the Transformer system meeting the test-BLEU, FLOP and parameter constraints in the attached comparison brief, and calculate savings against the highest-BLEU system. If a relevant indexed table is unusable, find an archived copy of the original paper, preserve the language/test-split constraints and explain how the missing values were recovered.

Route: Perform H01 analysis; controlled table corruption/missing chunk is returned; use web_search archived original table; verify table identity, reconcile units and complete selection.

Required checks: Require changed evidence route after observed unusable result, all H01 constraints and original-table provenance.

Evidence family: `transformer_table_recovery`.

Authoring requirements: Requires a new deterministic per-task/per-run failure scenario in the shared tool registry, reset between runs and identical across frameworks. Archive a valid fallback with the same as-of semantics; a missing fixture or arbitrary search miss is not a controlled failure. Verify the no-failure companion meets level 4; the failure must force a changed plan, not just a repeated call. Score the failure observation, changed tool/query/route, retained constraints and complete final result from the trajectory. Attach the exact H01 constraints in the brief and make the intended table failure reproducible for matching requests.

### R03 · target 5/5

Recommend a ManuRAG variant using the weighted-correctness and minimum-recall constraints in the attached workload brief. If a retrieved result omits a metric needed for eligibility, recover it from the archived original table before choosing. Show eligibility, ranking and the winner's mechanism.

Route: Perform H02 analysis; controlled response omits necessary recall field; change query or source to acquire missing table evidence; recompute eligibility and ranking.

Required checks: Require observed missing-field condition, changed retrieval route, correct recovered metric and complete H02 result.

Evidence family: `manurag_table_recovery`.

Authoring requirements: Requires a new deterministic per-task/per-run failure scenario in the shared tool registry, reset between runs and identical across frameworks. Archive a valid fallback with the same as-of semantics; a missing fixture or arbitrary search miss is not a controlled failure. Verify the no-failure companion meets level 4; the failure must force a changed plan, not just a repeated call. Score the failure observation, changed tool/query/route, retained constraints and complete final result from the trajectory. Attach the exact H02 workload weights and thresholds and ensure the missing field changes a decision.

### R04 · target 5/5

Audit the ACE AppWorld comparison against archived official records using matching date, model version and evaluation split. If the first archived leaderboard endpoint is unavailable, use another eligible archived source and complete the comparability audit. Calculate a gap only when the records are comparable.

Route: Perform H09 discovery/comparability analysis; controlled web endpoint failure; change source/query; verify replacement archive, version, split and dates; conditionally calculate gap.

Required checks: Require observed failure, changed route, fallback as-of validity, comparability reasoning and grounded numeric answer or abstention.

Evidence family: `ace_leaderboard_recovery`.

Authoring requirements: Requires a new deterministic per-task/per-run failure scenario in the shared tool registry, reset between runs and identical across frameworks. Archive a valid fallback with the same as-of semantics; a missing fixture or arbitrary search miss is not a controlled failure. Verify the no-failure companion meets level 4; the failure must force a changed plan, not just a repeated call. Score the failure observation, changed tool/query/route, retained constraints and complete final result from the trajectory.

### R05 · target 5/5

Complete the two-week monitoring plan under the 90-minute, Tier-1-first rules in the attached brief. If an official publisher source is unavailable, use an eligible archived alternative to verify the selected releases. Preserve the session optimization and report any unresolved publisher rather than guessing.

Route: Perform H05 optimization; controlled publisher-lookup failure; change to official alternative archive; reconcile identity/date and retain the chosen-session constraints.

Required checks: Require observed failure and changed route, all H05 interval/selection checks, publisher evidence and justified partial uncertainty if fallback is insufficient.

Evidence family: `calendar_monitoring_recovery`.

Authoring requirements: Requires a new deterministic per-task/per-run failure scenario in the shared tool registry, reset between runs and identical across frameworks. Archive a valid fallback with the same as-of semantics; a missing fixture or arbitrary search miss is not a controlled failure. Verify the no-failure companion meets level 4; the failure must force a changed plan, not just a repeated call. Score the failure observation, changed tool/query/route, retained constraints and complete final result from the trajectory. Attach all H05 rules and separate designed answerable fallback cases from designed partial-answer cases.

