"""Build dataset/tasks/phase1_v3.json: phase1-v2 with 14 harder hard tasks.

The 14 phase1-v2 hard tasks that Qwen3.8-27B solved in more than two thirds of
its 12 sweep cells (4 thinking modes x 3 frameworks) are replaced; the other
86 tasks are copied unchanged. The new tasks are harder by design rather than
by length:

  dependent retrieval  what to look up next depends on what was found first
  distractors          near-identical figures elsewhere in the same paper
                       (dev vs test BLEU, two type-breakdown samples, ablation
                       tables next to the main results, columns in a
                       different order from the other tables)
  no scaffolding       the field contract names conclusions, not entities or
                       intermediate steps; entity names are free strings
  conditionals/gaps    decisions that depend on a condition, and values the
                       papers do not report (must be null)
  derived values       numbers the papers do not print, from 2-3 reported ones

Every gold value comes from the PDF text (checked against the tables), and each
check passes the usual self-tests. Run:

    python scripts/build_phase1_v3.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.scoring import score  # noqa: E402

V2 = ROOT / "dataset/tasks/phase1_v2.json"
V3 = ROOT / "dataset/tasks/phase1_v3.json"
REPLACED = ["hard-001", "hard-005", "hard-006", "hard-008", "hard-011", "hard-012", "hard-013",
            "hard-017", "hard-018", "hard-019", "hard-020", "hard-022", "hard-023", "hard-024"]

FORMAT = (
    "Report the requested conclusions using only the fields below. Return one compact JSON object "
    "with exactly these keys. Use JSON numbers (not quoted numbers), booleans, and null for values "
    "that the evidence cannot determine. A nullable type is an available answer option, not a hint "
    "that the value is missing. Sets may use any order; other arrays preserve order. Keep rounding "
    "specified above; otherwise give computed non-integers to at least four decimal places. Dates "
    "use YYYY-MM-DD and times use 24-hour HH:MM in ET. Name methods, models, datasets and systems "
    "as the source paper does."
)


# -- contract helpers ---------------------------------------------------------
def N(value, tol=0.0, integer=False):
    return {"type": "number", "value": value, "tolerance": tol, "integer": integer}


def S(value, *aliases, choices=None):
    spec = {"type": "string", "value": value, "aliases": list(aliases)}
    if choices:
        spec["choices"] = list(choices)
    return spec


def B(value):
    return {"type": "boolean", "value": value}


def NULL(public="number"):
    return {"type": "null", "public": public}


def O(**fields):
    return {"type": "object", "fields": fields}


def SET(*elements):
    return {"type": "array", "elements": list(elements), "unordered": True}


def public_shape(spec) -> str:
    """The field/type contract shown to the model: types only, never values."""
    kind = spec["type"]
    if kind == "object":
        return "{" + ", ".join(f"{k}: {public_shape(v)}" for k, v in spec["fields"].items()) + "}"
    if kind == "array":
        inner = public_shape(spec["elements"][0]) if spec["elements"] else "value"
        return f"set of {inner}"
    if kind == "string":
        if spec.get("choices"):
            return "enum(" + ", ".join(f'"{c}"' for c in spec["choices"]) + ") | null"
        return "string | null"
    if kind == "null":
        return f"{spec.get('public', 'number')} | null"
    return f"{kind} | null"


def gold(spec):
    kind = spec["type"]
    if kind == "object":
        return {k: gold(v) for k, v in spec["fields"].items()}
    if kind == "array":
        return [gold(e) for e in spec["elements"]]
    if kind == "null":
        return None
    return spec["value"]


def strip_public(spec):
    """The scorer's contract language has no 'public' key; drop it."""
    spec = copy.deepcopy(spec)
    if spec["type"] == "null":
        spec.pop("public", None)
    for child in spec.get("fields", {}).values():
        child.update(strip_public(child))
    spec["elements"] = [strip_public(e) for e in spec.get("elements", [])] if spec["type"] == "array" else None
    if spec["elements"] is None:
        spec.pop("elements")
    return spec


# -- names as the papers print them, with the variants models commonly use ----
L13B = S("Llama2-13b-chat", "LLaMA2-13B-Chat", "L13B", "Llama-2-13b-chat", "Llama-2-13B-Chat",
         "LLaMA-2-13B-Chat", "Llama2-13B-Chat", "Llama 2 13B Chat", "Llama2 13b chat")
L7B = S("Llama2-7b-chat", "LLaMA2-7B-Chat", "L7B", "Llama-2-7b-chat", "Llama-2-7B-Chat",
        "LLaMA-2-7B-Chat", "Llama2-7B-Chat", "Llama 2 7B Chat", "Llama2 7b chat")
TWOWIKI = S("2WikiMultihopQA", "2WikiMultiHopQA", "2Wiki", "2WikiMQA", "2-WikiMultihopQA", "2WikiMHQA")
IIRC = S("IIRC")
STRAT = S("StrategyQA", "Strategy QA")
FSRAG = S("FS-RAG", "FSRAG", "FS RAG")
SRRAG = S("SR-RAG", "SRRAG", "SR RAG", "single-round RAG")
WORAG = S("wo-RAG", "w/o RAG", "wo RAG", "woRAG", "without RAG", "no RAG", "no retrieval")
RAGHQ = S("RAGhq", "RAG_hq", "RAG-hq", "RAG hq", "RAGHQ", "RAG (hq)")
M4 = S("ManuRAG4", "ManuRAG 4", "ManuRAG-4")
TBASE = S("Transformer (base model)", "Transformer (base)", "Transformer base", "Transformer base model",
          "base Transformer", "Transformer-base", "Transformer (base model) [this work]")
CONVS2S = S("ConvS2S", "ConvS2S [9]", "ConvS2S (single model)", "ConvS2S single")


def r(x, n):
    return round(x, n)


# -- the 14 tasks ---------------------------------------------------------------
def tasks() -> list[dict]:
    out = []

    def add(slot, question, contract, explanation, evidence, steps, caps, family, answerability="answerable"):
        out.append(dict(slot=slot, question=question, contract=contract, explanation=explanation,
                        evidence=evidence, steps=steps, caps=caps, family=family,
                        answerability=answerability))

    # 1. DRAGIN: dependent selection across Tables 2 and 3
    add("hard-001",
        "In the DRAGIN paper's main results, first find the backbone LLM whose DRAGIN exact-match score on "
        "HotpotQA is highest. For that backbone only, find the dataset on which DRAGIN triggers the fewest "
        "retrievals per question. On that dataset, with that backbone, which method other than DRAGIN has the "
        "highest score, and by how many points (score x 100) does DRAGIN exceed it? Give the margin to one "
        "decimal place.",
        O(backbone=L13B, dataset=TWOWIKI, strongest_other_method=FSRAG, dragin_margin_points=N(3.4, 0.05)),
        "HotpotQA EM for DRAGIN: Llama2-13b-chat 0.314 (highest), Vicuna 0.288, Llama2-7b 0.232. Llama2-13b "
        "retrievals per question: 2Wiki 2.631 (fewest), HotpotQA 3.505, StrategyQA 4.786, IIRC 2.829. On 2Wiki "
        "with Llama2-13b the best other method is FS-RAG 0.270; DRAGIN 0.304, margin 3.4 points.",
        [("Dynamic RAG by TsinguaU.pdf", "Table 2"), ("Dynamic RAG by TsinguaU.pdf", "Table 3")],
        ["DRAGIN Table 2 HotpotQA EM per backbone", "DRAGIN Table 3 retrieval counts for that backbone",
         "Table 2 scores on the selected dataset"],
        ["dependent_retrieval", "comparison", "calculation", "filtering"], "D_results")

    # 2. Transformer: two EN-FR values, and dev vs test gains
    add("hard-005",
        "The Transformer paper reports its big model's English-to-French BLEU in more than one place. Give the "
        "value in its main results table and the value stated in the running text of its results section, and "
        "whether they agree. Then, for English-to-German, give the big model's BLEU gain over the base model on "
        "the development set used in its model-variations table, and on the test set used in its main results "
        "table.",
        O(en_fr_bleu_table=N(41.8, 0.005), en_fr_bleu_text=N(41.0, 0.005), values_agree=B(False),
          en_de_dev_gain=N(0.6, 0.05), en_de_test_gain=N(1.1, 0.05)),
        "Table 2 and the abstract give 41.8; Section 6.1 text says 41.0. Table 3 (newstest2013 dev): base 25.8, "
        "big 26.4, gain 0.6. Table 2 (newstest2014 test): 27.3 vs 28.4, gain 1.1.",
        [("Attention Is All You Need.pdf", "Table 2"), ("Attention Is All You Need.pdf", "Section 6.1"),
         ("Attention Is All You Need.pdf", "Table 3")],
        ["Transformer Table 2 big EN-FR BLEU", "Section 6.1 text EN-FR BLEU", "Table 3 dev BLEU base and big"],
        ["comparison", "calculation", "incomplete_evidence"], "T_tables")

    # 3. Transformer: BLEU per training FLOP across Table 2
    ratios = {"GNMT + RL": 24.6 / 23, "ConvS2S": 25.16 / 9.6, "MoE": 26.03 / 20,
              "GNMT + RL Ensemble": 26.30 / 180, "ConvS2S Ensemble": 26.36 / 77,
              "Transformer (base model)": 27.3 / 3.3, "Transformer (big)": 28.4 / 23}
    add("hard-006",
        "In the Transformer paper's main comparison table, consider only systems that have both an "
        "English-to-German BLEU score and an English-to-German training cost. Which system gives the most "
        "English-to-German BLEU per 10^18 training FLOPs, and which gives the second most? Report both "
        "ratios (BLEU per 10^18 FLOPs) to four decimal places.",
        O(best=O(system=TBASE, bleu_per_1e18_flops=N(r(ratios["Transformer (base model)"], 4), 0.001)),
          second=O(system=CONVS2S, bleu_per_1e18_flops=N(r(ratios["ConvS2S"], 4), 0.001))),
        "EN-DE BLEU/FLOPs (x1e18): " + ", ".join(f"{k} {v:.4f}" for k, v in ratios.items())
        + ". ByteNet has no cost and Deep-Att rows have no EN-DE values.",
        [("Attention Is All You Need.pdf", "Table 2")],
        ["Transformer Table 2 EN-DE BLEU and training cost for every system"],
        ["calculation", "comparison", "filtering", "constrained_analysis"], "T_tables")

    # 4. HotpotQA: relative drop distractor -> full wiki
    def drop(a, b):
        return r((a - b) / a * 100, 2)
    add("hard-008",
        "In the HotpotQA paper's baseline results on the test sets, compute the relative drop, as a percentage "
        "of the distractor-setting value, when moving from the distractor setting to the full wiki setting, for "
        "answer EM, supporting-fact EM and joint EM (two decimals each). Which of the three drops most in "
        "relative terms?",
        O(answer_em_drop_pct=N(drop(45.46, 25.23), 0.01), supporting_fact_em_drop_pct=N(drop(22.24, 5.07), 0.01),
          joint_em_drop_pct=N(drop(12.04, 2.63), 0.01),
          largest_relative_drop=S("joint_em", choices=["answer_em", "supporting_fact_em", "joint_em"])),
        "Distractor test answer EM 45.46, sup EM 22.24, joint EM 12.04; full wiki test 25.23, 5.07, 2.63. "
        "Drops 44.50%, 77.20%, 78.16%.",
        [("HOTPOTQA ADataset for Diverse.pdf", "Table 4")],
        ["HotpotQA Table 4 test results, distractor and full wiki"],
        ["calculation", "comparison"], "H_eval")

    # 5. ManuRAG: context table, whose columns are in a different order
    add("hard-011",
        "In the ManuRAG paper's retrieved-context evaluation, for the MathQ category: which method has the "
        "highest context recall, by how many percentage points does it exceed ManuRAG4's context recall, and "
        "what is ManuRAG4's context precision? Then name the method with the highest context precision on RQ.",
        O(mathq_recall_leader=S("RAG", "standard RAG", "naive RAG", "vanilla RAG", "baseline RAG", "plain RAG"),
          mathq_recall_margin_over_manurag4=N(r(67.83 - 62.26, 2), 0.005),
          manurag4_mathq_precision=N(93.53, 0.005), rq_precision_leader=RAGHQ),
        "Table 3 lists MCQ, MathQ, RQ (not the MathQ-first order of Tables 1-2). MathQ recall: RAG 67.83, "
        "ManuRAG4 62.26, RAGhq 57.00. ManuRAG4 MathQ precision 93.53. RQ precision: RAGhq 93.74 > RAG 92.24.",
        [("ManuRAG.pdf", "Table 3")],
        ["ManuRAG Table 3 context recall and precision by category"],
        ["filtering", "comparison", "calculation"], "M_context")

    # 6. ManuRAG: answer quality vs retrieval leaders, per category
    def cat(fc, cp, same):
        return O(factual_correctness_leader=fc, context_precision_leader=cp, same_method=B(same))
    add("hard-012",
        "In the ManuRAG paper, for each question category (MathQ, MCQ and RQ), name the method with the highest "
        "factual correctness in the answer-quality results and the method with the highest context precision "
        "in the retrieved-context results, and say whether they are the same method.",
        O(MathQ=cat(M4, M4, True), MCQ=cat(M4, M4, True), RQ=cat(M4, RAGHQ, False)),
        "Factual correctness leaders: MathQ ManuRAG4 62.03, MCQ ManuRAG4 87.61, RQ ManuRAG4 45.75 (RAGhq "
        "45.64). Context precision (Table 3, columns MCQ, MathQ, RQ): MathQ ManuRAG4 93.53, MCQ ManuRAG4 86.84, "
        "RQ RAGhq 93.74.",
        [("ManuRAG.pdf", "Tables 1 and 2"), ("ManuRAG.pdf", "Table 3")],
        ["ManuRAG factual correctness per category", "ManuRAG Table 3 context precision per category"],
        ["comparison", "cross_table", "filtering", "conditional_decision"], "M_results+M_context")

    # 7. ACE: derived per-rollout latency, headline average, missing dollar figure
    gepa, ace = 53898 / 1434, 9517 / 357
    app, fin = (53898 - 9517) / 53898 * 100, (65104 - 5503) / 65104 * 100
    add("hard-013",
        "ACE reports adaptation cost for an offline AppWorld comparison and an online FiNER comparison. "
        "(1) For the offline AppWorld comparison, compute each method's average adaptation latency per rollout "
        "(seconds per rollout) and the percentage reduction in per-rollout latency (two decimals). (2) The "
        "paper states an average adaptation-latency reduction; give the two latency reductions it averages and "
        "their mean (two decimals). (3) Give the online FiNER dollar-cost reduction percentage (two decimals), "
        "and the offline AppWorld dollar-cost reduction if the paper reports one.",
        O(appworld=O(baseline_method=S("GEPA", "ReAct + GEPA", "ReAct+GEPA"),
                     baseline_seconds_per_rollout=N(r(gepa, 4), 0.01), ace_seconds_per_rollout=N(r(ace, 4), 0.01),
                     per_rollout_latency_reduction_pct=N(r((gepa - ace) / gepa * 100, 2), 0.05)),
          headline_average=O(appworld_latency_reduction_pct=N(r(app, 2), 0.06),
                             finer_latency_reduction_pct=N(r(fin, 2), 0.06), mean_pct=N(r((app + fin) / 2, 2), 0.06)),
          finer_dollar_reduction_pct=N(r((17.7 - 2.9) / 17.7 * 100, 2), 0.05),
          appworld_dollar_reduction_pct=NULL()),
        f"GEPA 53898 s / 1434 rollouts = {gepa:.4f}; ACE 9517 / 357 = {ace:.4f}; per-rollout reduction "
        f"{(gepa - ace) / gepa * 100:.2f}%. Headline 86.9% = mean of {app:.2f}% (AppWorld) and {fin:.2f}% "
        "(FiNER). FiNER dollars 17.7 -> 2.9 = 83.62%. No AppWorld dollar cost is reported.",
        [("Agentic Context Engineering.pdf", "Table 4"), ("Agentic Context Engineering.pdf", "Section 4.6")],
        ["ACE Table 4 offline AppWorld latency and rollouts", "ACE online FiNER latency and dollar cost",
         "ACE headline adaptation-latency claim"],
        ["calculation", "incomplete_evidence", "dependent_retrieval"], "A_cost", "partial")

    # 8. ACE: ablation attribution, including a component that hurts one metric
    add("hard-017",
        "Using ACE's AppWorld ablation results, attribute the average-score gain of full offline ACE over the "
        "ReAct baseline to three parts: multi-epoch adaptation (full ACE vs ACE without multi-epoch), the "
        "Reflector (ACE without multi-epoch vs ACE without Reflector or multi-epoch), and everything else (ACE "
        "without Reflector or multi-epoch vs ReAct). In the online setting, how much does offline warmup add to "
        "the average? Finally, with multi-epoch off, does adding the Reflector raise the test-challenge TGC, and "
        "by how much does that TGC change?",
        O(multi_epoch_points=N(2.6, 0.05), reflector_points=N(1.7, 0.05), everything_else_points=N(12.7, 0.05),
          online_warmup_points=N(3.4, 0.05), reflector_raises_challenge_tgc=B(False),
          reflector_challenge_tgc_change=N(-1.0, 0.05)),
        "Averages: ReAct 42.4; w/o Reflector or multi-epoch 55.1; w/o multi-epoch 56.8; full 59.4 (2.6 + 1.7 + "
        "12.7 = 17.0). Online 56.1 -> 59.5 with warmup (+3.4). Test-challenge TGC: 55.9 without Reflector, 54.9 "
        "with it: -1.0.",
        [("Agentic Context Engineering.pdf", "Table 3")],
        ["ACE Table 3 ablation rows, all columns"],
        ["calculation", "comparison", "conditional_decision"], "A_cost")

    # 9. HotpotQA: two type breakdowns and the collection ratios
    add("hard-018",
        "The HotpotQA paper analyses reasoning types in two samples: one from the dev and test sets and one from "
        "the training data. Which reasoning category changes most, in absolute percentage points, between the two "
        "samples, and what is that change (dev/test share minus training share)? Give the comparison-question "
        "share in each sample, and the comparison and yes/no shares (percent of all questions) implied by the "
        "ratios in the paper's data-collection algorithm.",
        O(largest_change_category=S("Type II", "type 2", "TypeII", "Type-II", "Type 2"),
          largest_change_points=N(-14.0, 0.05), comparison_share_dev_test=N(27.0, 0.05),
          comparison_share_training=N(20.0, 0.05), algorithm_comparison_share=N(25.0, 0.05),
          algorithm_yes_no_share=N(12.5, 0.05)),
        "Dev/test: Type I 42, Comparison 27, Type II 15, Type III 6, Other 2, single-hop 6, unanswerable 2. "
        "Training (train-medium + train-hard): Type I 38, Type II 29, Comparison 20, Other 7, Type III 2, "
        "single-hop 2, unanswerable 2. Largest change Type II 15 - 29 = -14. Algorithm 1: r1 = 0.75 bridge, so "
        "comparison 25%; yes/no ratio 0.5 of comparisons = 12.5%.",
        [("HOTPOTQA ADataset for Diverse.pdf", "Table 3 and Section 4"),
         ("HOTPOTQA ADataset for Diverse.pdf", "Algorithm 1")],
        ["HotpotQA dev/test reasoning-type shares", "HotpotQA training-sample reasoning-type shares",
         "HotpotQA Algorithm 1 ratios"],
        ["comparison", "calculation", "cross_table"], "H_design")

    # 10. Transformer: reproduce the training-cost method
    big, base = 8 * 3.5 * 86400 * 9.5e12, 8 * 12 * 3600 * 9.5e12
    add("hard-019",
        "The Transformer paper explains how it estimates training cost and gives a sustained-throughput figure "
        "for the GPU it trained on. Using that method and figure with the reported training times and number of "
        "GPUs, estimate the training FLOPs of the big model and of the base model, and give the ratio of the "
        "cost in the main results table to your estimate for each (four decimals). Does the method reproduce "
        "both table values to within 1%?",
        O(gpu_tflops_used=N(9.5, 0.005),
          big=O(estimated_flops=N(big, 1e16), table_to_estimate_ratio=N(r(2.3e19 / big, 4), 0.0005)),
          base=O(estimated_flops=N(base, 1e15), table_to_estimate_ratio=N(r(3.3e18 / base, 4), 0.0005)),
          reproduces_within_1pct=B(True)),
        f"FLOPs = time x GPUs x sustained TFLOPS; P100 = 9.5 TFLOPS. Big: 3.5 days x 8 GPUs -> {big:.6g}; table "
        f"2.3e19, ratio {2.3e19 / big:.4f}. Base: 12 hours x 8 -> {base:.6g}; table 3.3e18, ratio "
        f"{3.3e18 / base:.4f}. Using steps x step time instead would be the wrong method.",
        [("Attention Is All You Need.pdf", "Section 6.1 and footnote 5"),
         ("Attention Is All You Need.pdf", "Section 5.2"), ("Attention Is All You Need.pdf", "Table 2")],
        ["Transformer cost-estimation method and P100 TFLOPS", "Transformer training times and GPU count",
         "Table 2 training costs"],
        ["dependent_retrieval", "calculation", "conditional_decision"], "T_train+T_tables")

    # 11. DRAGIN: scan all 12 cells of two tables
    def cell(bb, ds, method=None):
        f = dict(backbone=bb, dataset=ds)
        if method:
            f["best_method"] = method
        return O(**f)
    add("hard-020",
        "Across all three backbone LLMs and four datasets in DRAGIN's main results (exact match; accuracy for "
        "StrategyQA), in how many of the 12 backbone-dataset cells does DRAGIN have the highest score among all "
        "methods? List the cells where it does not, with the method that is highest there. Separately, using "
        "the paper's retrieval-count comparison, does DRAGIN use fewer retrievals per question than FS-RAG in "
        "every backbone-dataset cell? List any exception cells.",
        O(cells_where_dragin_is_best=N(10, 0, True),
          cells_where_another_method_is_best=SET(cell(L13B, IIRC, SRRAG), cell(L7B, STRAT, WORAG)),
          fewer_retrievals_than_fs_rag_everywhere=B(False),
          fs_rag_exception_cells=SET(cell(L7B, STRAT))),
        "Table 2: DRAGIN is highest except Llama2-13b IIRC (SR-RAG 0.196 > 0.185) and Llama2-7b StrategyQA "
        "(wo-RAG 0.659 > 0.641). Table 3: DRAGIN fewer than FS-RAG except Llama2-7b StrategyQA (4.629 > 4.546). "
        "The ablation tables later in the paper report different numbers for some cells and are not the main "
        "results.",
        [("Dynamic RAG by TsinguaU.pdf", "Table 2"), ("Dynamic RAG by TsinguaU.pdf", "Table 3")],
        ["DRAGIN Table 2 all backbones and datasets", "DRAGIN Table 3 retrieval counts all backbones"],
        ["comparison", "filtering", "cross_table", "constrained_analysis"], "D_results")

    # 12. HotpotQA: joint F1 vs product of averages
    pd_, pf = 58.99 * 66.62 / 100, 34.40 * 40.69 / 100
    add("hard-022",
        "HotpotQA defines joint precision and recall as products of the answer and supporting-fact values. For "
        "the baseline's distractor and full wiki test results, compute the product of answer F1 and "
        "supporting-fact F1 (as a percentage, two decimals) and compare it with the reported joint F1. Is the "
        "reported joint F1 higher than that product in each setting? Which part of the paper's metric "
        "definition makes this possible?",
        O(distractor=O(product_of_f1=N(r(pd_, 2), 0.01), reported_joint_f1=N(41.37, 0.005), joint_higher=B(True)),
          full_wiki=O(product_of_f1=N(r(pf, 2), 0.01), reported_joint_f1=N(17.85, 0.005), joint_higher=B(True)),
          explanation=S("metrics_averaged_per_example", choices=[
              "metrics_averaged_per_example", "reporting_error", "different_test_sets", "joint_f1_uses_em"])),
        f"Distractor: 58.99 x 66.62 = {pd_:.2f} < joint F1 41.37. Full wiki: 34.40 x 40.69 = {pf:.2f} < 17.85. "
        "The paper computes joint P/R per example and averages over examples; the product of averages is not "
        "the average of per-example products.",
        [("HOTPOTQA ADataset for Diverse.pdf", "Section 5.2"), ("HOTPOTQA ADataset for Diverse.pdf", "Table 4")],
        ["HotpotQA joint metric definition", "HotpotQA Table 4 answer, supporting-fact and joint F1"],
        ["calculation", "conditional_decision", "comparison"], "H_eval")

    # 13. Calendar: two windows, conditional counting
    add("hard-023",
        "Using the economic calendar for this week and next week, find every date with a Tier 1 release. On those "
        "dates, how many releases are not Tier 1? List the dates on which two or more releases share the same "
        "time slot, and give the earliest time (ET) of any release that is not Tier 1 on a date that also has a "
        "Tier 1 release.",
        O(tier1_dates=SET(S("2026-04-09"), S("2026-04-10"), S("2026-04-14")),
          non_tier1_releases_on_tier1_dates=N(1, 0, True),
          shared_time_slot_dates=SET(S("2026-04-09")),
          earliest_non_tier1_time_on_tier1_date=S("10:00", "10:00 ET", "10:00 AM")),
        "This week: Tier 1 on 04-09 (GDP and Personal Income, both 8:30) and 04-10 (CPI); next week: 04-14 "
        "(PPI). Only non-Tier-1 release on those dates: Manufacturer's Shipments 04-09 at 10:00. Shared slot: "
        "04-09 8:30.",
        [("dataset/fixtures/fred_releases", "this_week and next_week")],
        ["fred_releases this_week", "fred_releases next_week"],
        ["filtering", "conditional_decision", "calculation"], "F_this+F_next")

    # 14. ACE: finance results, matching the regime
    add("hard-024",
        "For ACE's financial benchmarks, in the online setting without ground-truth labels: does ACE beat the "
        "base model on FiNER and on Formula, and by how many points does it beat or trail the DC baseline in the "
        "same setting on each? Which ACE configuration (offline or online, with or without labels) gives the "
        "largest Formula gain over the base model, and how large is that gain?",
        O(online_unlabeled=O(beats_base_finer=B(False), beats_base_formula=B(True),
                             ace_minus_dc_finer=N(-1.0, 0.05), ace_minus_dc_formula=N(16.0, 0.05)),
          best_formula=O(setting=S("offline", choices=["offline", "online"]), uses_labels=B(True),
                         gain_points=N(18.0, 0.05))),
        "Base 70.7 / 67.5. Online, no labels: ACE 67.3 / 78.5 (FiNER -3.4, Formula +11.0); DC 68.3 / 62.5, so "
        "ACE - DC = -1.0 / +16.0. Formula: offline labeled 85.5 (+18.0), offline unlabeled 83.0, online labeled "
        "76.5, online unlabeled 78.5.",
        [("Agentic Context Engineering.pdf", "Table 2")],
        ["ACE Table 2 finance results, all configurations"],
        ["comparison", "filtering", "calculation", "conditional_decision"], "A_results")
    return out


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    v2 = json.loads(V2.read_text())
    by_slot = {t["task_id"].replace("phase1-v2-", ""): t for t in v2}
    template = by_slot["hard-012"]
    docs = {p.name: sha(p) for p in (ROOT / "documents").glob("*.pdf")}
    new = {}
    for spec in tasks():
        contract = spec["contract"]
        check = {"mode": "structured", "params": {"contract_version": 1, "contract": strip_public(contract)}}
        question = (f"{spec['question']}\n\n{FORMAT}\n\nField/type contract (types and allowed codes, "
                    f"not answers):\n{public_shape(contract)}")
        reference = json.dumps(gold(contract), separators=(",", ":"))
        old = by_slot[spec["slot"]]
        task = {
            **{k: copy.deepcopy(template[k]) for k in ("acceptable_tool_sets", "expected_args", "required_no_tool",
                                                         "replay_fixture_id", "grading_rubric", "answer_format",
                                                         "grading_version", "evaluation", "as_of_date")},
            "task_id": f"phase1-v3-{spec['slot']}",
            "question": question,
            "task_type": "analysis",
            "difficulty": "hard",
            "reference_answer": reference,
            "reference_context": f"[{spec['family']}] {spec['explanation']}",
            "reference_explanation": spec["explanation"],
            "answer_check": check,
            "ideal_llm_calls": min(10, len(spec["steps"]) + 1),
            "source": "authored_phase1_v3",
            "notes": "Harder replacement for " + old["task_id"] + " (solved in >2/3 of Qwen sweep cells). "
                     "Gold values checked against the PDF text.",
            "key_facts_proposed": False,
            "needs_review": False,
            "review": {"by": "Claude", "date": "2026-10-20", "human_verified": False,
                       "grounding": "Gold values read from the PDF text tables/sections and computed in "
                                    "scripts/build_phase1_v3.py; not yet human-verified or piloted."},
            "dataset_version": "phase1_v3",
            "difficulty_score": 5,
            "difficulty_profile": {"dependencies": max(0, len(spec["steps"]) - 1), "tools": 1,
                                   "branches": 2, "distractors": 1, "calculation": 1, "recovery": 0},
            "capabilities": spec["caps"],
            "evidence": [{"source": f"documents/{f}" if f.endswith(".pdf") else f, "section": s,
                          **({"sha256": docs[f]} if f in docs else {})} for f, s in spec["evidence"]],
            "evidence_family": spec["family"],
            "solution_steps": [{"tool": "fred_releases" if "fred" in st else "retrieve",
                                "query_or_window": st, "depends_on": list(range(i))} for i, st in enumerate(spec["steps"])],
            "derived_from": [old["task_id"]],
            "answerability": spec["answerability"],
            "replaces": old["task_id"],
        }
        if "fred" in spec["family"].lower() or spec["family"].startswith("F_"):
            task["acceptable_tool_sets"] = [["fred_releases"], ["fred_releases", "web_search"]]
        new[spec["slot"]] = task
    assert sorted(new) == sorted(REPLACED), sorted(set(REPLACED) ^ set(new))

    out = []
    for t in v2:
        slot = t["task_id"].replace("phase1-v2-", "")
        out.append(new[slot] if slot in new else {**t, "dataset_version": "phase1_v3"})

    # Self-tests: every reference passes its own check; no question passes alone.
    bad = []
    for t in out:
        if t["answer_check"]["mode"] == "judge_only":
            continue
        if not score(t["reference_answer"], t["answer_check"]).success:
            bad.append(f"{t['task_id']}: reference fails")
        if score(t["question"], t["answer_check"]).success:
            bad.append(f"{t['task_id']}: question passes alone")
    if bad:
        raise SystemExit("self-test failures:\n  " + "\n  ".join(bad))
    V3.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    levels = {}
    for t in out:
        levels[t["difficulty"]] = levels.get(t["difficulty"], 0) + 1
    print(f"wrote {V3.relative_to(ROOT)}: {len(out)} tasks {levels}; {len(new)} new hard tasks; self-tests ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
