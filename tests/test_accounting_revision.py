"""Regression cases from the Nemotron 90-run sweep and cap edge cases."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.accounting import cap_hits, completion_type
from agent.evaluation import evaluate, semantic_threshold, valid_semantic_judgment
from agent.harness.runner import Runner
from agent.telemetry import RunRecord
from test_evaluation import judgment
from test_scoring import task


class AccountingRevisionTests(unittest.TestCase):
    def test_live_runner_stores_the_same_per_call_flags_as_offline_export(self):
        call = {"run_id": "r", "call_idx": 1, "finish_reason": "length",
                "completion_tokens": 8192, "effective_output_cap": 8192}
        runner = object.__new__(Runner)
        runner.writer = SimpleNamespace(read=lambda _: [call])
        runner._proxy_snapshot = {"blocked_requests": 0, "termination_reason": ""}
        record = RunRecord("r", "c", "t", 0, 0, 0, max_llm_calls=10,
                           max_tokens_per_call=8192, max_total_output_tokens=81920)
        runner._fold_in_llm_calls(record)
        self.assertTrue(record.output_cap_reached)
        self.assertTrue(record.per_call_output_cap_reached)
        self.assertFalse(record.total_output_cap_reached)
        self.assertEqual(record.termination_reason, "per_call_output_cap")
        self.assertEqual(record.completion_type, "truncated")

    def test_per_call_length_is_a_cap_hit_below_the_run_total(self):
        run = {"status": "truncated", "output_tokens": 9000, "max_tokens_per_call": 8192,
               "max_total_output_tokens": 81920}
        hits = cap_hits(run, [{"completion_tokens": 8192, "effective_output_cap": 8192,
                               "finish_reason": "length"}])
        self.assertTrue(hits["output_cap_reached"])
        self.assertTrue(hits["per_call_output_cap_reached"])
        self.assertFalse(hits["total_output_cap_reached"])
        self.assertEqual(hits["output_cap_type"], "per_call")

    def test_smaller_remaining_budget_is_a_total_cap_hit(self):
        hits = cap_hits({"status": "truncated", "max_tokens_per_call": 10, "max_total_output_tokens": 13},
                        [{"call_idx": 1, "completion_tokens": 5, "effective_output_cap": 10, "finish_reason": "tool_calls"},
                         {"call_idx": 2, "completion_tokens": 8, "effective_output_cap": 8, "finish_reason": "length"}])
        self.assertTrue(hits["output_cap_reached"])
        self.assertFalse(hits["per_call_output_cap_reached"])
        self.assertTrue(hits["total_output_cap_reached"])
        self.assertEqual(hits["output_cap_type"], "total")

    def test_both_caps_can_be_reached_at_once(self):
        hits = cap_hits({"status": "truncated", "max_tokens_per_call": 10, "max_total_output_tokens": 20},
                        [{"call_idx": 1, "completion_tokens": 10, "finish_reason": "tool_calls"},
                         {"call_idx": 2, "completion_tokens": 10, "finish_reason": "length"}])
        self.assertEqual(hits["output_cap_type"], "both")

    def test_cap_hit_is_preserved_even_if_a_later_call_recovers(self):
        hits = cap_hits({"status": "completed", "termination_reason": "natural_stop", "max_tokens_per_call": 10,
                         "max_llm_calls": 3, "max_total_output_tokens": 30},
                        [{"call_idx": 1, "completion_tokens": 10, "finish_reason": "length"},
                         {"call_idx": 2, "completion_tokens": 3, "finish_reason": "stop"}])
        self.assertTrue(hits["per_call_output_cap_reached"])
        self.assertEqual(hits["completion_type"], "natural_completion")

    def test_natural_completion_at_the_call_cap_remains_natural(self):
        hits = cap_hits({"status": "completed", "termination_reason": "natural_stop", "max_llm_calls": 1},
                        [{"finish_reason": "stop", "completion_tokens": 10}])
        self.assertTrue(hits["call_cap_reached"])
        self.assertEqual(hits["completion_type"], "natural_completion")

    def test_forced_final_and_step_limit_are_distinct(self):
        self.assertEqual(completion_type("completed", "forced_final_answer", True, "stop"), "forced_final")
        self.assertEqual(completion_type("step_limit", "call_cap", False, "tool_calls"), "step_limit")
        self.assertEqual(completion_type("truncated", "per_call_output_cap", False, "length"), "truncated")

    def test_strict_success_requires_natural_completion_for_every_framework(self):
        record = task("easy", 15)
        for framework in ("langgraph", "crewai", "openai_agents"):
            for kind in ("natural_completion", "forced_final", "unknown_completion"):
                with self.subTest(framework=framework, completion=kind):
                    result = evaluate(record, "2250", "completed", [], completion_type=kind)
                    self.assertTrue(result["success"])
                    self.assertTrue(result["format_success"])
                    self.assertEqual(result["strict_success"], kind == "natural_completion")

    def test_invalid_or_stale_thresholds_and_inconsistent_verdicts_need_rejudging(self):
        record = task("medium", 11)
        answer = record["reference_answer"]
        for mutation in ({"threshold": 0.5}, {"metric_version": "old"}, {"score": 9},
                         {"score": float("nan")}, {"score": 0.7, "passed": True}):
            entry = {**judgment(record, answer), **mutation}
            self.assertFalse(valid_semantic_judgment(entry, record, answer))
            self.assertEqual(evaluate(record, answer, "completed", [], [entry])["score_state"], "pending_judge")

    def test_threshold_validation(self):
        for value in (-0.1, 1.1, True, float("nan"), "0.8"):
            with self.assertRaises(ValueError):
                semantic_threshold({"evaluation": {"semantic_threshold": value}})

    def test_nemotron_recorded_lengths_are_all_per_call_caps(self):
        root = Path(__file__).resolve().parents[1] / "output/runs/thinking_sweep_NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16_20261016_120206"
        if not root.exists():
            self.skipTest("local sweep artifacts unavailable")
        count, forced = 0, 0
        for mode in ("off", "on"):
            d = root / mode
            calls = [json.loads(line) for line in (d / "llm_call.jsonl").read_text().splitlines()]
            for line in (d / "run.jsonl").read_text().splitlines():
                run = json.loads(line)
                result = cap_hits(run, [c for c in calls if c["run_id"] == run["run_id"]])
                if run["status"] == "truncated":
                    count += 1
                    self.assertTrue(result["output_cap_reached"])
                    self.assertTrue(result["per_call_output_cap_reached"])
                    self.assertFalse(result["total_output_cap_reached"])
                forced += result["completion_type"] == "forced_final"
        self.assertEqual(count, 6)
        self.assertEqual(forced, 7)
