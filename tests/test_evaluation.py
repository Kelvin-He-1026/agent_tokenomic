"""Offline integration tests for the 100-task quality reporting contract."""
import copy
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

from agent.evaluation import EVALUATION_VERSION, SEMANTIC_METRIC_VERSION, answer_fingerprint, evaluate
from agent.telemetry import RunRecord
from scripts import export_csv
from test_scoring import TASKS, task


def judgment(record, answer, passed=True, judge_id="judge-a", metric="semantic_correctness"):
    return {"run_id": record["task_id"], "metric_name": metric,
            "score": 1.0 if passed else 0.0, "passed": passed, "judge_id": judge_id,
            "judge_model": judge_id, "evaluation_version": EVALUATION_VERSION,
            "metric_version": SEMANTIC_METRIC_VERSION, "threshold": 0.8,
            "answer_fingerprint": answer_fingerprint(record, answer), "rationale": "fixture judgment"}


class QualityContractTests(unittest.TestCase):
    def test_all_100_have_same_quality_columns_after_judging(self):
        for record in TASKS.values():
            answer = record["reference_answer"]
            result = evaluate(record, answer, "completed", [], [judgment(record, answer)])
            with self.subTest(task=record["task_id"]):
                self.assertTrue(result["success"])
                self.assertTrue(result["semantic_correct"])
                self.assertEqual(result["score_state"], "scored")

    def test_20_semantic_tasks_are_pending_without_a_judge(self):
        results = [evaluate(t, t["reference_answer"], "completed", []) for t in TASKS.values()]
        self.assertEqual(sum(r["score_state"] == "pending_judge" for r in results), 20)
        self.assertEqual(sum(r["success"] is True for r in results), 80)

    def test_diagnostics_cannot_replace_semantic_correctness(self):
        record = task("medium", 11)
        answer = record["reference_answer"]
        for metric in ("faithfulness", "task_completion", "tool_correctness"):
            result = evaluate(record, answer, "completed", [], [judgment(record, answer, metric=metric)])
            self.assertEqual(result["score_state"], "pending_judge")

    def test_stale_answers_rubrics_versions_and_unscored_records_are_not_reused(self):
        record = task("medium", 11)
        answer = record["reference_answer"]
        cases = [judgment(record, answer + " changed"), judgment(record, answer)]
        cases[1]["evaluation_version"] = "old"
        changed = copy.deepcopy(record)
        changed["grading_rubric"].append("new requirement")
        cases.append(judgment(changed, answer))
        cases.append(judgment(record, answer))
        cases[-1]["passed"] = None
        for entry in cases:
            self.assertEqual(evaluate(record, answer, "completed", [], [entry])["score_state"], "pending_judge")

    def test_multiple_judges_require_primary_selection(self):
        record = task("medium", 11)
        answer = record["reference_answer"]
        entries = [judgment(record, answer), judgment(record, answer, False, "judge-b")]
        self.assertEqual(evaluate(record, answer, "completed", [], entries)["score_state"], "ambiguous_judge")
        self.assertIs(evaluate(record, answer, "completed", [], entries, "judge-a")["success"], True)
        self.assertIs(evaluate(record, answer, "completed", [], entries, "judge-b")["success"], False)
        self.assertEqual(export_csv.select_primary_judge(entries[:1]), "judge-a")
        self.assertEqual(export_csv.select_primary_judge(entries, "judge-b"), "judge-b")
        with self.assertRaises(ValueError):
            export_csv.select_primary_judge(entries)

    def test_content_format_and_termination_are_independent(self):
        record = task("easy", 15)
        result = evaluate(record, "The answer is 2250 runs", "completed", [])
        self.assertIs(result["success"], True)
        self.assertIs(result["semantic_correct"], True)
        self.assertIs(result["format_ok"], False)
        self.assertIs(result["strict_success"], False)
        result = evaluate(record, "2251", "completed", [])
        self.assertIs(result["semantic_correct"], False)
        self.assertIs(result["format_ok"], True)
        for status in ("step_limit", "truncated", "timeout", "framework_error"):
            result = evaluate(record, "2250", status, [])
            self.assertIs(result["semantic_correct"], True)
            self.assertIs(result["format_ok"], True)
            self.assertIs(result["success"], False)

    def test_judged_correct_partial_answer_does_not_complete_a_run(self):
        record = task("medium", 11)
        answer = record["reference_answer"]
        result = evaluate(record, answer, "step_limit", [], [judgment(record, answer)])
        self.assertIs(result["semantic_correct"], True)
        self.assertIs(result["success"], False)

    def test_unparseable_content_is_unknown_and_an_explicit_full_denominator_failure(self):
        for record, answer in [(task("easy", 17), '{"weights":'), (task("easy", 15), "2250 or 2200")]:
            result = evaluate(record, answer, "completed", [])
            self.assertEqual(result["semantic_correct"], "")
            self.assertIs(result["format_ok"], False)
            self.assertIs(result["success"], False)
            self.assertEqual(result["score_state"], "unscorable_format")

    def test_wrong_typed_fields_fail_format_and_wrong_values_fail_content(self):
        record = task("easy", 17)
        wrong = json.loads(record["reference_answer"])
        wrong["weights"] = "intermediate_forward_values"
        result = evaluate(record, json.dumps(wrong), "completed", [])
        self.assertIs(result["semantic_correct"], False)
        self.assertIs(result["format_ok"], True)

    def test_empty_semantic_answer_needs_no_paid_judgment(self):
        result = evaluate(task("medium", 11), "", "completed", [])
        self.assertIs(result["success"], False)
        self.assertEqual(result["score_source"], "empty_answer")

    def test_real_export_joins_judgments_for_all_100_tasks(self):
        old_tasks = dict(export_csv.TASKS)
        try:
            export_csv.TASKS.clear()
            export_csv.TASKS.update(TASKS)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                runs, entries = [], []
                for i, record in enumerate(TASKS.values()):
                    run = RunRecord(record["task_id"], "cfg", record["task_id"], 0, i, 0,
                                    final_answer=record["reference_answer"])
                    runs.append(asdict(run))
                    if record["answer_check"]["mode"] == "judge_only":
                        entries.append(judgment(record, run.final_answer))
                for table, rows in (("run", runs), ("eval", entries), ("plan", runs)):
                    (path / f"{table}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
                rows = export_csv.flatten_dir(path)
                self.assertEqual(len(rows), 100)
                self.assertEqual(sum(r["success"] is True for r in rows), 100)
                self.assertFalse(set().union(*(set(r) for r in rows)) - set(export_csv.COLUMNS))
                report = export_csv.quality_report([path], rows)["cells"][0]
                self.assertTrue(report["quality_complete"])
                self.assertEqual(report["confirmed_success_rate"], 1.0)
                missing = export_csv.quality_report([path], rows[1:])["cells"][0]
                self.assertFalse(missing["quality_complete"])
                self.assertEqual(len(missing["missing_runs"]), 1)
                self.assertIsNone(missing["confirmed_success_rate"])
        finally:
            export_csv.TASKS.clear()
            export_csv.TASKS.update(old_tasks)

    def test_changed_task_definition_is_not_silently_rescored(self):
        record = task("easy", 15)
        old = dict(export_csv.TASKS)
        try:
            export_csv.TASKS[record["task_id"]] = record
            result = export_csv._success({"task_id": record["task_id"], "task_fingerprint": "old",
                                          "final_answer": "2250", "status": "completed"}, [])
            self.assertEqual(result["score_state"], "task_mismatch")
            self.assertEqual(result["success"], "")
        finally:
            export_csv.TASKS.clear()
            export_csv.TASKS.update(old)

    def test_same_named_sweep_directories_do_not_mix_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            dirs, rows = [], []
            for model in ("model-a", "model-b"):
                path = Path(directory) / model / "off"
                path.mkdir(parents=True)
                plan = {"config_id": model, "session": "s", "task_id": "t", "repeat_idx": 0, "run_order_idx": 0}
                (path / "plan.jsonl").write_text(json.dumps(plan) + "\n")
                dirs.append(path)
                rows.append({**plan, "run_dir": "off", "run_dir_path": str(path.resolve()),
                             "total_tokens": 10, "e2e_ms": 1})
            report = export_csv.quality_report(dirs, rows)
            self.assertEqual(len(report["cells"]), 2)
            self.assertEqual([c["observed_runs"] for c in report["cells"]], [1, 1])
            self.assertTrue(all(not c["missing_runs"] for c in report["cells"]))

    def test_plan_survives_interruption_before_any_result_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            plan = {"config_id": "cfg", "session": "s", "task_id": "t", "repeat_idx": 0, "run_order_idx": 0}
            (path / "plan.jsonl").write_text(json.dumps(plan) + "\n")
            cell = export_csv.quality_report([path], [])["cells"][0]
            self.assertFalse(cell["quality_complete"])
            self.assertEqual(cell["observed_runs"], 0)
            self.assertEqual(len(cell["missing_runs"]), 1)
            output = path / "results.csv"
            with patch("sys.argv", ["export_csv", "--runs", str(path), "--out", str(output),
                                    "--no-timestamp", "--tasks", str(Path(__file__).resolve().parents[1]
                                                                             / "dataset/tasks/phase1_v2.json")]):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(export_csv.main(), 0)
            self.assertTrue(output.exists())
            exported = json.loads(output.with_suffix(".quality.json").read_text())
            self.assertFalse(exported["cells"][0]["quality_complete"])
            self.assertEqual(len(exported["cells"][0]["missing_runs"]), 1)

    def test_malformed_outer_json_cannot_be_replaced_by_a_correct_nested_fragment(self):
        check = {"mode": "structured", "params": {"separate_format": True, "contract": {
            "type": "object", "fields": {"x": {"type": "number", "value": 1}}}}}
        record = {"answer_check": check}
        result = evaluate(record, '{"x":0,"x":{"x":1}}', "completed", [])
        self.assertEqual(result["semantic_correct"], "")
        self.assertEqual(result["score_state"], "unscorable_format")

    def test_summary_includes_failures_and_pending_cost_even_without_completed_runs(self):
        rows = []
        for state, success in (("pending_judge", ""), ("scored", False)):
            rows.append(dict(is_first_in_dir=False, status="framework_error", model_repo="mock", framework="fw",
                             thinking_setting="off", success=success, score_state=state,
                             total_tokens=100, e2e_ms=1000, reasoning_tokens_missing=False,
                             has_approx_fixture=False, max_llm_calls=10, n_llm_calls=2, task_id="t"))
        output = io.StringIO()
        with redirect_stdout(output):
            export_csv.summarize(rows)
        self.assertIn("confirmed", output.getvalue())
        self.assertIn("pending", output.getvalue())


if __name__ == "__main__":
    unittest.main()
