"""Verify the installed DeepEval adapter contract with an entirely local stub."""
import unittest

from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from scripts.judge_runs import make_metric, metrics_for_run


class StubJudge(DeepEvalBaseLLM):
    def __init__(self, raw_score=10):
        self.prompts = []
        self.raw_score = raw_score
        super().__init__(model="offline-stub")

    def load_model(self):
        return None

    def get_model_name(self):
        return "offline-stub"

    def generate(self, prompt, schema=None):
        self.prompts.append(prompt)
        return schema(score=self.raw_score, reason="stub: equivalent content")

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)


class SemanticMetricTests(unittest.TestCase):
    def test_raw_zero_to_ten_score_is_normalized_and_threshold_is_inclusive(self):
        for raw, passed in [(0, False), (7.9, False), (8, True), (10, True)]:
            with self.subTest(raw=raw):
                metric = make_metric("semantic_correctness", StubJudge(raw), {"evaluation": {"semantic_threshold": 0.8}})
                metric.measure(LLMTestCase(input="Question", actual_output="Answer", expected_output="Reference",
                                           context=["Source"]), _show_indicator=False)
                self.assertAlmostEqual(metric.score, raw / 10)
                self.assertEqual(metric.is_successful(), passed)

    def test_old_metric_list_always_connects_semantic_judge_unless_explicitly_opted_out(self):
        task = {"answer_check": {"mode": "judge_only"}}
        run = {"status": "completed", "final_answer": "answer"}
        self.assertEqual(metrics_for_run(["faithfulness", "task_completion"], task, run, []),
                         ["semantic_correctness", "faithfulness", "task_completion"])
        self.assertEqual(metrics_for_run(["faithfulness"], task, run, [], True), ["faithfulness"])
        self.assertEqual(metrics_for_run(["faithfulness"], {"answer_check": {"mode": "key_facts", "params": {"key_facts": ["answer"]}}}, run, []), ["faithfulness"])

    def test_metric_uses_rubric_and_reference_context_with_one_schema_call(self):
        judge = StubJudge()
        metric = make_metric("semantic_correctness", judge, {
            "grading_rubric": ["Explicitly distinguish unsupported combined-system performance."],
            "evaluation": {"semantic_threshold": 0.8}})
        case = LLMTestCase(input="Return JSON explaining the mechanism.", actual_output="Equivalent prose.",
                           expected_output="Reference conclusion.", context=["Frozen source evidence."])
        metric.measure(case, _show_indicator=False)
        self.assertTrue(metric.is_successful())
        self.assertEqual(metric.threshold, 0.8)
        self.assertEqual(len(judge.prompts), 1)
        self.assertIn("unsupported combined-system performance", judge.prompts[0])
        self.assertIn("Frozen source evidence", judge.prompts[0])
        self.assertIn("Ignore presentation and formatting", judge.prompts[0])


if __name__ == "__main__":
    unittest.main()
