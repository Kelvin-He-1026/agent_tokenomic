"""Offline regression checks for deterministic grading; no LLM or API calls."""
import copy
import json
import unittest
from collections import Counter
from pathlib import Path

from agent.scoring import score

ROOT = Path(__file__).resolve().parents[1]
TASKS = {t["task_id"]: t for t in json.loads((ROOT / "dataset/tasks/phase1_v2.json").read_text())}


def task(tier, number):
    return TASKS[f"phase1-v2-{tier}-{number:03d}"]


def grade(record, answer):
    text = answer if isinstance(answer, str) else json.dumps(answer)
    return score(text, record["answer_check"]).success


def reference(record):
    return json.loads(record["reference_answer"])


class StructuredScoringTests(unittest.TestCase):
    def test_all_80_deterministic_references_pass_and_only_20_need_judge(self):
        outcomes = Counter()
        for record in TASKS.values():
            with self.subTest(task=record["task_id"]):
                result = score(record["reference_answer"], record["answer_check"])
                outcomes[result.success] += 1
                self.assertIsNot(result.success, False)
                self.assertIsNot(score(record["question"], record["answer_check"]).success, True)
        self.assertEqual(outcomes, {True: 80, None: 20})
        self.assertEqual(Counter(t["difficulty"] for t in TASKS.values()), {"easy": 25, "medium": 50, "hard": 25})

    def test_original_phase1_reference_and_question_results_unchanged(self):
        original = json.loads((ROOT / "dataset/tasks/phase1.json").read_text())
        self.assertEqual(Counter(score(t["reference_answer"], t["answer_check"]).success for t in original), {True: 45, None: 5})
        self.assertEqual(Counter(score(t["question"], t["answer_check"]).success for t in original), {False: 45, None: 5})

    def test_swapped_variant_properties_fail(self):
        record = task("medium", 10)
        answer = reference(record)
        a, b = answer["indexing"]["ManuRAG4"], answer["indexing"]["ManuRAG2"]
        answer["indexing"]["ManuRAG4"], answer["indexing"]["ManuRAG2"] = b, a
        self.assertFalse(grade(record, answer))

    def test_swapped_metric_values_fail(self):
        record = task("medium", 21)
        answer = reference(record)
        answer["test_bleu"]["base"], answer["test_bleu"]["big"] = answer["test_bleu"]["big"], answer["test_bleu"]["base"]
        self.assertFalse(grade(record, answer))

    def test_swapped_role_properties_fail(self):
        record = task("easy", 17)
        answer = reference(record)
        answer["weights"], answer["activations"] = answer["activations"], answer["weights"]
        self.assertFalse(grade(record, answer))

    def test_wrong_release_date_order_fails(self):
        record = task("medium", 38)
        answer = reference(record)
        answer["dates"].reverse()
        self.assertFalse(grade(record, answer))

    def test_missing_subanswer_fails(self):
        record = task("medium", 1)
        answer = reference(record)
        del answer["training_counts"]
        self.assertFalse(grade(record, answer))

    def test_extra_fabricated_release_fails(self):
        record = task("medium", 37)
        answer = reference(record)
        answer["next_week_eligible"].append("Employment Situation")
        self.assertFalse(grade(record, answer))

    def test_wrong_set_member_even_at_right_length_fails(self):
        record = task("hard", 1)
        answer = reference(record)
        answer["eligible"][0] = "ManuRAG2"
        self.assertFalse(grade(record, answer))

    def test_set_order_does_not_matter(self):
        record = task("hard", 1)
        answer = reference(record)
        answer["eligible"].reverse()
        self.assertTrue(grade(record, answer))

    def test_duplicate_set_members_do_not_substitute_for_missing_member(self):
        record = task("hard", 1)
        answer = reference(record)
        answer["eligible"][0] = answer["eligible"][1]
        self.assertFalse(grade(record, answer))

    def test_ordered_query_words_must_keep_original_order(self):
        record = task("hard", 15)
        answer = reference(record)
        answer["qfs_query_words"].reverse()
        self.assertFalse(grade(record, answer))

    def test_fraction_cannot_be_replaced_by_percent(self):
        record = task("medium", 7)
        answer = reference(record)
        answer["hotpot_em_fraction"] *= 100
        self.assertFalse(grade(record, answer))

    def test_numeric_tolerance_accepts_rounding_but_rejects_wrong_result(self):
        record = task("medium", 7)
        answer = reference(record)
        answer["activations_per_expected_exact_answer"] = 11.1624
        self.assertTrue(grade(record, answer))
        answer["activations_per_expected_exact_answer"] = 11.17
        self.assertFalse(grade(record, answer))

    def test_integer_counts_reject_fractional_values(self):
        record = task("medium", 1)
        answer = reference(record)
        answer["valid_examples"] = 112779.5
        self.assertFalse(grade(record, answer))

    def test_boolean_is_not_a_number(self):
        record = task("medium", 1)
        answer = reference(record)
        answer["valid_examples"] = True
        self.assertFalse(grade(record, answer))

    def test_number_is_not_a_boolean(self):
        record = task("easy", 23)
        answer = reference(record)
        answer["premise_correct"] = 0
        self.assertFalse(grade(record, answer))

    def test_unknown_numeric_value_is_not_zero_false_or_string_unknown(self):
        record = task("medium", 40)
        for bad in [0, False, "unknown", "null", "not available", []]:
            with self.subTest(value=bad):
                answer = reference(record)
                answer["april20_24_release_count"] = bad
                self.assertFalse(grade(record, answer))

    def test_known_value_cannot_be_null(self):
        record = task("medium", 40)
        answer = reference(record)
        answer["entries_total"] = None
        self.assertFalse(grade(record, answer))

    def test_unknown_policy_is_not_an_invented_method(self):
        record = task("hard", 10)
        answer = reference(record)
        answer["initial_policy"]["HotpotQA"] = "DRAGIN"
        self.assertFalse(grade(record, answer))

    def test_case_whitespace_and_authored_model_aliases(self):
        record = task("easy", 21)
        answer = reference(record)
        answer.update(agent="  ibm-cuga  ", agent_model="gpt 4.1", ace_model="deepseek v3.1", surpassed_split="challenge")
        self.assertTrue(grade(record, answer))

    def test_substring_or_wrong_model_version_does_not_pass(self):
        record = task("easy", 21)
        for bad in ["IBM CUGA is not the agent", "Other IBM CUGA agent", "CUGA2"]:
            answer = reference(record)
            answer["agent"] = bad
            self.assertFalse(grade(record, answer))
        answer = reference(record)
        answer["agent_model"] = "GPT-4.1-mini"
        self.assertFalse(grade(record, answer))

    def test_numeric_strings_preserve_content_but_fail_format(self):
        record = task("medium", 1)
        answer = reference(record)
        answer["valid_examples"] = "112,779"
        result = score(json.dumps(answer), record["answer_check"])
        self.assertTrue(result.semantic_correct)
        self.assertFalse(result.format_ok)

    def test_duplicate_json_keys_rejected_at_all_depths(self):
        check = {"mode": "structured", "params": {"contract": {"type": "object", "fields": {"row": {"type": "object", "fields": {"x": {"type": "number", "value": 1}}}}}}}
        for text in ['{"row":{"x":1},"row":{"x":1}}', '{"row":{"x":1,"x":1}}']:
            self.assertFalse(score(text, check).success)

    def test_nonfinite_and_huge_numbers_rejected_without_crashing(self):
        check = {"mode": "structured", "params": {"contract": {"type": "object", "fields": {"x": {"type": "number", "value": 1}}}}}
        for value in ["NaN", "Infinity", "-Infinity", "1e9999", "9" * 500]:
            self.assertFalse(score('{"x":' + value + '}', check).success)

    def test_whole_answer_code_fence_is_well_formatted(self):
        record = task("easy", 17)
        gold = record["reference_answer"]
        result = score("```json\n" + gold + "\n```", record["answer_check"])
        self.assertTrue(result.success)
        self.assertTrue(result.format_ok)

    def test_json_in_prose_is_graded_on_content_and_flagged(self):
        record = task("easy", 17)
        gold = record["reference_answer"]
        for text in ["The answer is " + gold, gold + " but those roles are reversed.",
                     "```json\n" + gold + "\n```\nActually, no."]:
            result = score(text, record["answer_check"])
            self.assertTrue(result.success)
            self.assertFalse(result.format_ok)

    def test_last_object_in_prose_wins(self):
        record = task("easy", 17)
        wrong = reference(record)
        wrong[next(iter(wrong))] = "not the answer"
        gold = record["reference_answer"]
        self.assertTrue(grade(record, "Draft: " + json.dumps(wrong) + "\nFinal: " + gold))
        self.assertFalse(grade(record, "Final: " + gold + "\nCorrection: " + json.dumps(wrong)))

    def test_strict_format_restores_format_gating(self):
        record = copy.deepcopy(task("easy", 17))
        record["answer_check"]["params"]["strict_format"] = True
        self.assertFalse(grade(record, "The answer is " + record["reference_answer"]))
        self.assertTrue(grade(record, record["reference_answer"]))

    def test_prose_without_json_still_fails(self):
        record = task("easy", 17)
        result = score("The roles are reversed.", record["answer_check"])
        self.assertFalse(result.success)
        self.assertFalse(result.format_ok)

    def test_extra_fields_fail_format_without_changing_requested_fields(self):
        record = task("easy", 17)
        answer = reference(record)
        answer["explanation"] = "Additional presentation field."
        result = score(json.dumps(answer), record["answer_check"])
        self.assertTrue(result.semantic_correct)
        self.assertFalse(result.format_ok)

    def test_all_required_fields_must_match_even_with_correct_winner(self):
        record = task("hard", 1)
        answer = reference(record)
        answer["weighted_factual_index_percent"]["RAG"] += 1
        self.assertFalse(grade(record, answer))

    def test_matching_handles_overlapping_aliases(self):
        contract = {"type": "array", "unordered": True, "elements": [
            {"type": "string", "value": "a", "aliases": ["b"]}, {"type": "string", "value": "b"}]}
        check = {"mode": "structured", "params": {"contract": contract}}
        self.assertTrue(score('["b","a"]', check).success)
        self.assertFalse(score('["a","a"]', check).success)

    def test_noncompleted_structured_run_fails(self):
        record = task("hard", 1)
        for status in ["timeout", "step_limit", "truncated", "framework_error"]:
            self.assertFalse(score(record["reference_answer"], record["answer_check"], status).success)

    def test_invalid_authored_tolerance_is_a_configuration_error(self):
        bad = {"mode": "structured", "params": {"contract": {"type": "number", "value": 1, "tolerance": -1}}}
        with self.assertRaises(ValueError):
            score("1", bad)


class StrictNumericTests(unittest.TestCase):
    def test_correct_scalar_formats(self):
        record = task("easy", 15)
        for text in ["2250", "2,250", "2.25e3", "2250.0", " 2250 ", "```json\n2250\n```"]:
            self.assertTrue(grade(record, text), text)

    def test_quoting_right_number_in_wrong_answer_fails(self):
        record = task("easy", 15)
        for text in ["2250 is wrong; the answer is 2200", "[2250,2200]", "2250 2200", "22,50"]:
            self.assertFalse(grade(record, text), text)

    def test_unique_number_in_prose_or_quotes_is_content_only_success(self):
        record = task("easy", 15)
        for text in ['"2250"', "The answer is 2250 runs."]:
            result = score(text, record["answer_check"])
            self.assertTrue(result.semantic_correct)
            self.assertFalse(result.format_ok)

    def test_integer_rounding_request_rejects_old_74_point_5_loophole(self):
        record = task("medium", 24)
        self.assertTrue(grade(record, "74"))
        for text in ["74.5", "75", "74 is incorrect; use 75"]:
            self.assertFalse(grade(record, text))

    def test_nonfinite_scalar_rejected(self):
        record = task("easy", 15)
        for text in ["NaN", "Infinity", "1e9999"]:
            self.assertFalse(grade(record, text))

    def test_legacy_numeric_behavior_preserved_unless_opted_in(self):
        legacy = {"mode": "numeric_tolerance", "params": {"value": 2250, "tolerance": 0}}
        self.assertTrue(score("2250 runs", legacy).success)
        strict = copy.deepcopy(legacy)
        strict["params"]["single_number"] = True
        self.assertFalse(score("2250 runs", strict).success)


if __name__ == "__main__":
    unittest.main()
