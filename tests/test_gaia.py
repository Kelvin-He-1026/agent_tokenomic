"""Offline checks for the GAIA task set: scorer, attachment reader, URL guard, tool lists."""
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from agent.config import RunConfig
from agent.prompts import task_prompt
from agent.scoring import gaia_match, score
from agent.tools import impl, schemas
from agent.tools.impl import ToolFailure

ROOT = Path(__file__).resolve().parents[1]
TEXT = ROOT / "GAIA_stack/tasks/phase1-GAIA-text.json"
GAIA_CFG = RunConfig.from_yaml(ROOT / "GAIA_stack/configs/gaia_qwen_tp2.yaml")
# The GAIA task files are not committed (GAIA_stack/scripts/fetch_gaia.py).
needs_data = unittest.skipUnless(TEXT.exists(), "GAIA task files not fetched")


class GaiaMatchTests(unittest.TestCase):
    def test_numbers_ignore_units_and_separators(self):
        self.assertTrue(gaia_match("$1,234", "1234"))
        self.assertTrue(gaia_match("17%", "17"))
        self.assertFalse(gaia_match("1235", "1234"))

    def test_strings_ignore_case_spaces_punctuation(self):
        self.assertTrue(gaia_match("Right ", "right"))
        self.assertTrue(gaia_match("St. Petersburg", "st petersburg"))
        self.assertFalse(gaia_match("Saint Petersburg", "St. Petersburg"))

    def test_lists_compare_element_wise(self):
        self.assertTrue(gaia_match("a, b, 3", "a,b,3"))
        self.assertFalse(gaia_match("a, b", "a,b,3"))
        self.assertFalse(gaia_match("b, a, 3", "a,b,3"))


class GaiaScoreTests(unittest.TestCase):
    check = {"mode": "gaia_exact", "params": {"answer": "42"}}

    def test_final_answer_line(self):
        s = score("Working...\nFINAL ANSWER: 42", self.check)
        self.assertTrue(s.success)
        self.assertTrue(s.format_ok)

    def test_missing_marker_falls_back_to_last_line(self):
        s = score("I think the answer is\n42", self.check)
        self.assertTrue(s.semantic_correct)
        self.assertFalse(s.format_ok)

    def test_wrong_answer(self):
        self.assertFalse(score("FINAL ANSWER: 41", self.check).success)

    @needs_data
    def test_every_reference_answer_passes(self):
        for t in json.loads(TEXT.read_text()):
            with self.subTest(task=t["task_id"]):
                ans = f"FINAL ANSWER: {t['reference_answer']}"
                self.assertTrue(score(ans, t["answer_check"]).success)


class GaiaToolTests(unittest.TestCase):
    @needs_data
    def test_every_readable_attachment_opens(self):
        for t in json.loads(TEXT.read_text()):
            if t.get("file_name"):
                with self.subTest(file=t["file_name"]):
                    self.assertTrue(impl.live_read_file(t["file_name"], GAIA_CFG).strip())

    def test_read_file_stays_in_attachments_dir(self):
        with self.assertRaises(ToolFailure):
            impl.live_read_file("../../.env", GAIA_CFG)

    def test_fetch_url_blocks_local_addresses(self):
        for url in ("http://localhost:8100/control/release", "http://127.0.0.1/", "http://169.254.169.254/",
                    "file:///etc/passwd"):
            with self.subTest(url=url), self.assertRaises(ToolFailure):
                impl.live_fetch_url(url, GAIA_CFG)


class SearchFallbackTests(unittest.TestCase):
    """tavily_then_ddg, with both engines mocked: no network, no credits."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        marker = Path(self.tmp.name) / "tavily_exhausted"
        self.patches = [
            mock.patch.object(impl, "_tavily_marker", lambda cfg: marker),
            mock.patch.object(impl, "live_ddg_search", lambda q, cfg: f"ddg:{q}"),
            mock.patch.dict(os.environ, {"TAVILY_API_KEY": "test"}),
        ]
        for p in self.patches:
            p.start()
        self.cfg = replace(GAIA_CFG, web_search_backend="tavily_then_ddg")

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_uses_tavily_while_it_has_credit(self):
        with mock.patch.object(impl, "_tavily_search", lambda q: f"tavily:{q}"):
            self.assertEqual(impl.live_web_search("x", self.cfg), "tavily:x")
        self.assertEqual(impl.LAST_SEARCH_ENGINE, "tavily")

    def test_out_of_credit_switches_to_ddg_for_good(self):
        from tavily.errors import ForbiddenError

        calls = []

        def broke(q):
            calls.append(q)
            raise ForbiddenError("plan limit")

        with mock.patch.object(impl, "_tavily_search", broke):
            self.assertEqual(impl.live_web_search("a", self.cfg), "ddg:a")
            self.assertEqual(impl.live_web_search("b", self.cfg), "ddg:b")
        self.assertEqual(calls, ["a"])  # Tavily is not asked again once exhausted
        self.assertEqual(impl.LAST_SEARCH_ENGINE, "ddg")

    def test_rate_limit_falls_back_for_that_query_only(self):
        from tavily.errors import UsageLimitExceededError

        def busy(q):
            raise UsageLimitExceededError("429")

        with mock.patch.object(impl, "_tavily_search", busy), mock.patch.object(impl.time, "sleep"):
            self.assertEqual(impl.live_web_search("a", self.cfg), "ddg:a")
        self.assertFalse(impl._tavily_exhausted(self.cfg))


class ToolListTests(unittest.TestCase):
    def test_phase1_tools_unchanged(self):
        self.assertEqual(schemas.tool_schema_hash(), "d875ccc078766ae3")
        self.assertEqual(schemas.tool_specs(()), schemas.TOOL_SCHEMAS)

    def test_gaia_config_offers_only_gaia_tools(self):
        self.assertEqual([t["name"] for t in schemas.tool_specs(GAIA_CFG.tools)],
                         ["web_search", "fetch_url", "read_file"])
        self.assertNotEqual(schemas.tool_schema_hash(GAIA_CFG.tools), schemas.tool_schema_hash())

    def test_search_backend(self):
        phase1 = RunConfig.from_yaml(ROOT / "configs/pilot_qwen_tp2.yaml")
        self.assertEqual(phase1.web_search_backend, "tavily")
        self.assertEqual(GAIA_CFG.web_search_backend, "tavily_then_ddg")
        # A non-default engine is a different cell; the default leaves Phase 1 ids alone.
        ddg = RunConfig.from_dict({**phase1.to_dict(), "web_search_backend": "ddg"})
        self.assertNotEqual(phase1.config_id("p", "t"), ddg.config_id("p", "t"))

    def test_tool_list_does_not_change_config_id(self):
        cfg = RunConfig.from_yaml(ROOT / "configs/pilot_qwen_tp2.yaml")
        other = RunConfig.from_dict({**cfg.to_dict(), "tools": ["read_file"]})
        self.assertEqual(cfg.config_id("p", "t"), other.config_id("p", "t"))

    @needs_data
    def test_prompt_names_attachment_and_format(self):
        t = next(t for t in json.loads(TEXT.read_text()) if t.get("file_name"))
        p = task_prompt(t)
        self.assertIn(t["file_name"], p)
        self.assertIn("FINAL ANSWER", p)


if __name__ == "__main__":
    unittest.main()
