"""Exercise the actual proxy with an in-process mock upstream; no network/GPU."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from agent.config import LimitsConfig, ModelConfig, RunConfig, SamplingConfig
from agent.harness.runner import Runner
from agent.proxy import server
from agent.telemetry import RunRecord


class ProxyBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.saved_state = dict(server.state)
        server.state.update(run_id="unbound", session="", inflight=0, call_idx=0)
        self.requests = []
        self.fail_transport = False
        self.omit_usage = False
        self.http_status = 200

        def upstream(request):
            payload = json.loads(request.content)
            self.requests.append(payload)
            if self.fail_transport:
                raise httpx.ReadError("test transport error", request=request)
            count = min(5, payload["max_completion_tokens"])
            usage = {"prompt_tokens": 11, "completion_tokens": count,
                     "completion_tokens_details": {"reasoning_tokens": min(2, count)}}
            data = {"choices": [{"message": {"content": "2250"}, "finish_reason": "stop"}]}
            if not self.omit_usage:
                data["usage"] = usage
            if payload.get("stream"):
                events = [{"choices": [{"delta": {"content": "2250"}, "finish_reason": "stop"}]}]
                if not self.omit_usage:
                    events.append({"choices": [], "usage": usage})
                content = "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"
                return httpx.Response(self.http_status, text=content, headers={"content-type": "text/event-stream"})
            return httpx.Response(self.http_status, json=data)

        self.upstream = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        self.client_patch = patch.object(server, "client", self.upstream)
        self.client_patch.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.upstream.aclose()
        self.client_patch.stop()
        server.state.clear()
        server.state.update(self.saved_state)
        server._writers.pop(self.directory.name, None)
        self.directory.cleanup()

    async def bind(self, calls=2, per_call=5, total=10):
        response = await self.client.post("/control/run", json={
            "run_id": "run-test", "out_dir": self.directory.name, "session": "test",
            "limits": {"max_llm_calls": calls, "max_tokens_per_call": per_call,
                       "max_total_output_tokens": total, "task_timeout_s": 60}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["accounting_policy"], "proxy-v1")

    async def generate(self, **kwargs):
        return await self.client.post("/v1/chat/completions", json={
            "model": "mock", "messages": [{"role": "user", "content": "question"}], **kwargs})

    def rows(self):
        path = Path(self.directory.name) / "llm_call.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    async def test_call_cap_counts_upstream_calls_and_separates_rejections(self):
        await self.bind()
        self.assertEqual((await self.generate(max_tokens=999)).status_code, 200)
        self.assertEqual((await self.generate(max_completion_tokens=1)).status_code, 200)
        self.assertEqual((await self.generate()).status_code, 429)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual([r["max_completion_tokens"] for r in self.requests], [5, 5])
        self.assertTrue(all("max_tokens" not in r for r in self.requests))
        self.assertEqual(len(self.rows()), 2)
        state = (await self.client.get("/control/pending")).json()
        self.assertEqual(state["blocked_requests"], 1)
        self.assertEqual(state["termination_reason"], "call_cap")
        self.assertEqual(state["output_tokens"], 10)

    async def test_total_output_cap_includes_reasoning_and_shrinks_last_request(self):
        await self.bind(calls=10, per_call=5, total=7)
        await self.generate()
        await self.generate()
        response = await self.generate()
        self.assertEqual(response.json()["error"]["code"], "output_cap")
        self.assertEqual([r["max_completion_tokens"] for r in self.requests], [5, 2])
        self.assertEqual(sum(r["completion_tokens"] for r in self.rows()), 7)
        self.assertEqual(sum(r["reasoning_tokens"] for r in self.rows()), 4)

    async def test_streamed_calls_obey_same_caps_and_persist_usage(self):
        await self.bind(calls=1, total=5)
        response = await self.generate(stream=True)
        self.assertIn("[DONE]", response.text)
        self.assertTrue(self.requests[0]["stream_options"]["include_usage"])
        row = self.rows()[0]
        self.assertEqual(row["completion_tokens"], 5)
        self.assertEqual(row["reasoning_tokens"], 2)
        self.assertEqual(row["finish_reason"], "stop")
        self.assertEqual((await self.generate(stream=True)).status_code, 429)
        self.assertEqual((await self.client.get("/control/pending")).json()["inflight"], 0)

    async def test_failed_transport_is_counted_logged_and_flushed(self):
        await self.bind(calls=1)
        self.fail_transport = True
        with self.assertRaises(httpx.ReadError):
            await self.generate()
        self.assertIn("ReadError", self.rows()[0]["error"])
        self.assertTrue(self.rows()[0]["usage_missing"])
        self.assertEqual((await self.client.get("/control/pending")).json()["inflight"], 0)
        self.assertEqual((await self.generate()).status_code, 429)
        self.assertEqual(len(self.requests), 1)

    async def test_failed_stream_also_persists_a_call_and_releases_inflight(self):
        await self.bind(calls=1)
        self.fail_transport = True
        with self.assertRaises(Exception):
            await self.generate(stream=True)
        self.assertEqual(len(self.rows()), 1)
        self.assertIn("ReadError", self.rows()[0]["error"])
        self.assertEqual((await self.client.get("/control/pending")).json()["inflight"], 0)

    async def test_missing_usage_is_explicit_and_blocks_further_unaccounted_generation(self):
        await self.bind(calls=10)
        self.omit_usage = True
        await self.generate()
        self.assertTrue(self.rows()[0]["usage_missing"])
        response = await self.generate()
        self.assertEqual(response.json()["error"]["code"], "usage_missing")
        self.assertEqual(len(self.requests), 1)

    async def test_deadline_blocks_new_generation_without_charging_a_model_call(self):
        await self.bind()
        server.state["deadline"] = 0
        response = await self.generate()
        self.assertEqual(response.json()["error"]["code"], "timeout")
        self.assertEqual(len(self.requests), 0)
        self.assertEqual(server.state["blocked_requests"], 1)

    async def test_multiple_samples_and_concurrent_requests_cannot_bypass_budget(self):
        await self.bind()
        response = await self.generate(n=2)
        self.assertEqual(response.json()["error"]["code"], "concurrency_violation")
        self.assertEqual(len(self.requests), 0)

    async def test_forced_finalization_is_detected_on_the_wire(self):
        await self.bind()
        server.state["forced_final_markers"] = ["force final marker"]
        await self.generate(messages=[{"role": "user", "content": "force final marker"}])
        self.assertTrue(self.rows()[0]["forced_final_answer"])

    async def test_normal_user_and_tool_messages_do_not_mark_forced_finalization(self):
        await self.bind()
        server.state["forced_final_markers"] = ["force final marker"]
        await self.generate(messages=[{"role": "tool", "content": "force final marker"},
                                      {"role": "user", "content": "ordinary question"}])
        self.assertFalse(self.rows()[0]["forced_final_answer"])


class TerminationTests(unittest.TestCase):
    def record(self, **kwargs):
        return RunRecord("run", "cfg", "task", 0, 0, 0, **kwargs)

    def test_natural_completion_at_last_call_remains_successful(self):
        record = self.record(max_llm_calls=2, n_llm_calls=2, call_cap_reached=True)
        Runner._classify_ending(record, [{"finish_reason": "stop"}])
        self.assertEqual((record.status, record.termination_reason), ("completed", "natural_stop"))

    def test_forced_finalization_uses_same_completion_rule_and_keeps_provenance(self):
        record = self.record(forced_final_answer=True, final_answer="2250", framework_status="completed")
        Runner._classify_ending(record, [{"finish_reason": "stop"}])
        self.assertEqual((record.status, record.termination_reason), ("completed", "forced_final_answer"))
        self.assertEqual(record.final_answer, "2250")
        self.assertEqual(record.framework_status, "completed")

    def test_tool_request_before_and_at_cap_are_distinguished(self):
        for at_cap in (True, False):
            record = self.record(max_llm_calls=10, call_cap_reached=at_cap)
            Runner._classify_ending(record, [{"finish_reason": "tool_calls"}])
            self.assertEqual(record.termination_reason, "call_cap" if at_cap else "incomplete_trajectory")

    def test_length_is_distinguished_from_natural_cap_completion(self):
        for at_cap in (True, False):
            record = self.record(output_cap_reached=True, total_output_cap_reached=at_cap)
            Runner._classify_ending(record, [{"finish_reason": "length"}])
            self.assertEqual(record.status, "truncated")
            self.assertEqual(record.termination_reason, "output_cap" if at_cap else "per_call_output_cap")

    def test_missing_telemetry_and_upstream_errors_cannot_complete(self):
        record = self.record(budget_enforced=True)
        Runner._classify_ending(record, [])
        self.assertEqual(record.termination_reason, "telemetry_missing")
        record = self.record()
        Runner._classify_ending(record, [{"error": "HTTP 500"}])
        self.assertEqual(record.termination_reason, "upstream_error")
        self.assertEqual(record.status, "framework_error")

    def test_config_records_resource_policy_and_validates_caps(self):
        cfg = RunConfig("langgraph", ModelConfig("mock"))
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(cfg, SimpleNamespace(name="langgraph", framework_version=lambda: "test"), directory)
            runner.write_config()
            record = runner.writer.read("config")[0]
            self.assertEqual(record["max_total_output_tokens"], 10240)
            self.assertEqual(record["accounting_policy"], "proxy-v1")
        for cap in (0, -1, True):
            with self.assertRaises(ValueError):
                LimitsConfig(max_llm_calls=cap)
            with self.assertRaises(ValueError):
                SamplingConfig(max_tokens=cap)


if __name__ == "__main__":
    unittest.main()
