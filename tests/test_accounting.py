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
        self.saved_state, self.saved_runs = dict(server.state), dict(server.runs)
        server.runs.clear()
        server.runs["unbound"] = server.RunState("unbound", self.directory.name)
        server.state.update(current="unbound", session="", last_bind=0.0)
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
        server.runs.clear()
        server.runs.update(self.saved_runs)
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
        server.runs["run-test"].deadline = 0
        response = await self.generate()
        self.assertEqual(response.json()["error"]["code"], "timeout")
        self.assertEqual(len(self.requests), 0)
        self.assertEqual(server.runs["run-test"].blocked_requests, 1)

    async def test_multiple_samples_and_concurrent_requests_cannot_bypass_budget(self):
        await self.bind()
        response = await self.generate(n=2)
        self.assertEqual(response.json()["error"]["code"], "concurrency_violation")
        self.assertEqual(len(self.requests), 0)

    async def test_forced_finalization_is_detected_on_the_wire(self):
        await self.bind()
        server.runs["run-test"].forced_final_markers = ["force final marker"]
        await self.generate(messages=[{"role": "user", "content": "force final marker"}])
        self.assertTrue(self.rows()[0]["forced_final_answer"])

    async def test_normal_user_and_tool_messages_do_not_mark_forced_finalization(self):
        await self.bind()
        server.runs["run-test"].forced_final_markers = ["force final marker"]
        await self.generate(messages=[{"role": "tool", "content": "force final marker"},
                                      {"role": "user", "content": "ordinary question"}])
        self.assertFalse(self.rows()[0]["forced_final_answer"])

    async def test_concurrent_runs_keep_separate_budgets_and_rows(self):
        for run_id in ("run-a", "run-b"):
            response = await self.client.post("/control/run", json={
                "run_id": run_id, "concurrent": True, "out_dir": self.directory.name, "session": "test",
                "limits": {"max_llm_calls": 1, "max_tokens_per_call": 5,
                           "max_total_output_tokens": 5, "task_timeout_s": 60}})
            self.assertEqual(response.status_code, 200)
        body = {"model": "mock", "messages": [{"role": "user", "content": "q"}]}
        self.assertEqual((await self.client.post("/r/run-a/v1/chat/completions", json=body)).status_code, 200)
        # The header route, which Hermes uses: the sandbox allows only /v1 paths.
        response = await self.client.post("/v1/chat/completions", json=body, headers={"X-Run-Id": "run-b"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual((await self.client.post("/r/run-a/v1/chat/completions", json=body)).status_code, 429)
        self.assertEqual(sorted(r["run_id"] for r in self.rows()), ["run-a", "run-b"])
        b = (await self.client.get("/control/pending", params={"run_id": "run-b"})).json()
        self.assertEqual((b["call_idx"], b["termination_reason"]), (1, ""))
        a = (await self.client.get("/control/pending", params={"run_id": "run-a"})).json()
        self.assertEqual(a["termination_reason"], "call_cap")
        # Concurrent binds leave the plain route on the run bound last.
        self.assertEqual(server.state["current"], "unbound")

    async def test_ended_run_refuses_requests_without_changing_its_ending(self):
        await self.bind(calls=5)
        await self.client.post("/control/end", json={"run_id": "run-test"})
        response = await self.generate()
        self.assertEqual(response.json()["error"]["code"], "run_ended")
        self.assertEqual(len(self.requests), 0)
        state = (await self.client.get("/control/pending")).json()
        self.assertEqual((state["termination_reason"], state["ended"]), ("", True))
        # Closed by the harness (no reason given): "finished"; the refusal is counted.
        self.assertEqual((state["ended_by"], state["refused_after_end"], state["upstream_errors"]),
                         ("finished", 1, 0))

    async def test_first_close_reason_is_kept(self):
        await self.bind(calls=5)
        await self.client.post("/control/end", json={"run_id": "run-test", "reason": "level_end"})
        await self.client.post("/control/end", json={"run_id": "run-test", "reason": "finished"})
        state = (await self.client.get("/control/pending")).json()
        self.assertEqual(state["ended_by"], "level_end")


class EnergyShareTests(unittest.TestCase):
    def test_gpu_follows_llm_calls_and_cpu_splits_among_runs_in_flight(self):
        from agent.energy_share import UNATTRIBUTED, apportion

        # 1 J/ms on each channel; run a 0-100, run b 50-150; only a calls
        # the LLM, 0-80; 150-200 has no run in flight.
        counters = {"gpu0": lambda t: t, "package-0": lambda t: t}
        out = apportion(counters, {"a": (0, 100), "b": (50, 150), "c": (200, 200)},
                        {"a": [(0, 80)]}, {"gpu0"})
        self.assertAlmostEqual(out["a"]["gpu0"], 80 + 10)  # its call, then half of 80-100
        self.assertAlmostEqual(out["b"]["gpu0"], 10 + 50)
        self.assertAlmostEqual(out["a"]["package-0"], 50 + 25)
        self.assertAlmostEqual(out["b"]["package-0"], 25 + 50)
        self.assertEqual(out[UNATTRIBUTED]["gpu0"], 0)
        self.assertAlmostEqual(sum(v["gpu0"] for v in out.values()), 150)


class StackPinningTests(unittest.TestCase):
    def test_cores_per_socket_take_the_gpu_node_first_then_its_neighbours(self):
        import tempfile

        from agent.stacks import cpulist, socket_cpus

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "cpu").mkdir()
            (root / "cpu/online").write_text("0-15\n")
            for c in range(16):
                (root / f"cpu/cpu{c}/topology").mkdir(parents=True)
                (root / f"cpu/cpu{c}/topology/physical_package_id").write_text(str(c // 8))
            for n, cpus in enumerate(("0-3", "4-7", "8-11", "12-15")):
                (root / f"node/node{n}").mkdir(parents=True)
                (root / f"node/node{n}/cpulist").write_text(cpus)
            self.assertEqual(cpulist(socket_cpus(6, (1, 3), root)), "2-7,10-15")
            with self.assertRaises(SystemExit):
                socket_cpus(9, (1, 3), root)


class SpanTests(unittest.TestCase):
    def test_spans_tile_the_run_and_name_orchestration(self):
        from agent.spans import build_spans

        calls = [{"call_id": "c1", "call_idx": 1, "request_ts": 100, "latency_ms": 50},
                 {"call_id": "c2", "call_idx": 2, "request_ts": 200, "latency_ms": 50}]
        tools = [{"tool_name": "web_search", "tool_call_id": "t1", "start_ts": 160, "latency_ms": 20}]
        spans = build_spans("r", (0, 300), calls, tools, [("hermes_boot", 10, 60)])
        self.assertEqual([(s.kind, s.name, s.start_ts, s.end_ts) for s in spans], [
            ("orch", "startup", 0, 10), ("orch", "hermes_boot", 10, 60), ("orch", "startup", 60, 100),
            ("llm", "call_1", 100, 150), ("orch", "loop", 150, 160), ("tool", "web_search", 160, 180),
            ("orch", "loop", 180, 200), ("llm", "call_2", 200, 250), ("orch", "finish", 250, 300)])
        self.assertEqual(sum(s.dur_ms for s in spans), 300)


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
