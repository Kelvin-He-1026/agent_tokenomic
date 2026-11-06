"""Offline regression checks: no Docker, model, dataset or external endpoint."""
from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext

from terminal_bench.agents import ProjectReActAgent
from terminal_bench.budget_proxy import BudgetProxy
from terminal_bench.export_results import classify, export
from terminal_bench.run import build_job, make_parser


class Upstream:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass

            def do_POST(self):
                owner.requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                status,body = next(owner.responses)
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(payload)))
                self.end_headers();self.wfile.write(payload)

        self.server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'
        return self

    def __exit__(self,*args):
        self.server.shutdown();self.server.server_close();self.thread.join()


def reply(tokens,finish='stop',message=None):
    return 200,{'id':'mock','object':'chat.completion','model':'Qwen/Qwen3.8-27B',
                'choices':[{'index':0,'finish_reason':finish,'message':message or {'role':'assistant','content':'done'}}],
                'usage':{'prompt_tokens':3,'completion_tokens':tokens,'total_tokens':3+tokens}}


class BudgetTests(unittest.TestCase):
    def test_total_budget_lowers_next_cap_and_denies_further_calls(self):
        with tempfile.TemporaryDirectory() as tmp, Upstream([reply(7),reply(5)]) as server:
            proxy = BudgetProxy(server.url,'q',tmp,max_calls=4,per_call_tokens=10,total_tokens=12)
            for _ in range(2):
                self.assertEqual(proxy.handle({'model':'q','max_tokens':10})[0],200)
            self.assertEqual([r['max_tokens'] for r in server.requests],[10,5])
            self.assertEqual(proxy.handle({'model':'q'})[0],429)
            self.assertEqual(proxy.denied[-1]['reason'],'total_output_cap')
            self.assertEqual(proxy.output_tokens,12)
            self.assertFalse(proxy.calls[-1]['per_call_output_cap_reached'])
            self.assertTrue(proxy.calls[-1]['total_output_cap_reached'])

    def test_per_call_length_flag_independent_of_total_budget(self):
        with tempfile.TemporaryDirectory() as tmp, Upstream([reply(4,'length')]) as server:
            proxy=BudgetProxy(server.url,'q',tmp,max_calls=2,per_call_tokens=8,total_tokens=30)
            proxy.handle({'model':'q'})
            self.assertTrue(proxy.calls[0]['per_call_output_cap_reached'])
            self.assertFalse(proxy.calls[0]['total_output_cap_reached'])

    def test_failed_requests_count_and_unknown_usage_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp, Upstream([(500,{'error':{'message':'failed'}})]) as server:
            proxy=BudgetProxy(server.url,'q',tmp,max_calls=2,per_call_tokens=8,total_tokens=30)
            self.assertEqual(proxy.handle({'model':'q'})[0],500)
            self.assertEqual(len(proxy.calls),1)
            self.assertEqual(proxy.handle({'model':'q'})[0],429)
            self.assertEqual(proxy.denied[-1]['reason'],'usage_unknown')

    def test_call_cap_and_token_field(self):
        with tempfile.TemporaryDirectory() as tmp, Upstream([reply(1)]) as server:
            proxy=BudgetProxy(server.url,'q',tmp,max_calls=1,per_call_tokens=8,total_tokens=30)
            proxy.handle({'model':'q','max_completion_tokens':20,'max_tokens':30})
            self.assertEqual(server.requests[0]['max_completion_tokens'],8)
            self.assertNotIn('max_tokens',server.requests[0])
            self.assertEqual(proxy.handle({'model':'q'})[0],429)
            self.assertEqual(proxy.denied[-1]['reason'],'call_cap')

    def test_model_and_streaming_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            proxy=BudgetProxy('http://127.0.0.1:1/v1','q',tmp,max_calls=1,per_call_tokens=8,total_tokens=30)
            self.assertEqual(proxy.handle({'model':'other'})[0],400)
            self.assertEqual(proxy.handle({'model':'q','stream':True})[0],400)
            self.assertEqual(len(proxy.calls),0)

    def test_malformed_usage_remains_unknown_and_raw_response_is_preserved(self):
        status,response=reply(1)
        response['usage']['completion_tokens']='unknown'
        with tempfile.TemporaryDirectory() as tmp, Upstream([(status,response)]) as server:
            proxy=BudgetProxy(server.url,'q',tmp,max_calls=2,per_call_tokens=8,total_tokens=30)
            self.assertEqual(proxy.handle({'model':'q'})[0],200)
            self.assertIsNone(proxy.calls[0]['completion_tokens'])
            self.assertFalse(proxy.usage_complete)
            self.assertEqual(proxy.handle({'model':'q'})[0],429)
            original=json.loads((Path(tmp)/'001_response.json').read_text())
            self.assertEqual(original['usage']['completion_tokens'],'unknown')


class AdapterTest(unittest.IsolatedAsyncioTestCase):
    async def test_graph_executes_tool_in_environment_and_logs_full_exchange(self):
        class Environment:
            def __init__(self):self.commands=[]
            async def exec(self,command,timeout_sec):
                self.commands.append(command)
                return ExecResult(stdout='338350\n',stderr='',return_code=0)

        tool_message={'role':'assistant','content':None,'tool_calls':[{'id':'call_1','type':'function',
                      'function':{'name':'terminal','arguments':json.dumps({'command':'compute squares'})}}]}
        with tempfile.TemporaryDirectory() as tmp, Upstream([reply(4,'tool_calls',tool_message),reply(3)]) as server:
            from terminal_bench.agents import ROOT
            import yaml
            cfg=yaml.safe_load((ROOT/'terminal_bench/configs/qwen_tp2.yaml').read_text())
            cfg['base_url']=server.url
            cfg['limits']['max_llm_calls']=2
            cfg['limits']['max_total_output_tokens']=10
            path=Path(tmp)/'config.json';path.write_text(json.dumps(cfg))
            agent=ProjectReActAgent(logs_dir=Path(tmp)/'logs',cell_config=path)
            env=Environment();ctx=AgentContext()
            await agent.setup(env)
            await agent.run('Run a terminal command.',env,ctx)
            self.assertEqual(env.commands,['compute squares'])
            self.assertEqual(ctx.n_output_tokens,7)
            self.assertEqual(ctx.metadata['completion_type'],'natural_completion')
            self.assertEqual(ctx.metadata['n_llm_calls'],2)
            self.assertTrue(any(m['role']=='tool' for m in server.requests[1]['messages']))
            self.assertEqual(server.requests[0]['chat_template_kwargs']['enable_thinking'],False)
            self.assertTrue((Path(tmp)/'logs/raw/002_response.json').exists())

    async def test_call_cap_is_measured_termination_and_keeps_verifier_eligible(self):
        class Environment:
            async def exec(self,command,timeout_sec):
                return ExecResult(stdout='done',stderr='',return_code=0)

        message={'role':'assistant','content':None,'tool_calls':[{'id':'t1','type':'function',
                  'function':{'name':'terminal','arguments':'{"command":"test"}'}}]}
        with tempfile.TemporaryDirectory() as tmp, Upstream([reply(2,'tool_calls',message)]) as server:
            from terminal_bench.agents import ROOT
            import yaml
            cfg=yaml.safe_load((ROOT/'terminal_bench/configs/qwen_tp2.yaml').read_text())
            cfg['base_url']=server.url;cfg['limits']['max_llm_calls']=1
            path=Path(tmp)/'cell.json';path.write_text(json.dumps(cfg))
            agent=ProjectReActAgent(logs_dir=Path(tmp)/'logs',cell_config=path)
            ctx=AgentContext();await agent.setup(Environment())
            await agent.run('Execute the task.',Environment(),ctx)
            # LangGraph's built-in step guard replaces the requested tool call
            # with a synthetic final message. This must not count as natural.
            self.assertEqual(ctx.metadata['completion_type'],'forced_final')
            self.assertTrue(ctx.metadata['forced_final_answer'])
            self.assertTrue(ctx.metadata['call_cap_reached'])
            self.assertFalse(ctx.metadata['natural_completion'])
            self.assertEqual(len(server.requests),1)


class ExportTests(unittest.TestCase):
    def row(self,reward):
        return {'task_name':'test','trial_name':'trial','agent_info':{'name':'react'},
                'verifier_result':{'rewards':{'reward':reward}},
                'agent_result':{'metadata':{'natural_completion':True,'completion_type':'natural_completion'}}}

    def test_verifier_and_termination_separate(self):
        r=self.row(1);r['agent_result']['metadata'].update(natural_completion=False,completion_type='step_limit')
        scored=classify(r)
        self.assertTrue(scored['verifier_pass'])
        self.assertFalse(scored['strict_success'])

    def test_missing_nan_and_nonbinary_reward_not_success(self):
        for reward in [None,float('nan'),0.5,'1']:
            with self.subTest(reward=reward):
                row=classify(self.row(reward))
                self.assertFalse(row['reward_valid'])
                self.assertIsNone(row['verifier_pass'])
        self.assertFalse(classify(self.row(0))['verifier_pass'])

    def test_unknown_termination_is_not_natural(self):
        r=self.row(1);r['agent_result']={}
        self.assertIsNone(classify(r)['strict_success'])

    def test_qualified_task_names_normalized_for_plan_coverage(self):
        r=self.row(1);r['task_name']='terminal-bench/test'
        self.assertEqual(classify(r)['task_name'],'test')

    def test_export_includes_missing_trials_and_invalid_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trial';p.mkdir()
            (p/'result.json').write_text(json.dumps(self.row(1)))
            rows,summary=export(tmp,{'task_names':['test'],'n_attempts':3,'expected_trials':3,'agent':'react'})
            self.assertEqual(len(rows),3)
            self.assertEqual(summary['observed_trials'],1)
            self.assertEqual(summary['verifier_pass_rate_all_planned'],1/3)
            self.assertEqual(summary['verifier_pass_rate_valid_only'],1)
            self.assertFalse(summary['coverage_complete'])


class PlanTests(unittest.TestCase):
    def test_fixed_dataset_preview_and_resource_controls(self):
        args=make_parser().parse_args(['--agent','react','--suite','pilot','--task-cpus','4','--concurrency','2'])
        job,plan,cell,_=build_job(args)
        self.assertEqual(plan['dataset_version'],'4.1.0')
        self.assertTrue(job.datasets[0].version == '4.1.0' or job.datasets[0].ref)
        self.assertIn('terminal-bench/html-js-filter',job.datasets[0].task_names)
        self.assertEqual(job.environment.override_cpus,4)
        self.assertEqual(plan['expected_trials'],36)
        self.assertEqual(job.retry.max_retries,0)
        self.assertEqual(cell['limits']['max_llm_calls'],200)

    def test_native_hermes_requires_explicit_container_endpoint(self):
        args=make_parser().parse_args(['--agent','hermes-native'])
        with self.assertRaisesRegex(ValueError,'container-base-url'):
            build_job(args)

    def test_local_task_schema_valid(self):
        from harbor.models.task.task import Task
        from terminal_bench.agents import ROOT
        task=Task(task_dir=ROOT/'terminal_bench/fixtures/terminal-smoke')
        self.assertEqual(task.config.environment.gpus,0)


if __name__=='__main__':
    unittest.main()
