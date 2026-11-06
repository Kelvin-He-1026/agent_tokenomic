"""Export Harbor trial results; preserve invalid, failed and missing attempts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


def duration(timing):
    if not timing or not timing.get('started_at') or not timing.get('finished_at'):
        return None
    return (datetime.fromisoformat(timing['finished_at']) - datetime.fromisoformat(timing['started_at'])).total_seconds()


def classify(result, reward_key='reward'):
    from harbor.models.trial.error_policy import policy_for

    rewards = (result.get('verifier_result') or {}).get('rewards') or {}
    raw_reward = rewards.get(reward_key)
    exceptions = [e for e in [result.get('exception_info')] if e]
    exceptions += [s['exception_info'] for s in result.get('step_results') or [] if s.get('exception_info')]
    valid = (isinstance(raw_reward,(int,float)) and not isinstance(raw_reward,bool)
             and math.isfinite(raw_reward) and raw_reward in (0,1)
             and all(policy_for(e.get('exception_type','')).reward_valid for e in exceptions))
    agent = result.get('agent_result') or {}
    metadata = agent.get('metadata') or {}
    completion = metadata.get('completion_type') or ('error' if exceptions else 'unknown')
    passed = raw_reward == 1 if valid else None
    natural = metadata.get('natural_completion')
    strict = (passed and natural and not exceptions) if natural is not None and valid else (False if passed is False else None)
    return {'task_name':result['task_name'].rsplit('/',1)[-1],
            'source_task_name':result['task_name'],'trial_name':result.get('trial_name',''),
            'agent':(result.get('agent_info') or {}).get('name'),
            'reward':raw_reward,'reward_valid':valid,'verifier_pass':passed,
            'strict_success':strict,'completion_type':completion,
            'natural_completion':natural,'budget_enforced':metadata.get('budget_enforced'),
            'forced_final_answer':metadata.get('forced_final_answer'),
            'error_type':'|'.join(e.get('exception_type','') for e in exceptions),
            'error_message':' | '.join(e.get('exception_message','') for e in exceptions),
            'input_tokens':agent.get('n_input_tokens'),'output_tokens':agent.get('n_output_tokens'),
            'cached_tokens':agent.get('n_cache_tokens'),'cost_usd':agent.get('cost_usd'),
            'trial_seconds':duration(result),'agent_seconds':duration(result.get('agent_execution')),
            'setup_seconds':duration(result.get('environment_setup')),'verifier_seconds':duration(result.get('verifier')),
            'n_llm_calls':metadata.get('n_llm_calls'),'n_tool_calls':metadata.get('n_tool_calls'),
            'llm_request_ms':metadata.get('llm_request_ms'),'terminal_wall_ms':metadata.get('terminal_wall_ms'),
            'call_cap_reached':metadata.get('call_cap_reached'),
            'per_call_output_cap_reached':metadata.get('per_call_output_cap_reached'),
            'total_output_cap_reached':metadata.get('total_output_cap_reached'),
            'output_cap_reached':metadata.get('output_cap_reached'),
            'output_cap_type':metadata.get('output_cap_type'),
            'usage_complete':metadata.get('usage_complete'),'raw_rewards':json.dumps(rewards),
            'row_state':'observed'}


def export(job_dir, plan=None, reward_key='reward'):
    job_dir = Path(job_dir)
    rows = []
    for path in sorted(job_dir.rglob('result.json')):
        data = json.loads(path.read_text())
        if 'task_name' not in data or 'agent_info' not in data:
            continue  # Job aggregate is also named result.json.
        row = classify(data,reward_key)
        row['result_path'] = str(path)
        rows.append(row)
    observed = len(rows)
    counts = Counter(r['task_name'] for r in rows)
    missing = []
    unexpected = []
    duplicates = {}
    if plan:
        for task in plan['task_names']:
            missing_count = max(0,plan['n_attempts'] - counts[task])
            missing.extend([task]*missing_count)
            for index in range(missing_count):
                rows.append({'task_name':task,'trial_name':f'missing-{index+1}',
                             'agent':plan['agent'],'row_state':'missing','reward_valid':False,
                             'completion_type':'missing','error_type':'MissingTrialResult'})
        unexpected = sorted(set(counts) - set(plan['task_names']))
        duplicates = {t:n for t,n in counts.items() if t in plan['task_names'] and n > plan['n_attempts']}
    valid = sum(r.get('reward_valid') is True for r in rows)
    passes = sum(r.get('verifier_pass') is True for r in rows)
    expected = plan['expected_trials'] if plan else observed
    summary = {'expected_trials':expected,'observed_trials':observed,'missing_trials':missing,
               'unexpected_tasks':unexpected,'excess_trials':duplicates,'valid_scored_trials':valid,
               'verifier_passes':passes,'verifier_pass_rate_all_planned':passes/expected if expected else None,
               'verifier_pass_rate_valid_only':passes/valid if valid else None,
               'strict_successes':sum(r.get('strict_success') is True for r in rows),
               'strict_success_unknown':sum(r.get('strict_success') is None for r in rows),
               'completion_types':dict(Counter(r.get('completion_type') for r in rows)),
               'coverage_complete':not missing and not unexpected and not duplicates and observed == expected,
               'note':'All-planned rate includes missing/invalid attempts in denominator. Valid-only rate is separate. Unknown costs/tokens/termination remain blank, never zero.'}
    return rows,summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job_dir',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--reward-key',default='reward')
    p.add_argument('--out',type=Path)
    args = p.parse_args(argv)
    if not args.job_dir.is_dir():
        p.error('job_dir does not exist')
    plan_path = args.plan or args.job_dir.parents[1] / 'plans' / (args.job_dir.name + '.json')
    plan = json.loads(plan_path.read_text()) if plan_path.is_file() else None
    rows,summary = export(args.job_dir,plan,args.reward_key)
    if not rows:
        p.error('No trial results or planned trials; specify --plan to retain missing attempts')
    output = args.out or args.job_dir.with_suffix('.csv')
    output.parent.mkdir(parents=True,exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with output.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    output.with_suffix('.quality.json').write_text(json.dumps(summary,indent=2))
    print(output)
    print(json.dumps(summary,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
