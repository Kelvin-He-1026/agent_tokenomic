"""Package the completed 30-task Hermes/LangGraph cell without changing sources."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output'
RAW = OUTPUT / 'runs/gaia_hermes_20261027_154255'
STEM = 'gaia_hermes_20261027_201558'
DEST = OUTPUT / 'downloads'
REVIEW = DEST / f'{STEM}_review'


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def truth(value):
    return str(value).lower() == 'true'


def fenced(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    fence = '```'
    while fence in text:
        fence += '`'
    return f'{fence}\n{text}\n{fence}\n'


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    REVIEW.mkdir(parents=True, exist_ok=True)
    step_dir = REVIEW / 'steps'
    step_dir.mkdir(exist_ok=True)
    rows = read_csv(OUTPUT / f'{STEM}.csv')
    runs = {r['run_id']: r for r in rows}
    assert len(runs) == len(rows) == 180
    assert len({r['task_id'] for r in rows}) == 30
    calls = defaultdict(list)
    for line in (RAW / 'llm_call.jsonl').read_text().splitlines():
        call = json.loads(line)
        calls[call['run_id']].append(call)
    transcript_rows = [json.loads(line) for line in (OUTPUT / f'{STEM}_transcripts.jsonl').read_text().splitlines()]
    transcripts = {r['run_id']: r for r in transcript_rows}
    assert set(transcripts) == set(runs) == set(calls)
    audit = {'tasks': 30, 'runs': len(rows), 'llm_calls': sum(map(len, calls.values())),
             'missing_blobs': [], 'invalid_blobs': [], 'call_count_mismatches': [],
             'request_context_rewrites': [], 'frameworks': {}}
    source_files = sorted(p for p in RAW.rglob('*') if p.is_file())
    export_files = sorted(p for p in OUTPUT.glob(f'{STEM}*') if p.is_file())
    if (OUTPUT / 'gaia_hermes.log').exists():
        export_files.append(OUTPUT / 'gaia_hermes.log')
    all_sources = source_files + export_files
    initial_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in all_sources}
    run_index = []

    for row in rows:
        run_id = row['run_id']
        ordered = sorted(calls[run_id], key=lambda c: c['call_idx'])
        if len(ordered) != int(row['n_llm_calls']):
            audit['call_count_mismatches'].append(run_id)
        trace = transcripts[run_id]
        filename = f"{row['framework']}__{row['task_id']}__repeat{row['repeat_idx']}__{run_id}.md"
        text = [f"# {row['framework']} / {row['task_id']} / repeat {row['repeat_idx']}\n",
                f"run_id: {run_id}\n\n",
                f"difficulty: {row['difficulty']} | status: {row['status']} | strict_success: {row['strict_success']}\n\n",
                f"completion_type: {row['completion_type']} | calls: {len(ordered)} | e2e_seconds: {float(row['e2e_ms'])/1000:.3f}\n\n",
                '## Question\n', fenced(trace.get('question', '')),
                '## Final answer\n', fenced(trace.get('final_answer', '')),
                '## Recorded error\n', fenced(row['error']),
                '\nThe messages below come directly from raw request blobs. Each step shows the request suffix after its longest common prefix with the preceding request. A rewritten context is explicitly marked. Full requests, tool schemas, and responses are in the raw-data ZIP.\n']
        previous = []
        for call in ordered:
            index = call['call_idx']
            request_file = RAW / 'blobs' / run_id / f'{index:03d}_request.json'
            response_file = RAW / 'blobs' / run_id / f'{index:03d}_response.json'
            loaded = []
            for path in [request_file, response_file]:
                if not path.is_file():
                    audit['missing_blobs'].append(str(path.relative_to(RAW)))
                    loaded.append({})
                else:
                    try:
                        loaded.append(json.loads(path.read_text()))
                    except json.JSONDecodeError:
                        audit['invalid_blobs'].append(str(path.relative_to(RAW)))
                        loaded.append({})
            request, response = loaded
            messages = request.get('messages') or []
            prefix = 0
            while prefix < min(len(previous), len(messages)) and previous[prefix] == messages[prefix]:
                prefix += 1
            rewrite = bool(previous) and prefix < len(previous)
            if rewrite:
                audit['request_context_rewrites'].append({'run_id': run_id, 'call_idx': index, 'common_prefix_messages': prefix})
            text.extend([f'\n## Step {index}\n',
                         f"HTTP: {call.get('http_status')} | finish_reason: {call.get('finish_reason')} | latency_ms: {call.get('latency_ms')} | input_tokens: {call.get('prompt_tokens')} | output_tokens: {call.get('completion_tokens')}\n\n",
                         f"Raw files: blobs/{run_id}/{index:03d}_request.json and {index:03d}_response.json\n\n",
                         f"Context rewritten: {rewrite}; unchanged prefix: {prefix} messages.\n\n",
                         '### New or replaced request messages\n'])
            for message in messages[prefix:]:
                text.extend([f"\nRole: {message.get('role', 'unknown')}\n\n", fenced(message)])
            text.extend(['\n### Model response (complete JSON)\n', fenced(response)])
            previous = messages
        (step_dir / filename).write_text(''.join(text))
        run_index.append({**row, 'steps_file': f'steps/{filename}'})

    summary = []
    for framework in sorted({r['framework'] for r in rows}):
        group = [r for r in rows if r['framework'] == framework]
        successful = sum(truth(r['strict_success']) for r in group)
        entry = {'framework': framework, 'runs': len(group), 'unique_tasks': len({r['task_id'] for r in group}),
                 'strict_successes': successful, 'strict_success_rate': successful / len(group),
                 'completion_types': dict(Counter(r['completion_type'] for r in group)),
                 'llm_calls': sum(int(r['n_llm_calls']) for r in group),
                 'total_agent_seconds': sum(float(r['e2e_ms']) / 1000 for r in group),
                 'mean_agent_seconds': statistics.mean(float(r['e2e_ms']) / 1000 for r in group),
                 'input_tokens': sum(int(r['input_tokens']) for r in group),
                 'output_tokens': sum(int(r['output_tokens']) for r in group)}
        audit['frameworks'][framework] = entry
        summary.append(entry)
    audit['texts_csv_truncated_rows'] = sum(bool(r.get('truncated_in_csv')) for r in read_csv(OUTPUT / f'{STEM}_texts.csv'))
    assert not audit['missing_blobs'] and not audit['invalid_blobs'] and not audit['call_count_mismatches']
    (REVIEW / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    fields = ['framework','task_id','difficulty','repeat_idx','run_id','status','success','strict_success','format_ok','completion_type','n_llm_calls','n_tool_calls','input_tokens','output_tokens','e2e_ms','error','steps_file']
    write_csv(REVIEW / 'run_index.csv', run_index, fields)
    task_summary = []
    for task_id in sorted({r['task_id'] for r in rows}):
        for framework in ['hermes', 'langgraph']:
            group = [r for r in rows if r['task_id'] == task_id and r['framework'] == framework]
            task_summary.append({'task_id':task_id,'difficulty':group[0]['difficulty'],'framework':framework,
                                 'runs':len(group),'strict_successes':sum(truth(r['strict_success']) for r in group),
                                 'mean_e2e_seconds':round(statistics.mean(float(r['e2e_ms'])/1000 for r in group),3),
                                 'mean_llm_calls':round(statistics.mean(int(r['n_llm_calls']) for r in group),3)})
    write_csv(REVIEW / 'task_summary.csv', task_summary, list(task_summary[0]))
    readme = '''# GAIA 30-task run — outputs and step review

Source run: gaia_hermes_20261027_154255
Source export: gaia_hermes_20261027_201558

30 distinct tasks, 10 per difficulty; Hermes and LangGraph each ran every task 3 times: 180 attempts. Qwen/Qwen3.8-27B, thinking off. This is a selected subset of the 68-task local GAIA text set, not full GAIA coverage.

| Runtime | Strict successes | Natural completions | Step limit | Framework error | LLM calls |
| --- | --- | --- | --- | --- | --- |
| Hermes | 28/90 (31.1%) | 34 | 39 | 17 | 1840 |
| LangGraph | 36/90 (40.0%) | 61 | 29 | 0 | 1349 |

All 3189 calls have both request and response JSON files (6378 blobs), and counts match all 180 run records. The original JSONL transcripts cover all 180 attempts. Four rows in the original texts CSV are truncated; use JSONL or the step files for full output. The step files are rebuilt from raw blobs and explicitly track request-context rewrites; the original transcript exporter used message-count slicing, which can omit replaced context.

The 17 Hermes framework errors record termination while a tool was requested, with a returned message about stopped web_search retries/tool-call guard. These records are not proof of one identical underlying network cause. Hermes also has 39 step-limit runs.

Hermes uses its own prompts, schemas, tools and live search services; it is a separate deployment arm rather than a framework-only comparison with matched tools. Hermes tool execution timing is not independently captured in tool_call.jsonl; it is part of orchestration time. Requested tool calls and subsequent returned tool messages are preserved in raw LLM request/response blobs. This bundle adds no missing sandbox-internal telemetry and does not rerun or rescore tasks.

Files:
- run_index.csv: all 180 attempts with links to per-run step documents.
- task_summary.csv: each of the 30 tasks, both runtimes, aggregate over 3 repeats.
- steps/: full per-call model responses, new/replaced request messages, tool arguments and returned tool messages when present.
- audit.json: machine-readable completeness audit and aggregate statistics.
- source_sha256.csv: SHA-256 and byte length for every original source file.
- Original exports: summary CSV, per-LLM-call CSV, texts CSV, full transcripts JSONL, quality JSON and driver log.

The raw ZIP preserves the source raw directory including plan/config/run/llm_call/tool_call/power JSONL and all blobs. Historical absolute paths inside original files are unchanged; after extraction resolve them using the blobs/<run_id>/ paths inside this archive.

Total agent seconds sum task durations, not a separately measured steady-state throughput window. Do not infer CPU/GPU bottlenecks or a statistically significant runtime effect from these descriptive counts alone.

GAIA questions/answers and traces remain private research data; do not publish them in a crawlable repository.
'''
    (REVIEW / 'README.md').write_text(readme)
    hash_rows = [{'source_path':str(p.relative_to(OUTPUT)),'bytes':p.stat().st_size,'sha256':initial_hashes[p]} for p in all_sources]
    write_csv(REVIEW / 'source_sha256.csv',hash_rows,['source_path','bytes','sha256'])
    raw_zip = DEST / f'{STEM}_raw.zip'
    outputs_zip = DEST / f'{STEM}_outputs_and_steps.zip'
    with zipfile.ZipFile(raw_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in source_files:
            archive.write(path, f'raw/{RAW.name}/{path.relative_to(RAW)}')
        archive.write(REVIEW / 'README.md','README.md')
        archive.write(REVIEW / 'source_sha256.csv','source_sha256.csv')
    with zipfile.ZipFile(outputs_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in export_files:
            archive.write(path,f'original_exports/{path.name}')
        for path in sorted(REVIEW.rglob('*')):
            if path.is_file():archive.write(path,f'review/{path.relative_to(REVIEW)}')
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == initial_hashes[p] for p in all_sources), 'Source changed during packaging'
    for path in [raw_zip,outputs_zip]:
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            print(json.dumps({'file':str(path),'bytes':path.stat().st_size,'entries':len(archive.infolist()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
    print(json.dumps({'tasks':30,'runs':180,'calls':audit['llm_calls'],'context_rewrite_calls':len(audit['request_context_rewrites']),'source_files':len(all_sources)}))


if __name__ == '__main__':
    main()
