"""Private per-trial, non-streaming model gateway. No shared global run state.

Logs request bodies/responses, never HTTP credentials. Failed upstream requests
consume a call; denied requests do not. Unknown completion usage fails closed.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx


class BudgetProxy:
    def __init__(self, upstream, model, logs_dir, *, max_calls, per_call_tokens,
                 total_tokens, timeout=300, api_key='EMPTY', template_kwargs=None):
        if min(max_calls, per_call_tokens, total_tokens) <= 0:
            raise ValueError('Call and token limits must be positive')
        self.upstream = upstream.rstrip('/')
        self.model = model
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls, self.per_call_tokens, self.total_tokens = max_calls, per_call_tokens, total_tokens
        self.timeout, self.api_key = timeout, api_key
        self.template_kwargs = template_kwargs or {}
        self.calls = []
        self.denied = []
        self.usage_complete = True
        self.lock = threading.Lock()
        self.server = None

    def write(self, name, value):
        (self.logs_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))

    @property
    def output_tokens(self):
        return sum(c.get('completion_tokens') or 0 for c in self.calls)

    def handle(self, request):
        if not self.lock.acquire(blocking=False):
            return 409, {'error': {'message': 'Concurrent calls within one trial are unsupported'}}
        try:
            reason = ('usage_unknown' if not self.usage_complete else
                      'call_cap' if len(self.calls) >= self.max_calls else
                      'total_output_cap' if self.output_tokens >= self.total_tokens else '')
            if reason:
                self.denied.append({'reason': reason, 'time': time.time()})
                self.write('denied_requests.json', self.denied)
                return 429, {'error': {'message': reason}}
            if request.get('model') != self.model:
                return 400, {'error': {'message': 'Model differs from the pinned served name'}}
            if request.get('stream'):
                return 400, {'error': {'message': 'This adapter requires stream=false'}}
            index = len(self.calls) + 1
            remaining = self.total_tokens - self.output_tokens
            requested = request.get('max_completion_tokens', request.get('max_tokens', self.per_call_tokens))
            if not isinstance(requested, int) or requested <= 0:
                return 400, {'error': {'message': 'Invalid output cap'}}
            effective = min(requested, self.per_call_tokens, remaining)
            body = dict(request)
            key = 'max_completion_tokens' if 'max_completion_tokens' in body else 'max_tokens'
            if key == 'max_completion_tokens':
                body.pop('max_tokens', None)
            body[key] = effective
            body['stream'] = False
            body['chat_template_kwargs'] = self.template_kwargs
            self.write(f'{index:03d}_request.json', body)
            started = time.perf_counter()
            record = {'call_idx': index, 'request_ts': time.time(),
                      'requested_output_cap': requested, 'effective_output_cap': effective}
            try:
                with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                    response = client.post(self.upstream + '/chat/completions', json=body,
                                           headers={'Authorization': 'Bearer ' + self.api_key})
                status = response.status_code
                try:
                    result = response.json()
                except ValueError:
                    result = {'error': {'message': 'Upstream returned non-JSON content'}}
                usage = result.get('usage') or {}
                prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
                def valid_count(value):
                    return isinstance(value,int) and not isinstance(value,bool) and value >= 0
                complete = all(valid_count(v) for v in (prompt, completion))
                if not complete:
                    self.usage_complete = False
                prompt = prompt if valid_count(prompt) else None
                completion = completion if valid_count(completion) else None
                choices = result.get('choices') or []
                finish = choices[0].get('finish_reason') if choices else None
                message = (choices[0].get('message') or {}) if choices else {}
                details = usage.get('completion_tokens_details') or {}
                cache_details = usage.get('prompt_tokens_details') or {}
                safe_completion = completion if isinstance(completion,int) and not isinstance(completion,bool) else 0
                total_hit = self.output_tokens + safe_completion >= self.total_tokens
                length_hit = finish == 'length'
                per_hit = safe_completion >= self.per_call_tokens or (length_hit and effective == self.per_call_tokens)
                total_hit = total_hit or (length_hit and effective < self.per_call_tokens)
                record.update(http_status=status, prompt_tokens=prompt, completion_tokens=completion,
                              reasoning_tokens=details.get('reasoning_tokens'),
                              cached_tokens=cache_details.get('cached_tokens'), usage_missing=not complete,
                              n_tool_calls_requested=len(message.get('tool_calls') or []),
                              finish_reason=finish, per_call_output_cap_reached=per_hit,
                              total_output_cap_reached=total_hit,
                              output_cap_reached=per_hit or total_hit or length_hit)
            except Exception as exc:
                self.usage_complete = False
                status = 502
                # Do not persist exception strings that could contain credentials.
                result = {'error': {'message': type(exc).__name__}}
                record.update(http_status=status, usage_missing=True, error=type(exc).__name__)
            record['latency_ms'] = (time.perf_counter() - started) * 1000
            self.calls.append(record)
            self.write(f'{index:03d}_response.json', result)
            self.write('llm_calls.json', self.calls)
            return status, result
        finally:
            self.lock.release()

    def __enter__(self):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.path != '/v1/chat/completions':
                    status, body = 404, {'error': {'message': 'Unknown route'}}
                else:
                    try:
                        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                        if not isinstance(body, dict):
                            raise ValueError('Expected object')
                        status, body = proxy.handle(body)
                    except (ValueError, TypeError):
                        status, body = 400, {'error': {'message': 'Invalid JSON request'}}
                encoded = json.dumps(body).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f'http://127.0.0.1:{self.server.server_port}/v1'
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

