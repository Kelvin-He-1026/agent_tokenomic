"""NemoClaw Hermes adapter: completion mapping, in-sandbox script, and the SSH relay.

The relay test needs Docker, /usr/sbin/sshd and a local busybox image; it is
skipped otherwise. Nothing here touches the NemoClaw sandbox or a model.
"""
from __future__ import annotations

import getpass
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terminal_bench.nemoclaw_agent import _KILL, _RUN, NemoClawHermesAgent  # noqa: E402
from terminal_bench.ssh_relay import SSHRelay, _pids_with_arg  # noqa: E402


class CompletionTests(unittest.TestCase):
    def check(self, expected, rc=0, usage=None, snapshot=None, answer='done', stderr=''):
        status, _ = NemoClawHermesAgent._completion(rc, usage or {'completed': True}, snapshot or {}, answer, stderr)
        self.assertEqual(status, expected)

    def test_mapping(self):
        self.check('natural_completion')
        self.check('step_limit', usage={'completed': False})
        self.check('step_limit', snapshot={'termination_reason': 'call_cap'})
        self.check('truncated', snapshot={'termination_reason': 'output_cap'})
        self.check('timeout', snapshot={'termination_reason': 'timeout'})
        self.check('accounting_error', snapshot={'termination_reason': 'usage_missing'})
        # Hermes exits 0 and prints its error as the answer.
        self.check('error', usage={'failed': True, 'completed': False})
        self.check('error', rc=1)
        self.check('error', rc=97)


class ScriptTests(unittest.TestCase):
    def test_scripts_parse(self):
        for script in (_RUN, 'inst=$1\n' + _KILL):
            subprocess.run(['sh', '-n', '-c', script], check=True)
            subprocess.run(['bash', '-n', '-c', script], check=True)


def _image():
    if not (shutil.which('docker') and Path('/usr/sbin/sshd').exists() and shutil.which('ssh')):
        return None
    out = subprocess.run(['docker', 'images', '-q', 'busybox'], capture_output=True, text=True)
    return out.stdout.split()[0] if out.returncode == 0 and out.stdout.split() else None


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@unittest.skipUnless(_image(), 'needs docker, sshd, ssh and a local busybox image')
class RelayTests(unittest.TestCase):
    def setUp(self):
        self.cid = subprocess.run(['docker', 'run', '-d', '--rm', '--network', 'none', _image(), 'sleep', '300'],
                                  capture_output=True, text=True, check=True).stdout.strip()
        self.tmp = Path(tempfile.mkdtemp(prefix='tbrelay-'))
        self.port = _free_port()
        self.relay = SSHRelay(self.cid, self.tmp / 'relay', workdir='/tmp', env={'TB_X': 'a b'},
                              host='127.0.0.1', port=self.port).start()
        known = self.tmp / 'known_hosts'
        known.write_text(self.relay.known_hosts_line('relay') + '\n')
        self.ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'HostKeyAlias=relay', '-o', f'UserKnownHostsFile={known}',
                    '-o', 'StrictHostKeyChecking=yes', '-i', str(self.relay.client_key), '-p', str(self.port),
                    f'{getpass.getuser()}@127.0.0.1']

    def tearDown(self):
        self.relay.stop()
        subprocess.run(['docker', 'rm', '-f', self.cid], capture_output=True)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_ssh(self, *args, stdin=None, options=()):
        cmd = self.ssh[:-1] + list(options) + self.ssh[-1:] + list(args)
        return subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=30)

    def test_command_runs_in_container_with_exec_context(self):
        r = self.run_ssh("pwd; echo $TB_X; cat > f; wc -c < f; exit 3", stdin='x' * 5000)
        self.assertEqual(r.returncode, 3)
        self.assertEqual(r.stdout.split('\n')[:3], ['/tmp', 'a b', '5000'])
        rows = (self.tmp / 'relay/relay_calls.jsonl').read_text().splitlines()
        self.assertEqual(len(rows), 1)

    def test_shell_and_forwarding_refused(self):
        self.assertEqual(self.run_ssh(stdin='').returncode, 2)
        self.assertNotEqual(self.run_ssh(stdin='', options=('-W', '127.0.0.1:22')).returncode, 0)

    def test_stop_ends_running_commands(self):
        p = subprocess.Popen(self.ssh + ['sleep 200'], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        time.sleep(1.5)
        self.relay.stop()
        self.assertIsNotNone(p.wait(timeout=15))
        self.assertEqual(_pids_with_arg(str(self.tmp / 'relay/relay.env').encode()), [])


if __name__ == '__main__':
    unittest.main()
