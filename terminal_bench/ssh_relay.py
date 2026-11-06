"""Per-trial SSH relay: NemoClaw Hermes's terminal reaches the Harbor task container.

Hermes in the NemoClaw sandbox runs its terminal and file tools over its SSH
backend (terminal.backend: ssh). The relay is an unprivileged OpenSSH sshd on
the Docker bridge address the sandbox already uses for the model route
(host.openshell.internal = 172.18.0.1). Its ForceCommand (relay/relay_exec.sh)
runs each command in this trial's task container, and nothing else:

  sandbox ssh -> socat -> OpenShell egress proxy (policy: terminal-bench-relay)
      -> 172.18.0.1:2222 sshd (this user, one fresh client key per trial)
      -> relay_exec.sh -> docker exec -i <task container> sh -c <command>

Port, TTY, agent and X11 forwarding, user rc files and shells are disabled;
the client key is valid only for this trial's relay, which stops with the
trial. The sandbox gets no Docker access and the task container gets no
network route to the sandbox.
"""
from __future__ import annotations

import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

TB = Path(__file__).resolve().parent
RELAY_EXEC = TB / 'relay/relay_exec.sh'
HOST_KEY = TB / '.relay/ssh_host_ed25519_key'
SSHD = '/usr/sbin/sshd'
DEFAULT_HOST, DEFAULT_PORT = '172.18.0.1', 2222


def _keygen(path: Path, comment: str) -> None:
    subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-C', comment, '-f', str(path)],
                   check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


def host_public_key() -> str:
    """The relay's host key, created once and kept, so sandbox known_hosts stays valid."""
    if not HOST_KEY.exists():
        HOST_KEY.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _keygen(HOST_KEY, 'terminal-bench-relay')
    return HOST_KEY.with_suffix('.pub').read_text().strip()


def _ppids() -> dict[int, int]:
    table = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                stat = (entry / 'stat').read_text()
                table[int(entry.name)] = int(stat.rsplit(')', 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
    return table


def _descendants(root: int) -> set[int]:
    table, found, frontier = _ppids(), set(), {root}
    while frontier:
        frontier = {p for p, parent in table.items() if parent in frontier and p not in found}
        found |= frontier
    return found


def _pgid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _pids_with_arg(arg: bytes) -> list[int]:
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if arg in (entry / 'cmdline').read_bytes().split(b'\0'):
                found.append(int(entry.name))
        except OSError:
            continue
    return found


class SSHRelay:
    def __init__(self, container: str, trial_dir: Path, *, workdir: str | None = None,
                 user: str | None = None, env: dict[str, str] | None = None,
                 host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        if not container or any(c.isspace() for c in container):
            raise ValueError('container id required')
        self.container, self.workdir, self.user = container, workdir or '', '' if user is None else str(user)
        self.env = env or {}
        self.host, self.port = host, port
        self.dir = Path(trial_dir).resolve()
        self.proc: subprocess.Popen | None = None

    # Files the sandbox needs: its private key, the relay's host key, an ssh_config entry.
    @property
    def client_key(self) -> Path:
        return self.dir / 'id_ed25519'

    def known_hosts_line(self, alias: str) -> str:
        return f'{alias} {" ".join(host_public_key().split()[:2])}'

    def start(self) -> 'SSHRelay':
        if self.proc is not None:
            raise RuntimeError('relay already started')
        if self.dir.exists():
            shutil.rmtree(self.dir)
        (self.dir / 'cmds').mkdir(parents=True)
        os.chmod(self.dir, 0o700)
        host_public_key()
        _keygen(self.client_key, 'terminal-bench-trial')
        pub = self.client_key.with_suffix('.pub').read_text().strip()
        # restrict: no forwarding, PTY, agent, X11 or user rc, whatever sshd_config says.
        (self.dir / 'authorized_keys').write_text(f'restrict {pub}\n')
        env_file = self.dir / 'task.env'
        for key, value in self.env.items():
            if not key.replace('_', 'a').isalnum() or '\n' in str(value):
                raise ValueError(f'unsupported task env entry {key!r}')
        env_file.write_text(''.join(f'{k}={v}\n' for k, v in self.env.items()))
        (self.dir / 'relay.env').write_text(''.join(
            f'{k}={shlex.quote(str(v))}\n' for k, v in {
                'CONTAINER': self.container, 'WORKDIR': self.workdir, 'EXEC_USER': self.user,
                'ENV_FILE': env_file, 'LOG': self.dir / 'relay_calls.jsonl'}.items()))
        config = self.dir / 'sshd_config'
        config.write_text('\n'.join([
            f'ListenAddress {self.host}:{self.port}',
            f'HostKey {HOST_KEY}',
            f'PidFile {self.dir / "sshd.pid"}',
            f'AuthorizedKeysFile {self.dir / "authorized_keys"}',
            f'AllowUsers {os.environ.get("USER") or os.getlogin()}',
            'PubkeyAuthentication yes', 'PasswordAuthentication no', 'KbdInteractiveAuthentication no',
            'UsePAM no', 'StrictModes no', 'PermitRootLogin no',
            'AllowTcpForwarding no', 'AllowStreamLocalForwarding no', 'AllowAgentForwarding no',
            'X11Forwarding no', 'PermitTunnel no', 'PermitTTY no', 'PermitUserRC no',
            'PermitUserEnvironment no', 'GatewayPorts no', 'DisableForwarding yes',
            # Hermes multiplexes every command over one ControlMaster connection.
            'MaxSessions 64', 'MaxStartups 16:30:64', 'ClientAliveInterval 30',
            f'ForceCommand {shlex.quote(str(RELAY_EXEC))} {shlex.quote(str(self.dir / "relay.env"))}',
            'LogLevel ERROR', '']))
        with open(self.dir / 'sshd.log', 'ab') as log:
            self.proc = subprocess.Popen([SSHD, '-D', '-e', '-f', str(config)], stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f'relay sshd exited: {(self.dir / "sshd.log").read_text()[-400:]}')
            try:
                with socket.create_connection((self.host, self.port), timeout=0.5):
                    return self
            except OSError:
                time.sleep(0.1)
        self.stop()
        raise RuntimeError(f'relay sshd did not listen on {self.host}:{self.port}')

    def stop(self) -> None:
        """Stop sshd, its sessions, and any relayed command still running.

        Per-connection sshd processes and the commands they run call setsid(),
        so they leave sshd's process group; collect every descendant first,
        then terminate each one's whole group (bash defers SIGTERM while it
        waits on its foreground docker exec, so the group must go together).
        """
        if self.proc is None:
            return
        victims = _descendants(self.proc.pid) | set(_pids_with_arg(str(self.dir / 'relay.env').encode()))
        groups = {g for g in (_pgid(p) for p in victims) if g and g != os.getpgid(0)}
        groups.add(self.proc.pid)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for g in groups:
                try:
                    os.killpg(g, sig)
                except (ProcessLookupError, PermissionError):
                    pass
            try:
                self.proc.wait(timeout=5)
                if not any(_alive(p) for p in victims):
                    break
            except subprocess.TimeoutExpired:
                continue
        self.proc = None
