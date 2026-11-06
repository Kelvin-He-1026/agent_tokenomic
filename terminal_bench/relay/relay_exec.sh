#!/usr/bin/env bash
# ForceCommand of the per-trial SSH relay (terminal_bench/ssh_relay.py).
#
# Every SSH session the relay accepts runs this script on the host, whatever
# the client asked for. It runs the client's command in the trial's task
# container, the way Harbor's own environment.exec does (docker compose exec
# on the main service: its workdir, environment and user), with stdin passed
# through so tar pipes and heredoc writes work:
#
#   docker exec -i [-w WORKDIR] [-u USER] [--env-file ENV] <container> sh -c "$SSH_ORIGINAL_COMMAND"
#
# sh -c mirrors what sshd does on a real host (the login shell runs the
# command line). Sessions without a command (an interactive shell) and
# subsystems (sftp) are refused. $1 is the trial's relay.env, written by the
# relay: CONTAINER, WORKDIR, EXEC_USER, ENV_FILE, LOG.
set -u
conf="${1:?relay.env path}"
# shellcheck source=/dev/null
. "$conf"
cmd="${SSH_ORIGINAL_COMMAND:-}"
if [[ -z "$cmd" || "$cmd" == "internal-sftp" || "$cmd" == "sftp" || "$cmd" == */sftp-server ]]; then
  echo "relay: only commands are supported (no shell, sftp or scp sessions)" >&2
  exit 2
fi
args=(exec -i)
[[ -n "${WORKDIR:-}" ]] && args+=(-w "$WORKDIR")
[[ -n "${EXEC_USER:-}" ]] && args+=(-u "$EXEC_USER")
[[ -s "${ENV_FILE:-}" ]] && args+=(--env-file "$ENV_FILE")
t0=$(date +%s%3N)
docker "${args[@]}" "$CONTAINER" sh -c "$cmd"
rc=$?
t1=$(date +%s%3N)
# One line per command: timing, exit code, command size. The full command
# text goes to cmds/, named by start time and PID (commands can be large).
[[ -n "${LOG:-}" ]] && {
  printf '%s' "$cmd" > "$(dirname "$LOG")/cmds/$t0-$$.sh" 2>/dev/null
  printf '{"t0":%s,"t1":%s,"rc":%s,"cmd_bytes":%s,"cmd_file":"cmds/%s-%s.sh"}\n' \
    "$t0" "$t1" "$rc" "${#cmd}" "$t0" "$$" >> "$LOG"
}
exit $rc
