#!/usr/bin/env bash
# Prepare a NemoClaw Hermes sandbox for benchmark runs. Safe to rerun.
#
#   ./scripts/setup_sandbox.sh [configs/gaia_hermes_qwen_tp2.yaml]
#
# 1. Creates the isolated Hermes profile named in the config (hermes.profile),
#    cloned from the sandbox's default profile, so the NemoClaw-managed
#    default config is never edited.
# 2. Turns off what carries state between runs: memory, the user profile,
#    skill creation nudges and the curator. Sets agent.max_turns.
# 3. Installs and enables the bench-spans plugin (hermes_plugins/), which
#    timestamps Hermes's tool calls for the harness.
# 4. Uploads the GAIA attachments to hermes.docs_dir.
# 5. Snapshots the profile. The adapter restores this snapshot before every
#    run, so no run can see an earlier run's sessions, memories or skills.
# 6. Writes manifests/hermes_sandbox.json: versions and image digest, no secrets.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ATK="${ATK:-$(cd "$HERE/.." && pwd)}"
CONFIG="$(realpath "${1:-$HERE/configs/gaia_hermes_qwen_tp2.yaml}")"
export NEMOCLAW_DISABLE_TELEMETRY=1

eval "$("$ATK/.venv/bin/python" - "$CONFIG" <<'PY'
import json, shlex, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
h, sp = cfg["hermes"], cfg["sampling"]
for k in ("sandbox", "profile", "max_turns", "docs_dir"):
    print(f"{k.upper()}={shlex.quote(str(h[k]))}")
print(f"PIN_SAMPLING={'1' if h.get('pin_sampling', True) else ''}")
print(f"WEB_JSON={shlex.quote(json.dumps({'backend': h.get('web_backend', ''), 'disable': h.get('web_disable', [])}))}")
print(f"SAMPLING_JSON={shlex.quote(json.dumps({'temperature': sp['temperature'], 'top_p': sp['top_p'], 'seed': sp['seed']}))}")
PY
)"
SNAPSHOT="/sandbox/.bench/${PROFILE}.tar"

sx() {  # run a shell script in the sandbox, without NemoClaw's node warnings
  nemohermes "$SANDBOX" exec --no-stdin -- sh -c "$1" 2> >(grep -v -i 'experimentalwarning\|trace-warnings' >&2)
}

echo "sandbox=$SANDBOX profile=$PROFILE max_turns=$MAX_TURNS docs=$DOCS_DIR pin_sampling=${PIN_SAMPLING:-no} $SAMPLING_JSON"

sx "test -d /sandbox/.hermes/profiles/$PROFILE || hermes profile create $PROFILE --clone --no-alias >/dev/null"
echo "profile ready"

sx "set -e
hermes -p $PROFILE config set memory.memory_enabled false
hermes -p $PROFILE config set memory.user_profile_enabled false
hermes -p $PROFILE config set curator.enabled false
hermes -p $PROFILE config set skills.creation_nudge_interval 0
hermes -p $PROFILE config set agent.max_turns $MAX_TURNS" >/dev/null
# Settings `config set` cannot reach:
# - The session-title call. Hermes starts it in the background, alongside the
#   first agent call, so it would break concurrency 1 at the proxy (which ends
#   the run as concurrency_violation). A one-shot run never shows a title.
# - Sampling. Hermes sends no temperature, top_p or seed, so vLLM would apply the
#   model's generation_config defaults. With pin_sampling the provider's
#   extra_body carries the cell's values, the same ones the frameworks send.
# - The run-id header. A placeholder in the snapshot; each run fills it in.
# - Web vendors. hermes.web_backend pins search and extract to one keyless
#   vendor; each hermes.web_disable vendor is set to tier "paid", which takes
#   its free endpoint out of the failover ring.
nemohermes "$SANDBOX" exec --no-stdin -- /opt/hermes/.venv/bin/python -c '
import json, sys, yaml
path, pin, sampling, web = sys.argv[1], sys.argv[2] == "1", json.loads(sys.argv[3]), json.loads(sys.argv[4])
c = yaml.safe_load(open(path))
c.setdefault("auxiliary", {}).setdefault("title_generation", {})["enabled"] = False
for cp in c.get("custom_providers") or []:
    if pin:
        cp["extra_body"] = sampling
    else:
        cp.pop("extra_body", None)
    # Each run writes its run id here (agent/hermes/oneshot_agent.py), so the
    # proxy can tell concurrent runs apart; the sandbox policy allows only
    # the plain /v1 paths, so the run cannot travel in the URL.
    cp["extra_headers"] = {**(cp.get("extra_headers") or {}), "X-Run-Id": "__ATK_RUN_ID__"}
w = c.setdefault("web", {})
for key in ("search_backend", "extract_backend"):
    if web["backend"]:
        w[key] = web["backend"]
    else:
        w.pop(key, None)
w["provider_tier"] = {v: "paid" for v in web["disable"]}
# Plugins are opt-in; the bench-spans plugin is installed below.
p = c.setdefault("plugins", {})
p["enabled"] = sorted(set(p.get("enabled") or []) | {"bench-spans"})
yaml.safe_dump(c, open(path, "w"), sort_keys=False)
' "/sandbox/.hermes/profiles/$PROFILE/config.yaml" "${PIN_SAMPLING:-0}" "$SAMPLING_JSON" "$WEB_JSON" \
  2> >(grep -v -i 'experimentalwarning\|trace-warnings' >&2)
sx "for k in memory.memory_enabled memory.user_profile_enabled curator.enabled skills.creation_nudge_interval agent.max_turns auxiliary.title_generation.enabled model.base_url web.search_backend web.extract_backend web.provider_tier plugins.enabled; do printf '  %s = %s\n' \$k \"\$(hermes -p $PROFILE config get \$k | tail -1)\"; done"

# exec --stdin caps a pipe at 4 MiB; upload has no such limit. The gateway is
# the sandbox's recorded one (sandboxes.json gatewayName).
GATEWAY="$(jq -r --arg s "$SANDBOX" '.sandboxes[$s].gatewayName // "nemoclaw"' ~/.nemoclaw/sandboxes.json)"

# The bench-spans plugin: inert unless a run sets HERMES_BENCH_SPANS.
sx "rm -rf /sandbox/.bench/upload && mkdir -p /sandbox/.bench/upload"
openshell sandbox upload -g "$GATEWAY" --no-git-ignore "$SANDBOX" "$HERE/hermes_plugins/bench-spans" /sandbox/.bench/upload >/dev/null
sx "set -e
mkdir -p /sandbox/.hermes/profiles/$PROFILE/plugins
rm -rf /sandbox/.hermes/profiles/$PROFILE/plugins/bench-spans
mv /sandbox/.bench/upload/bench-spans /sandbox/.hermes/profiles/$PROFILE/plugins/
rm -rf /sandbox/.bench/upload"
echo "plugin: bench-spans -> /sandbox/.hermes/profiles/$PROFILE/plugins/"
# Upload nests the source folder, so it goes to a staging dir and is flattened.
sx "rm -rf $DOCS_DIR /sandbox/.bench/upload && mkdir -p $DOCS_DIR /sandbox/.bench/upload"
openshell sandbox upload -g "$GATEWAY" --no-git-ignore "$SANDBOX" "$ATK/GAIA_stack/documents" /sandbox/.bench/upload >/dev/null
sx "mv /sandbox/.bench/upload/documents/* $DOCS_DIR/ && rm -rf /sandbox/.bench/upload"
echo "attachments: $(sx "find $DOCS_DIR -type f | wc -l") files in $DOCS_DIR (host: $(find "$ATK/GAIA_stack/documents" -type f | wc -l))"

# Snapshot with no sessions, memories or logs, whatever the profile collected
# while being configured.
sx "set -e
cd /sandbox/.hermes/profiles/$PROFILE
rm -rf sessions/* memories/* logs/* state.db state.db-* response_store.db 2>/dev/null || true
mkdir -p /sandbox/.bench
tar -C /sandbox/.hermes/profiles -cf $SNAPSHOT $PROFILE
sha256sum config.yaml | cut -c1-16 > /sandbox/.bench/${PROFILE}.config_sha"
echo "snapshot: $SNAPSHOT"

HERMES_VERSION="$(sx 'hermes --version | head -1')"
PROFILE_SHA="$(sx "cat /sandbox/.bench/${PROFILE}.config_sha")"
"$ATK/.venv/bin/python" - "$SANDBOX" "$HERMES_VERSION" "$PROFILE" "$PROFILE_SHA" "$HERE/manifests/hermes_sandbox.json" <<'PY'
import json, pathlib, subprocess, sys, time
sandbox, hermes_version, profile, profile_sha, out = sys.argv[1:]
reg = json.loads((pathlib.Path.home() / ".nemoclaw/sandboxes.json").read_text())["sandboxes"][sandbox]
keep = ("agent", "agentVersion", "nemoclawVersion", "openshellVersion", "imageTag", "provider",
        "model", "endpointUrl", "preferredInferenceApi", "toolDisclosure", "webSearchEnabled",
        "appliedPolicySelection", "hermesApiPort")
manifest = {k: reg.get(k) for k in keep}
manifest.update({
    "sandbox": sandbox,
    "hermes_version": hermes_version,
    "bench_profile": profile,
    "bench_profile_config_sha16": profile_sha,
    # Enabled network policy presets: they decide which sites Hermes's web tools reach.
    "policy_presets": [line.split("●", 1)[1].split("—")[0].strip()
                       for line in subprocess.run(["nemohermes", sandbox, "policy", "list"],
                                                  capture_output=True, text=True).stdout.splitlines()
                       if "●" in line],
    "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
})
pathlib.Path(out).write_text(json.dumps(manifest, indent=2) + "\n")
print(f"manifest: {out}")
PY
