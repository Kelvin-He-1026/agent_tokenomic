"""Download model weights into model/ once, so runs never touch the network.

A benchmark that pulls from the Hub mid-run is neither reproducible nor
offline-safe, and a download inside a measured tool call or first token would
land in the latency numbers. This script is the only place that reaches out.

It records the resolved commit hash per model in model/MANIFEST.json, so a
local copy stays traceable to a specific upstream revision.

    python scripts/fetch_models.py --config configs/pilot_qwen.yaml
    python scripts/fetch_models.py --repo BAAI/bge-large-en-v1.5 --dir bge-large-en-v1.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import RunConfig  # noqa: E402

MANIFEST = "MANIFEST.json"


def fetch(repo: str, target: Path, revision: str = "main", token: str | None = None) -> str:
    from huggingface_hub import snapshot_download

    print(f"fetching {repo}@{revision} -> {target}")
    path = snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=str(target),
        token=token,
        # Skip duplicate formats; vLLM and sentence-transformers read safetensors.
        ignore_patterns=["*.pth", "*.msgpack", "*.h5", "original/*"],
    )
    return _resolved_commit(repo, revision, token) or path


def _resolved_commit(repo: str, revision: str, token: str | None) -> str:
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo, revision=revision, token=token).sha or ""
    except Exception:
        return ""


def record(models_dir: Path, repo: str, local_dir: str, revision: str, commit: str) -> None:
    path = models_dir / MANIFEST
    data = json.loads(path.read_text()) if path.exists() else {}
    data[local_dir] = {"repo": repo, "revision": revision, "commit": commit}
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f"  recorded {local_dir}: commit={commit or 'unknown'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="fetch both models named by this config")
    ap.add_argument("--repo", help="fetch a single repo instead")
    ap.add_argument("--dir", help="local directory name for --repo")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--models-dir", default="model")
    ap.add_argument("--token", default=None, help="HF token for gated repos (e.g. Llama)")
    args = ap.parse_args()

    if args.config:
        cfg = RunConfig.from_yaml(args.config)
        models_dir = cfg.resolve(cfg.models_dir)
        wanted = [
            (cfg.model.repo, cfg.model.local_dir, cfg.model.revision),
            (cfg.embedding_model, cfg.embedding_local_dir, "main"),
        ]
    elif args.repo and args.dir:
        models_dir = Path(args.models_dir).resolve()
        wanted = [(args.repo, args.dir, args.revision)]
    else:
        ap.error("pass --config, or both --repo and --dir")
        return 2

    models_dir.mkdir(parents=True, exist_ok=True)
    for repo, local_dir, revision in wanted:
        target = models_dir / local_dir
        if (target / "config.json").exists():
            print(f"present {repo} -> {target}")
            continue
        commit = fetch(repo, target, revision, args.token)
        record(models_dir, repo, local_dir, revision, commit)

    print(f"\nmodels under {models_dir}")
    print("set HF_HUB_OFFLINE=1 for runs, so nothing reaches the network")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
