"""Turn a run config into vLLM serving settings, for serve_vllm.sh --config.

The config is the single source for how a model is served. Without this, the
serving settings lived in shell env vars that had to be kept in step with the
YAML by hand, and a mismatch (thinking on at the server, off in the config)
silently changed every token count.

Prints shell assignments for the serve script to eval. Values are shell-quoted.

    python scripts/serve_args.py configs/pilot_qwen.yaml
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import RunConfig  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: serve_args.py <config.yaml>", file=sys.stderr)
        return 2
    try:
        cfg = RunConfig.from_yaml(sys.argv[1])
    except (ValueError, TypeError) as exc:
        # Validation lives in ModelConfig, so a bad thinking/effort combination
        # is refused here, before any GPU memory is touched.
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    m = cfg.model
    values = {
        "LOCAL_DIR": m.local_dir,
        "SERVED_NAME": m.served_name,
        "TP": str(m.tp_size),
        "MAX_LEN": str(m.max_model_len),
        "GPU_UTIL": str(m.gpu_memory_utilization),
        "TOOL_PARSER": m.tool_call_parser,
        "REASONING_PARSER": m.reasoning_parser,
        "THINKING": m.thinking_setting,
        "REASONING_EFFORT": m.reasoning_effort,
        "TEMPLATE_KWARGS": json.dumps(m.chat_template_kwargs(cfg.as_of_date)),
        "SEED": str(cfg.sampling.seed),
    }
    for key, value in values.items():
        print(f"{key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
