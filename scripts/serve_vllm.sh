#!/usr/bin/env bash
# Serve a local model with vLLM for one benchmark cell.
#
# The run config is the single source for how the model is served: weights,
# parsers, context length, tensor parallelism and the thinking mode all come
# from it, so the server and the recorded cell cannot disagree.
#
#   ./scripts/serve_vllm.sh --config configs/pilot_qwen.yaml
#
# Weights come from model/<local_dir>, never the Hub: HF_HUB_OFFLINE=1 makes a
# missing local copy fail loudly instead of silently downloading mid-run.
#
# What was actually launched is written to output/serving_<port>.json, and the
# harness preflight compares it field by field against the config it is given.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" != "--config" || -z "${2:-}" ]]; then
  echo "usage: serve_vllm.sh --config <configs/cell.yaml>"
  echo "  thinking mode and effort are set in the config (model.thinking_setting,"
  echo "  model.reasoning_effort), not with env vars, so they are recorded with the results"
  exit 2
fi
CONFIG="$2"

# Every serving setting, derived and validated from the config. A bad
# thinking/effort combination is refused here, before any GPU memory is touched.
SETTINGS=$(.venv/bin/python scripts/serve_args.py "$CONFIG") || exit 1
eval "$SETTINGS"

MODEL_PATH="$(pwd)/model/${LOCAL_DIR}"
[[ -d "$MODEL_PATH" ]] || { echo "missing weights: $MODEL_PATH (run scripts/fetch_models.py)"; exit 1; }

PORT=${PORT:-8000}
# Hybrid Mamba models allocate one Mamba cache block per decode sequence, so
# vLLM's default max_num_seqs (1024) can exceed what fits and abort CUDA graph
# capture. This also caps Phase 4 concurrency, and it changes batching, so it
# is a controlled variable: identical across frameworks, recorded per cell.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}

ARGS=(
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --enable-auto-tool-choice
  --tool-call-parser "$TOOL_PARSER"
  --enable-prefix-caching
  --dtype bfloat16
  --seed "$SEED"
  # Thinking pinned server-side, never inherited from the chat template.
  # Qwen3.8's template defaults to thinking ON at reasoning_effort=xhigh.
  --default-chat-template-kwargs "$TEMPLATE_KWARGS"
)
# Splits reasoning into its own response field, so thinking tokens are counted
# and kept out of the answer. vLLM reports reasoning_tokens only when this is set.
[[ -n "$REASONING_PARSER" ]] && ARGS+=(--reasoning-parser "$REASONING_PARSER")

echo "serving $SERVED_NAME from $MODEL_PATH  (config: $CONFIG)"
echo "  tp=$TP port=$PORT tool-parser=$TOOL_PARSER reasoning-parser=${REASONING_PARSER:-none} max_num_seqs=$MAX_NUM_SEQS"
echo "  thinking=$THINKING effort=${REASONING_EFFORT:-n/a} template-kwargs=$TEMPLATE_KWARGS"

# Record what is being launched, for preflight to check against the config.
mkdir -p output
.venv/bin/python - "$CONFIG" "$PORT" "$MAX_NUM_SEQS" <<'PY'
import json, sys, time
from pathlib import Path
config, port, max_num_seqs = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, ".")
from agent.config import RunConfig
cfg = RunConfig.from_yaml(config)
m = cfg.model
Path(f"output/serving_{port}.json").write_text(json.dumps({
    "config": config,
    "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "served_name": m.served_name,
    "local_dir": m.local_dir,
    "tp_size": m.tp_size,
    "max_model_len": m.max_model_len,
    "tool_call_parser": m.tool_call_parser,
    "reasoning_parser": m.reasoning_parser,
    "thinking_setting": m.thinking_setting,
    "reasoning_effort": m.reasoning_effort,
    "chat_template_kwargs": m.chat_template_kwargs(cfg.as_of_date),
    "max_num_seqs": int(max_num_seqs),
}, indent=2))
PY

# vLLM JIT-compiles kernels at startup (hybrid Mamba layers, FlashInfer fused
# MoE) and needs a CUDA toolkit. There is no system toolkit here; the pip wheels
# ship one, but in a non-standard layout the build cannot link against, so
# setup_cuda_home.sh assembles a standard-layout CUDA home from symlinks.
# The JIT build shells out to `ninja` by name, so the venv's bin must be on PATH.
export PATH="$(pwd)/${VENV:-.venv/bin}:$PATH"
if [[ -z "${CUDA_HOME:-}" ]]; then
  CUDA_HOME=$(./scripts/setup_cuda_home.sh) || exit 1
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
  echo "CUDA_HOME=$CUDA_HOME (standard layout over the pip wheels)"
fi

# FlashInfer JIT-compiles its sampling kernels at startup, and the pip CUDA
# toolchain here emits PTX newer than its own ptxas accepts. The PyTorch
# sampler needs no compilation. This is server-side, so every framework sees
# the same sampler and parity is unaffected; it is recorded in the cell config.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "${VLLM:-.venv/bin/vllm}" serve "${ARGS[@]}"
