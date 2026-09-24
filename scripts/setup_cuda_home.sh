#!/usr/bin/env bash
# Build a standard-layout CUDA home from the pip CUDA wheels, out of symlinks.
#
# vLLM JIT-compiles kernels at startup (Mamba layers, FlashInfer fused MoE),
# and those builds assume a system CUDA install:
#   - libraries in $CUDA_HOME/lib64, not lib/
#   - unversioned names, e.g. libcudart.so; the wheels ship only libcudart.so.13
# Without them, compilation succeeds and the final link fails with
# "cannot find -lcudart". Dense models never hit this; MoE models do.
#
# Nothing inside the pip packages is modified, so a pip reinstall cannot break
# it; this script is idempotent and serve_vllm.sh reruns it on every launch.
#
#   ./scripts/setup_cuda_home.sh      # prints the CUDA_HOME to use
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cu13"
DST="$(pwd)/.venv/cuda_home"
[[ -x "$SRC/bin/nvcc" ]] || { echo "no pip CUDA toolchain at $SRC" >&2; exit 1; }

mkdir -p "$DST/lib64"
for d in bin include nvvm lib; do
  ln -sfn "$SRC/$d" "$DST/$d"
done

# Every library under lib64, plus the unversioned name the linker asks for.
for f in "$SRC"/lib/*; do
  ln -sfn "$f" "$DST/lib64/$(basename "$f")"
done
for so in "$SRC"/lib/*.so.*; do
  name=$(basename "$so")
  base=${name%%.so.*}.so                    # libcudart.so.13 -> libcudart.so
  [[ -e "$DST/lib64/$base" ]] || ln -sfn "$so" "$DST/lib64/$base"
done

echo "$DST"
