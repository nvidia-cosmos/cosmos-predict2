#!/usr/bin/env bash

# Used by `just install`.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <all_extras>"
    exit 1
fi
all_extras=$1

# Install build dependencies
extras="--extra $(<".cuda-version")"
uv sync --extra build $extras

# Set environment variables
if [ -f ".venv/bin/nvcc" ]; then
    echo "Using conda cuda"
    ln -sf $(pwd)/.venv/lib/python3.10/site-packages/nvidia/*/include/* $(pwd)/.venv/include/
    ln -sf $(pwd)/.venv/lib/python3.10/site-packages/nvidia/*/include/* $(pwd)/.venv/include/python3.10/
    export CUDA_HOME="$(pwd)/.venv"
else
    echo "Using system cuda"
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    if [ ! -d "/usr/local/cuda-$CUDA_VERSION" ]; then
        echo "Error: CUDA $CUDA_VERSION not installed. Please use conda or install https://developer.nvidia.com/cuda-toolkit-archive" >&2
        exit 1
    fi
    export CUDA_HOME="/usr/local/cuda-$CUDA_VERSION"
    export PATH="$CUDA_HOME/bin:$PATH"

    # Must use `clang`: https://github.com/astral-sh/uv/issues/11707
    if ! command -v clang &> /dev/null; then
        echo "Error: clang not installed." >&2
        exit 1
    fi
    export CXX=clang
fi
export _GLIBCXX_USE_CXX11_ABI=$(python -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)")

# Compile dependencies
for extra in $all_extras; do \
    echo "Compiling $extra. This may take a while..."; \
    extras+=" --extra $extra"; \
    uv sync --extra build $extras; \
done

# Remove build dependencies
uv sync $extras