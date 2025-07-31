#!/usr/bin/env bash

# Synchronize and compile dependencies.
# Used by `just install`.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <all_extras>"
    exit 1
fi
all_extras=$1

VENV="$(pwd)/.venv"
PATH="$VENV/bin:$PATH"

# Install build dependencies
extras="--extra $(<"$VENV/cuda-version")"
uv sync --extra build $extras

# Set build environment variables
eval $(python -c "
import torch
from packaging.version import Version
print(f'export TORCH_VERSION={Version(torch.__version__).base_version}')
print(f'export CUDA_VERSION={torch.version.cuda}')
print(f'export _GLIBCXX_USE_CXX11_ABI={1 if torch.compiled_with_cxx11_abi() else 0}')
")
export UV_CACHE_DIR="$(uv cache dir)/torch${TORCH_VERSION//./}_cu${CUDA_VERSION//./}_cxx11abi=${_GLIBCXX_USE_CXX11_ABI}"
if [ -f "$VENV/bin/nvcc" ]; then
    echo "Using conda CUDA"
    SITE_PACKAGES="$(python -c "import site; print(site.getsitepackages()[0])")"
    ln -sf "$SITE_PACKAGES"/nvidia/*/include/* "$VENV/include/"
    export CUDA_HOME="$VENV"
else
    echo "Using system CUDA"
    export CUDA_HOME="/usr/local/cuda-$CUDA_VERSION"
    export PATH="$CUDA_HOME/bin:$PATH"
    if [ ! -d "$CUDA_HOME" ]; then
        echo "Error: CUDA $CUDA_VERSION not installed. Please install https://developer.nvidia.com/cuda-toolkit-archive" >&2
        exit 1
    fi
fi
# Must use `clang`: https://github.com/astral-sh/uv/issues/11707
export CXX=clang
if ! command -v clang &> /dev/null; then
    echo "Error: clang not installed." >&2
    exit 1
fi
# transformer-engine: https://github.com/NVIDIA/TransformerEngine?tab=readme-ov-file#pip-installation
export NVTE_FRAMEWORK=pytorch

# Compile dependencies
for extra in $all_extras; do \
    echo "Compiling $extra. This may take a while..."; \
    extras+=" --extra $extra"; \
    uv sync --extra build $extras; \
done

# Remove build dependencies
uv sync $extras
