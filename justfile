default:
  just --list

install:
    #!/usr/bin/env bash
    set -euxo pipefail

    uv sync --group transformer_engine_build

    # `transformer-engine-torch` requires compiling
    if [ ! -d "/usr/local/cuda-12.6" ]; then
        echo "Error: CUDA 12.6 not installed. Please install https://developer.nvidia.com/cuda-12-6-0-download-archive" >&2
        exit 1
    fi
    export CUDA_HOME="/usr/local/cuda-12.6"
    export PATH="$CUDA_HOME/bin:$PATH"
    # Must use `clang`: https://github.com/astral-sh/uv/issues/11707
    if ! command -v clang &> /dev/null; then
        echo "Error: clang not installed." >&2
        exit 1
    fi
    export CXX=clang
    source .venv/bin/activate
    pip install transformer-engine-torch --no-deps --no-build-isolation -v

    uv sync --all-extras