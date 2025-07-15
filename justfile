default:
  just --list

install:
    #!/usr/bin/env bash
    set -euo pipefail

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

    uv venv --allow-existing
    requirements_file=$(mktemp)

    # Install build dependencies
    uv export -q --only-group build > $requirements_file
    uv pip install -r $requirements_file --no-deps

    # Compile packages
    echo "Compiling packages. This may take a while..."
    export _GLIBCXX_USE_CXX11_ABI=$(python -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)")
    uv export -q --only-group compile > $requirements_file
    uv pip install -v -r $requirements_file --no-build-isolation --no-cache --no-deps

    # Install all dependencies
    uv sync

    # Check installation
    uv run scripts/test_environment.py
