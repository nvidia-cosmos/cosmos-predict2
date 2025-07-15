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
    source .venv/bin/activate

    # Compile packages
    uv export -q --only-group build --format requirements.txt -o requirements-build.txt
    uv export -q --only-group compile --format requirements.txt -o requirements-compile.txt
    pip install -r requirements-build.txt
    export _GLIBCXX_USE_CXX11_ABI=$(python -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)")
    # export MAX_JOBS=4 # Avoid out of memory: https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
    echo "Compiling packages. This may take a while..."
    pip install -v -r requirements-compile.txt --no-build-isolation

    # uv sync --all-extras
