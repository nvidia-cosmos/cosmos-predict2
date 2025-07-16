default:
  just --list

# Run command.
_run *cmd:
    #!/usr/bin/env bash
    set -euo pipefail

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
    {{ cmd }}

# Install inference environment
install cuda='cu126':
    #!/usr/bin/env bash
    set -euo pipefail

    extras="--extra {{ cuda }}"
    uv sync --extra build $extras
    
    for extra in flash-attn transformer-engine natten; do \
        echo "Building $extra. This may take a while..."; \
        extras+=" --extra $extra"; \
        just -f {{ justfile() }} _run uv sync --extra build $extras; \
    done

    uv run $extras scripts/test_environment.py

# Create a new conda environment
_conda-env:
    rm -rf .venv
    conda env create -y --no-default-packages -f cosmos-predict2.yaml
    ln -sf "$(conda info --base)/envs/cosmos-predict2" .venv

# Install in a new conda environment
install-conda:
    just -f {{ justfile() }} install cu124

# # Install training environment
# install-training:
#     uv sync --only-group build --inexact
    
#     # --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext"
#     just -f {{ justfile() }} _sync -v --extra apex --inexact



#     # just -f {{ justfile() }} _sync -v --extra flash-attn-3 --inexact
#     uv sync --all-extras
#     uv run scripts/test_environment.py --training