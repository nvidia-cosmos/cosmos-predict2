default:
  just --list

extras := "flash-attn transformer-engine natten"
training_extras := "apex"

# Install inference in existing environment
install cuda='cu126':
    echo {{ cuda }} > .venv/cuda-version
    ./scripts/_sync.sh "{{ extras }}"
    ./.venv/bin/python scripts/test_environment.py

# Install training in existing environment
install-training:
    ./scripts/_sync.sh "{{ extras }} {{ training_extras }}"
    ./.venv/bin/python scripts/test_environment.py --training

# Create a new conda environment
_conda-env conda='conda':
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf .venv
    INFO=$({{ conda }} env create -y -f cosmos-predict2.yaml --json)
    VENV=$(echo $INFO | jq -r '."prefix"')
    ln -sf $VENV .venv

# Install inference in a new conda environment
install-conda conda='conda':
    just -f {{ justfile() }} _conda-env {{ conda }}
    just -f {{ justfile() }} install cu126
