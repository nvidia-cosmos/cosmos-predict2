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
_conda-env:
    rm -rf .venv
    conda env create -y --no-default-packages -f cosmos-predict2.yaml
    ln -sf "$(conda info --base)/envs/cosmos-predict2" .venv

# Install inference in a new conda environment
install-conda:
    just -f {{ justfile() }} _conda-env
    just -f {{ justfile() }} install cu126
