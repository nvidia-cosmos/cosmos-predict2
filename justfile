default:
  just --list

# Setup the repository
setup:
  uv tool install -U pre-commit
  pre-commit install -c .pre-commit-config-base.yaml

# Install the repository
install:
  uv sync --extra cu126

# Run linting and formatting
lint: setup
  pre-commit run --all-files || pre-commit run --all-files

# Run tests
test: lint

# Update the license
license: install
  uvx licensecheck --show-only-failing --ignore-packages "nvidia-*" "hf-xet" --zero
  uvx pip-licenses --python .venv/bin/python --format=plain-vertical --with-license-file --no-license-path --no-version --with-urls --output-file ATTRIBUTIONS.md

# Release a new version
release pypi_token='dry-run' *args:
  ./bin/release.sh {{pypi_token}} {{args}}

# Build the docker image
docker-build cuda_version='12.6.3' *args:
  docker build --build-arg CUDA_VERSION="{{cuda_version}}" -t cosmos-predict2:{{cuda_version}} -f uv.Dockerfile . {{args}}

# Run the docker container
docker cuda_version='12.6.3' *args:
  # https://github.com/astral-sh/uv-docker-example/blob/main/run.sh
  just -f {{justfile()}} docker-build "{{cuda_version}}"
  docker run --gpus all --rm -v .:/workspace -v /workspace/.venv -it cosmos-predict2:{{cuda_version}} {{args}}

# Run the arm docker container
docker-arm *args:
  docker run --gpus all --rm -v .:/workspace -it nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2 {{args}}
