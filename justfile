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

# https://spdx.org/licenses/
allow_licenses := "MIT BSD-2-CLAUSE BSD-3-CLAUSE APACHE-2.0 ISC"
ignore_package_licenses := "nvidia-* hf-xet certifi filelock matplotlib typing-extensions"

# Update the license
license: install
  uvx licensecheck --show-only-failing --only-licenses {{allow_licenses}} --ignore-packages {{ignore_package_licenses}} --zero
  uvx pip-licenses --python .venv/bin/python --format=plain-vertical --with-license-file --no-license-path --no-version --with-urls --output-file ATTRIBUTIONS.md
  pre-commit run --files ATTRIBUTIONS.md || true

# Pre-release checks
release-check:
  just license
  pre-commit run --all-files --hook-stage manual

# Release a new version
release pypi_token='dry-run' *args:
  ./bin/release.sh {{pypi_token}} {{args}}

# Build the docker image
docker-build cuda_version='12.6.3' *args:
  docker build --build-arg CUDA_VERSION="{{cuda_version}}" -t cosmos-predict2:{{cuda_version}} -f uv.Dockerfile . {{args}}

# Run the docker container
docker cuda_version='12.6.3' *args:
  # https://github.com/astral-sh/uv-docker-example/blob/main/run.sh
  just docker-build "{{cuda_version}}"
  docker run --gpus all --rm -v .:/workspace -v /workspace/.venv -it cosmos-predict2:{{cuda_version}} {{args}}

# Run the arm docker container
docker-arm *args:
  docker run --gpus all --rm -v .:/workspace -it nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2 {{args}}
