# Setup Guide

## System Requirements

* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* NVIDIA driver compatible with CUDA 12.6
* Linux

## Installation

### Clone the repository

```bash
git clone git@github.com:nvidia-cosmos/cosmos-predict2.git
cd cosmos-predict2
```

### ARM installation

When using an ARM platform, like GB200, special steps are required to install the `decord` package.
You need to make sure that [NVIDIA Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk/download) is downloaded in the root of the repository.
The installation will be handled by the Conda scripts or Dockerfile.

### Option 1: Virtual environment

System requirements:

* Linux x86-64
* glibc>=2.31 (e.g Ubuntu >=22.04)

Install system dependencies:

[uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Install the package into a new environment:

```shell
uv sync --extra cu126
source .venv/bin/activate
```

Or, install the package into the active environment (e.g. conda):

```shell
uv sync --extra cu126 --active --inexact
```

### Option 2: Docker container

Please make sure you have access to Docker on your machine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

For x86-64, build and run the container:

```bash
docker run --gpus all --rm -v .:/workspace -v /workspace/.venv -it $(docker build -f uv.Dockerfile -q .)
```

For arm, pull and run a pre-built container:

```bash
docker run --gpus all --rm -v .:/workspace -it nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2
```

## Downloading Checkpoints

1. Get a [Hugging Face Access Token](https://huggingface.co/settings/tokens) with `Read` permission
2. Install [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli): `uv tool install -U "huggingface_hub[cli]"`
3. Login: `hf auth login`
4. Accept the [Llama-Guard-3-8B terms](https://huggingface.co/meta-llama/Llama-Guard-3-8B).

To download a specific model:

```shell
./scripts/download_checkpoints.py --model_types <model_type> --model_sizes <model_size>
```

| Models | Link | Download Arguments | Notes |
|--------|------|--------------------|-------|
| Cosmos-Predict2-0.6B-Text2Image | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-0.6B-Text2Image) | `--model_types text2image --model_sizes 0.6B` | N/A |
| Cosmos-Predict2-2B-Text2Image | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image) | `--model_types text2image --model_sizes 2B` | N/A |
| Cosmos-Predict2-14B-Text2Image | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Text2Image) | `--model_types text2image --model_sizes 14B` | N/A |
| Cosmos-Predict2-2B-Video2World | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) | `--model_types video2world --model_sizes 2B` | Download 720P, 16FPS by default. Supports 480P and 720P resolution. Supports 10FPS and 16FPS |
| Cosmos-Predict2-14B-Video2World | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Video2World) | `--model_types video2world --model_sizes 14B` | Download 720P, 16FPS by default. Supports 480P and 720P resolution. Supports 10FPS and 16FPS |
| Cosmos-Predict2-2B-Sample-Action-Conditioned | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned) | `--model_types sample_action_conditioned` | Supports 480P and 4FPS. |
| Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1 | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1) | `--model_types sample_gr00t_dreams_gr1` | Supports 480P and 16FPS. |
| Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID) | `--model_types sample_gr00t_dreams_droid` | Supports 480P and 16FPS. |

To download all the checkpoints (requires ~250GB of disk space), run:

```shell
./scripts/download_checkpoints.py
```

To see the full list of options, run:

```shell
./scripts/download_checkpoints.py --help
```

## Troubleshooting

### CUDA/GPU Issues

* **CUDA driver version insufficient**: Update NVIDIA drivers to latest version compatible with CUDA 12.6+

* **Out of Memory (OOM) errors**: Use 2B models instead of 14B, or reduce batch size/resolution
* **Missing CUDA libraries**: Set paths with `export CUDA_HOME=$CONDA_PREFIX`

### Installation Issues

* **Conda environment conflicts**: Create fresh environment with `conda create -n cosmos-predict2-clean python=3.10 -y`

* **Flash-attention build failures**: Install build tools with `apt-get install build-essential`
* **Transformer engine linking errors**: Reinstall with `pip install --force-reinstall transformer-engine==1.12.0`

For other issues, check [GitHub Issues](https://github.com/nvidia-cosmos/cosmos-predict2/issues).
