# Setup Guide

## System Requirements

* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* Linux operating system (Ubuntu 20.04, 22.04, or 24.04 LTS)
* CUDA version 12.4 or later
* Python version 3.10 or later

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

Install system dependencies:

```sh
# Install uv/just
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv tool install rust-just

# Try to install decord when on ARM platform
bash scripts/install_decord_arm.sh
```

Install the package using your preferred environment:

1. conda

   Install system dependencies:

   * conda/mamba

   ```sh
   just install-conda <conda|mamba|micromamba>
   conda activate cosmos-predict2
   ```

2. venv

   Install system dependencies:

   * [CUDA 12.6](https://developer.nvidia.com/cuda-12-6-0-download-archive)
   * clang

   ```sh
   just install cu126
   source .venv/bin/activate
   ```

[Optional] Install training dependencies

```sh
just install-training
```

### Option 2: Docker container

Please make sure you have access to Docker on your machine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

* **Option 2A: Use pre-built Cosmos-Predict2 container**

   ```bash
   # Pull the Cosmos-Predict2 container
   docker pull nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2
   ```

* **Option 2B: Build container from Dockerfile**

   Make sure you are under the repo root.
   ```bash
   # Build the Docker image
   docker build -t cosmos-predict2-local -f Dockerfile .
   ```

* **Running the container**

   Use the following command to run either container, replacing `[CONTAINER_NAME]` with either `nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.2` or `cosmos-predict2-local`:

   ```bash
   # Run the container with GPU support and mount necessary directories
   docker run --gpus all -it --rm \
   -v /path/to/cosmos-predict2:/workspace \
   -v /path/to/datasets:/workspace/datasets \
   -v /path/to/checkpoints:/workspace/checkpoints \
   [CONTAINER_NAME]

   # Verify setup inside container
   python /workspace/scripts/test_environment.py
   ```

   > **Note**: Replace `/path/to/cosmos-predict2`, `/path/to/datasets`, and `/path/to/checkpoints` with your actual local paths.

## Downloading Checkpoints

1. Get a [Hugging Face](https://huggingface.co/settings/tokens) access token with `Read` permission
2. Login: `huggingface-cli login`
3. The [Llama-Guard-3-8B terms](https://huggingface.co/meta-llama/Llama-Guard-3-8B) must be accepted. Approval will be required before Llama Guard 3 can be downloaded.
4. Download models:

| Models | Link | Download Command | Notes |
|--------|------|------------------|-------|
| Cosmos-Predict2-2B-Text2Image | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image) | `python -m scripts.download_checkpoints --model_types text2image --model_sizes 2B` | N/A |
| Cosmos-Predict2-14B-Text2Image | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Text2Image) | `python -m scripts.download_checkpoints --model_types text2image --model_sizes 14B` | N/A |
| Cosmos-Predict2-2B-Video2World | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) | `python -m scripts.download_checkpoints --model_types video2world --model_sizes 2B` | Download 720P, 16FPS by default. Supports 480P and 720P resolution. Supports 10FPS and 16FPS |
| Cosmos-Predict2-14B-Video2World | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Video2World) | `python -m scripts.download_checkpoints --model_types video2world --model_sizes 14B` | Download 720P, 16FPS by default. Supports 480P and 720P resolution. Supports 10FPS and 16FPS |
| Cosmos-Predict2-2B-Sample-Action-Conditioned | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned) | `python -m scripts.download_checkpoints --model_types sample_action_conditioned` | Supports 480P and 4FPS. |
| Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1 | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1) | `python -m scripts.download_checkpoints --model_types sample_gr00t_dreams_gr1` | Supports 480P and 16FPS. |
| Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID | [🤗 Huggingface](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID) | `python -m scripts.download_checkpoints --model_types sample_gr00t_dreams_droid` | Supports 480P and 16FPS. |


For Video2World model with different resolution and FPS, you can pass `resolution` and `fps` flag to control which model checkpoint to download. For example, if you want a 2B model with 480P and 10FPS, you can do
```bash
python -m scripts.download_checkpoints --model_types video2world --model_sizes 2B --resolution 480 --fps 10
```

Tips: `model_types`, `model_sizes`, `fps` and `resolution` supports multiple values. So if you want a mega command to download {2,14}B Video2World models with {10,16} FPS and {480,720}P, you can download 2x2x2=8 models via
```bash
python -m scripts.download_checkpoints --model_types video2world --model_sizes 2B 14B --resolution 480 720 --fps 10 16
```

You can pass `--checkpoint_dir <path to ckpt>` if you want to control where to put the checkpoints.
You can also add `--verify_md5` flag to verify MD5 checksums of downloaded files. If checksums don't match, models will be automatically redownloaded.

To download models with [sparse attention](performance.md#sparse-attention-powered-by-natten), run the
script with the `--natten` option:

```bash
python -m scripts.download_checkpoints --model_types video2world --model_sizes 2B 14B --resolution 720 --fps 10 16 --natten
```

## Troubleshooting

### CUDA/GPU Issues
- **CUDA driver version insufficient**: Update NVIDIA drivers to latest version compatible with CUDA 12.6+
- **Out of Memory (OOM) errors**: Use 2B models instead of 14B, or reduce batch size/resolution
- **Missing CUDA libraries**: Set paths with `export CUDA_HOME=$CONDA_PREFIX`

### Installation Issues
- **Conda environment conflicts**: Create fresh environment with `conda create -n cosmos-predict2-clean python=3.10 -y`
- **Flash-attention build failures**: Install build tools with `apt-get install build-essential`
- **Transformer engine linking errors**: Reinstall with `pip install --force-reinstall transformer-engine==1.12.0`

For other issues, check [GitHub Issues](https://github.com/nvidia-cosmos/cosmos-predict2/issues).
