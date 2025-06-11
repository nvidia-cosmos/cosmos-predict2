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

### Option 1: Conda environment

Please make sure you have a Conda distribution installed ([instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)).

```bash
# Create and activate the environment
conda env create --file cosmos-predict2.yaml
conda activate cosmos-predict2

# Install dependencies
pip install -r requirements-conda.txt
pip install flash-attn==2.6.3 --no-build-isolation
# Patch Transformer engine linking issues
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
pip install transformer-engine[pytorch]==1.13.0

# Apex library for training (optional if inference only)
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext" git+https://github.com/NVIDIA/apex.git

# Verify setup
CUDA_HOME=$CONDA_PREFIX python scripts/test_environment.py
```

Make sure the `CUDA_HOME` environment variable points to your Conda installation directory by running:
```bash
export CUDA_HOME=$CONDA_PREFIX
```

### Option 2: Docker container

Please make sure you have access to Docker on your machine and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed.

* **Option 2A: Use pre-built Cosmos-Predict2 container**

   ```bash
   # Pull the Cosmos-Predict2 container
   docker pull nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.0
   ```

* **Option 2B: Build container from Dockerfile**

   Make sure you are under the repo root.
   ```bash
   # Build the Docker image
   docker build -t cosmos-predict2-local -f Dockerfile .
   ```

* **Running the container**

   Use the following command to run either container, replacing `[CONTAINER_NAME]` with either `nvcr.io/nvidia/cosmos/cosmos-predict2-container:1.0` or `cosmos-predict2-local`:

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
   ```bash
   # Download Text2Image models (2B and 14B)
   python -m scripts.download_checkpoints --model_sizes 2B 14B --model_types text2image --checkpoint_dir checkpoints

   # Download Video2World models (2B and 14B)
   python -m scripts.download_checkpoints --model_sizes 2B 14B --model_types video2world --checkpoint_dir checkpoints
   ```
   Add `--verify_md5` flag to verify MD5 checksums of downloaded files. If checksums don't match, models will be automatically redownloaded.

## Troubleshooting

### CUDA/GPU Issues
- **CUDA driver version insufficient**: Update NVIDIA drivers to latest version compatible with CUDA 12.4+
- **Out of Memory (OOM) errors**: Use 2B models instead of 14B, or reduce batch size/resolution
- **Missing CUDA libraries**: Set paths with `export CUDA_HOME=$CONDA_PREFIX`

### Installation Issues
- **Conda environment conflicts**: Create fresh environment with `conda create -n cosmos-predict2-clean python=3.10 -y`
- **Flash-attention build failures**: Install build tools with `apt-get install build-essential`
- **Transformer engine linking errors**: Reinstall with `pip install --force-reinstall transformer-engine==1.12.0`

For other issues, check [GitHub Issues](https://github.com/nvidia-cosmos/cosmos-predict2/issues).
