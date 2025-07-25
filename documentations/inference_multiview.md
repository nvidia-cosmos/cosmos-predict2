# Multiview Inference Guide

This guide explains how to run inference with **Cosmos-Predict2 Multiview** models.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Overview](#overview)
- [Examples](#examples)
  - [7 views generation (#7-views generation)
- [API Documentation](#api-documentation)
- [Prompt Engineering Tips](#prompt-engineering-tips)
- [Related Documentation](#related-documentation)

---
## Prerequisites

Before running inference:

1. **Environment setup**: Follow the [Setup guide](setup.md) for installation instructions.
2. **Model checkpoints**: Download required model weights following the [Downloading Checkpoints](setup.md#downloading-checkpoints) section in the Setup guide.
3. **Hardware considerations**: Review the [Performance guide](performance.md) for GPU requirements and model selection recommendations.

---
## Overview

Multiview models extend Video2World to handle multiple camera perspectives jointly. The MultiView pipeline supports 3 modes through the number of conditional frames variable. 

1. Text2World-Multiview : num_conditional_frames=0
2. Image2World-Multiview: num_conditional_frames=1
3. Video2World-Multiview: num_conditional_frames=5


The inference script is located at `examples/multiview.py`.

For a complete list of available arguments and options:
```bash
python -m examples.multiview --help
```

## Examples

### 7-views generation

The multiview generation script expects the video path and a path containing the input prompt (as `txt` file) or prompt embeddinggs tensor as `pkl` file. 

```bash
export NUM_GPUS=8

torchrun --nproc_per_node=${NUM_GPUS} examples/multiview.py  \
  --model_size 2B  \
  --num_gpus $NUM_GPUS  \
  --input_path  assets/multiview/sample1/video.mp4  \
  --prompt assets/multiview/sample1/caption.txt \
  --num_conditional_frames 1 \
  --fps 10   \
  --n_views 7 \
  --guidance 7.0 \
  --disable_prompt_refiner \
  --save_path output/multiview_2b_sample1_cond1.mp4

```

This distributes the computation across multiple GPUs, with each GPU processing a subset of the video frames.


> **Note:** Both parameters are required: `--nproc_per_node` tells PyTorch how many processes to launch, while `--num_gpus` tells the model how to distribute the workload.

**Important considerations for multi-GPU inference:**
- The number of GPUs should ideally be a divisor of the number of frames in the generated video
- All GPUs should have the same model capacity and memory
- Context parallelism works best with the 14B model where memory constraints are significant
- Requires NCCL support and proper GPU interconnect for efficient communication
- Significant speedup for video generation while maintaining the same quality'=

## API Documentation

In addition to the same parameters supported by `tex2world.py`, `multiview.py` also supports the following command-line arguments:

- `--n_views`: sets the number of camera views of the input condition and generated video

## Related Documentation

- [Multiview Inference Guide](inference_multiview.md) - Generate still images from text prompts
- [Setup Guide](setup.md) - Environment setup and checkpoint download instructions
- [Performance Guide](performance.md) - Hardware requirements and optimization recommendations
- [Training Multiview on Waymo Guide](multiview_post-training_waymo.md) - Information on training a multiview model on Waymo dataset.
