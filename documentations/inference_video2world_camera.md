# Video2World Camera-Conditioned Inference Guide

This guide provides instructions for running camera-conditioned video generation with Cosmos-Predict2 Video2World models.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Camera Conditioning Modes](#camera-conditioning-modes)
- [Data Preparation](#data-preparation)
- [Running Inference](#running-inference)
- [Advanced Configuration](#advanced-configuration)

## Overview

Camera-conditioned video generation extends the Video2World models to enable precise control over camera movements in generated videos. This feature allows you to:
- Generate videos with specific camera trajectories
- Control camera movement while maintaining scene consistency
- Apply multiple camera trajectories to the same input video

The system uses Plücker ray embeddings computed from camera extrinsics and intrinsics to condition the video generation process. The model uses the QwenVL text encoder for processing text prompts along with camera conditioning.

**Note**: This feature only accepts video input (not images) and requires mode-specific models.

## Prerequisites

### 1. Environment Setup
Follow the [Setup guide](setup.md) for installation instructions.

2. **Model checkpoints**: Download required model weights following the [Downloading Checkpoints](setup.md#downloading-checkpoints) section in the Setup guide.

**Important**: 
- Each mode requires its own specific model checkpoint trained for that camera conditioning mode
- The QwenVL text encoder will be downloaded automatically when needed (cached in `--cache_dir` if specified)
- Model checkpoints will be available through Hugging Face or NVIDIA NGC once released

### 3. Hardware Requirements
- Minimum: 1x NVIDIA GPU with 24GB VRAM (e.g., RTX 3090, A10)
- Recommended: 1x NVIDIA GPU with 40GB+ VRAM (e.g., A100, H100)
- Multi-GPU support available for context parallelism

### 4. Resolution Support
The system supports **720p** resolution videos/models: 704×1280 pixels (16:9 aspect ratio)

## Camera Conditioning Modes

The system supports two primary modes for camera-conditioned generation, each requiring its own model:

### 1. Camera Trajectory Mode
Allows users to provide custom camera trajectories for controlled camera movements. While we provide a few example trajectories, users are expected to create their own based on their specific needs:
- Example trajectories might include rotate, zoom, or arc movements
- Users define trajectories via extrinsics matrices
- Supports multiple trajectories from the same input video

### 2. AGI Bot Mode
Specifically designed for robotic manipulation with dual hand tracking:
- `camera_tgt_0`: Left hand camera trajectory
- `camera_tgt_1`: Right hand camera trajectory
- Includes corresponding intrinsics for each camera view
- Optimized for robotic vision applications

## Data Preparation

### Preparing Camera Trajectory Files

Camera trajectories are defined using extrinsics matrices (3x4) for each frame. Create text files in your camera directory:

#### Directory Structure
```
camera_trajectories/
├── rot_left.txt
├── rot_right.txt
├── zoom_in.txt
├── zoom_out.txt
├── arc_left.txt
├── arc_right.txt
└── intrinsics_focal525.txt  # Camera intrinsics
```

#### Extrinsics Format (trajectory_name.txt)
Each line represents a 3x4 extrinsics matrix for one frame (rotation + translation):
```
r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
```

Example for 93 frames (one line per frame):
```
1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
1.0 0.0 0.0 0.1 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
...
```

#### Intrinsics Format (intrinsics_focalXXX.txt)
Single line with 4 values: fx, fy, cx, cy

**Note**: Currently supported focal lengths are 24 and 50.

Example for focal length 24:
```
24.0 24.0 320.0 240.0
```

Example for focal length 50:
```
50.0 50.0 320.0 240.0
```

### Preparing AGI Bot Data

For AGI Bot mode, the naming convention is based on the input video filename (without extension or path) plus specific suffixes.

**Naming Pattern:**
```
<video_prefix>_camera_tgt_0.txt  # Left hand camera extrinsics
<video_prefix>_camera_tgt_1.txt  # Right hand camera extrinsics  
<video_prefix>_intrinsics_0.txt  # Left hand camera intrinsics
<video_prefix>_intrinsics_1.txt  # Right hand camera intrinsics
```

**Example:**
If your input video is `/path/to/robot_demo.mp4`, the video prefix is `robot_demo` and you need:
```
agi_bot_cameras/
├── robot_demo_camera_tgt_0.txt  # Left hand camera extrinsics
├── robot_demo_camera_tgt_1.txt  # Right hand camera extrinsics  
├── robot_demo_intrinsics_0.txt  # Left hand camera intrinsics
└── robot_demo_intrinsics_1.txt  # Right hand camera intrinsics
```

**Important**: The prefix must exactly match the input video filename (without extension).

## Running Inference

### Basic Camera Trajectory Example

Generate a video with a custom camera movement:
```bash
python examples/video2world_camera.py \
  --mode camera_trajectory \
  --model_path checkpoints/nvidia/Cosmos-Predict2-2B-Sample-Camera-Conditioned-Basic/ \
  --input_path path/to/input_video.mp4 \
  --camera_path camera_trajectories/ \
  --trajectories trajectory1 trajectory2 \
  --focal 50 \
  --prompt "A bustling city street" \
  --num_latent_conditional_frames 2 \
  --save_path output/multi_camera_video.mp4 \
  --seed 42
```

### AGI Bot Mode Example

For robotic manipulation with dual hand tracking:
```bash
python examples/video2world_camera.py \
  --mode agi_bot \
  --model_path checkpoints/nvidia/Cosmos-Predict2-2B-Sample-Camera-Conditioned-AGIBot/ \
  --input_path videos/robot_scene.mp4 \
  --camera_path agi_bot_cameras/ \
  --prompt "Robot hands manipulating objects" \
  --save_path output/agi_bot_video.mp4 \
  --seed 42
```

**Note**: The camera files must use `robot_scene` as the prefix (matching the input video filename).

### Using EMA Weights

Load a model with EMA weights for potentially better quality:
```bash
python examples/video2world_camera.py \
  --mode camera_trajectory \
  --model_path checkpoints/nvidia/Cosmos-Predict2-2B-Sample-Camera-Conditioned-Basic/ \
  --load_ema \
  --input_path input_video.mp4 \
  --camera_path camera_trajectories/ \
  --trajectories trajectory1 trajectory2 \
  --focal 24 \
  --prompt "Nature scene" \
  --save_path output/ema_video.mp4
```

### Multi-GPU Inference

For faster inference using context parallelism with torchrun:
```bash
torchrun --nproc_per_node=4 --master_port=12345 \
  examples/video2world_camera.py \
  --mode camera_trajectory \
  --model_path checkpoints/nvidia/Cosmos-Predict2-2B-Sample-Camera-Conditioned-Basic/ \
  --num_gpus 4 \
  --input_path input_video.mp4 \
  --camera_path camera_trajectories/ \
  --trajectories trajectory1 trajectory2 \
  --focal 50 \
  --prompt "Aerial view of landscape" \
  --save_path output/multi_gpu_video.mp4
```

**Note**: Multi-GPU inference requires using `torchrun` for proper distributed execution.

## Advanced Configuration

### Command-Line Arguments

#### Required Arguments
- `--mode`: Camera conditioning mode (`camera_trajectory` or `agi_bot`)
- `--model_path`: Path to directory containing model and tokenizer checkpoint files
- `--input_path`: Path to input video (must be .mp4 format)
- `--camera_path`: Directory containing camera trajectory files
- `--prompt`: Text prompt to guide generation
- `--save_path`: Output path for generated video

#### Mode-Specific Arguments
**For camera_trajectory mode:**
- `--trajectories`: Exactly two trajectory names (files in camera_path without .txt extension)
- `--focal`: Focal length for camera intrinsics (24 or 50)

**For agi_bot mode:**
- Video prefix is automatically extracted from input filename (used for camera file naming)

#### Generation Parameters
- `--prompt`: Text prompt for generation guidance
- `--negative_prompt`: Negative prompt for classifier-free guidance
- `--num_latent_conditional_frames`: Number of conditional frames (1 or 2)
- `--num_video_frames`: Number of frames to generate (default: 93)
- `--guidance`: Guidance scale (default: 7, range: 0-20)
- `--seed`: Random seed for reproducibility

#### Model Parameters
- `--load_ema`: Use EMA weights if available
- `--cache_dir`: Cache directory for QwenVL text encoder (optional)

#### System Parameters
- `--num_gpus`: Number of GPUs for context parallelism
- `--save_path`: Output path for generated video

### Creating Custom Camera Trajectories

To create custom camera movements:

1. **Define the camera path**: Create a sequence of extrinsics matrices representing camera position and orientation at each frame.

2. **Smooth interpolation**: Use smooth interpolation between keyframes for natural camera movement:
```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

# Define keyframes
keyframes = {
    0: {"position": [0, 0, 0], "rotation": [0, 0, 0]},    # Start
    46: {"position": [2, 0, 0], "rotation": [0, 30, 0]},  # Middle
    92: {"position": [4, 0, 0], "rotation": [0, 60, 0]}   # End
}

# Interpolate between keyframes
frames = []
for frame_idx in range(93):
    # Interpolate position and rotation
    # Convert to 3x4 extrinsics matrix
    # Save to file
```

3. **Test and refine**: Generate videos with your custom trajectory and adjust as needed.

### Tips for Best Results

1. **Input Quality**: Use high-quality input videos with clear subjects
2. **Prompt Engineering**: Provide detailed, descriptive prompts that match the scene
3. **Camera Movement**: Keep camera movements smooth and realistic
4. **Focal Length**: Use supported focal lengths:
   - 24: Wide angle view
   - 50: Standard view
5. **Guidance Scale**: Adjust guidance for balance between prompt adherence and quality:
   - Lower (3-5): More creative, less prompt adherence
   - Medium (7-10): Balanced
   - Higher (12-20): Strong prompt adherence, may reduce quality

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--num_video_frames` or use a smaller resolution
2. **Poor Quality**: Adjust `--guidance` scale or improve prompt description
3. **Unnatural Movement**: Check camera trajectory files for smooth interpolation

### Performance Optimization

- Use `--num_gpus` for multi-GPU speedup
- Enable `--load_ema` for potentially better quality
- Batch process multiple trajectories in one run for efficiency

## Related Documentation

- [Video2World Inference Guide](inference_video2world.md) - Basic video2world inference
- [Post-training Guide](post-training_video2world.md) - Training custom models
- [Performance Guide](performance.md) - Hardware requirements and optimization
