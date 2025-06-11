# Text2World Inference Guide

This guide provides instructions on running inference with Cosmos-Predict2 Text2World models.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Overview](#overview)
- [Examples](#examples)
  - [Single Video Generation](#single-video-generation)
  - [Batch Video Generation](#batch-video-generation)
  - [Multi-GPU Inference](#multi-gpu-inference)
- [API Documentation](#api-documentation)
- [Prompt Engineering Tips](#prompt-engineering-tips)

## Prerequisites

Before running inference:

1. **Environment setup**: Follow the [Setup guide](setup.md) for installation instructions.
2. **Model checkpoints**: Download required model weights following the [Downloading Checkpoints](setup.md#downloading-checkpoints) section in the Setup guide.
3. **Hardware considerations**: Review the [Performance guide](performance.md) for GPU requirements and model selection recommendations.

## Overview

The Text2World pipeline combines the Text2Image and Video2World models to generate videos directly from text prompts in a two-phase process:

1. **Text2Image generation**: The text prompt is processed by the Text2Image model to create a single still image that serves as the first frame.
2. **Video2World generation**: This image (single-frame conditioning) is then fed into the Video2World model along with the original text prompt to animate it into a dynamic video.

The temporary image created in the first phase is automatically cleaned up after the process completes successfully.

The inference script is located at `examples/text2world.py`.
It requires the input argument `--prompt` (text input).

For a complete list of available arguments and options:
```bash
python -m examples.text2world --help
```

## Examples

### Single Video Generation

This is a basic example for running inference on the 2B model with a text prompt.
The output is saved to `output/text2world_2b.mp4`.

```bash
# Set the input prompt
PROMPT="An autonomous welding robot arm operating inside a modern automotive factory, sparks flying as it welds a car frame with precision under bright overhead lights."
# Run text2world generation
python -m examples.text2world \
    --model_size 2B \
    --prompt "${PROMPT}" \
    --save_path output/text2world_2b.mp4
```

The 14B model can be run similarly by changing the model size parameter.

### Batch Video Generation

For generating multiple videos with different prompts, you can use a JSON file with batch inputs. The JSON file should contain an array of objects, where each object has:
- `prompt`: The text prompt describing the desired video (required)
- `output_video`: The path where the generated video should be saved (required)

An example can be found in `assets/text2world/batch_example.json`:
```json
[
  {
    "prompt": "An autonomous welding robot arm operating inside a modern automotive factory, sparks flying as it welds a car frame with precision under bright overhead lights.",
    "output_video": "output/welding-robot-factory.mp4"
  },
  {
    "prompt": "A wooden sailboat gently moves across a tranquil lake, its sail billowing slightly with the breeze. The water ripples around the hull as the boat glides forward. Mountains are visible in the background under a clear blue sky with scattered clouds.",
    "output_video": "output/sailboat-mountain-lake.mp4"
  },
  {
    "prompt": "A modern kitchen scene where a stand mixer is actively blending cake batter in a glass bowl. The beater rotates steadily, incorporating ingredients as the mixture swirls. A light dusting of flour is visible on the countertop, and sunshine streams in through a nearby window.",
    "output_video": "output/kitchen-mixer-baking.mp4"
  }
]
```

Specify the input via the `--batch_input_json` argument:
```bash
# Run batch text2world generation
python -m examples.text2world \
    --model_size 2B \
    --batch_input_json assets/text2world/batch_example.json
```

This will generate three separate videos according to the prompts specified in the JSON file, with each output saved to its corresponding path.

### Multi-GPU Inference

For faster inference, particularly with the 14B model, Text2World supports context parallelism for the video generation phase. This distributes the video frames across multiple GPUs, significantly reducing the time required for video generation.

To enable multi-GPU inference, set the `NUM_GPUS` environment variable and use `torchrun` to launch the script:

```bash
# Set the number of GPUs to use
export NUM_GPUS=8

# Run text2world generation with context parallelism using torchrun
torchrun --nproc_per_node=${NUM_GPUS} examples/text2world.py \
    --model_size 2B \
    --prompt "${PROMPT}" \
    --save_path output/text2world_2b_${NUM_GPUS}gpu.mp4 \
    --num_gpus ${NUM_GPUS}
```

This distributes the computation across multiple GPUs for the video generation phase (Video2World), with each GPU processing a subset of the video frames. The image generation phase (Text2Image) still runs on a single GPU.

> **Note:** Both parameters are required: `--nproc_per_node` tells PyTorch how many processes to launch, while `--num_gpus` tells the model how to distribute the workload. Using the same environment variable for both ensures they are synchronized.

Important considerations for multi-GPU inference:
- The number of GPUs should ideally be a divisor of the number of frames in the generated video
- All GPUs should have the same model capacity and memory
- For best results, use context parallelism with the 14B model where memory constraints are significant
- Requires NCCL support and proper GPU interconnect for efficient communication

This feature is particularly useful for generating higher quality videos with the 14B model, which would otherwise require significant processing time on a single GPU.

## API Documentation

The `predict2_text2world.py` script supports the following command-line arguments:

Model selection:
- `--model_size`: Size of the model to use (choices: "2B", "14B", default: "2B")

Input parameters:
- `--prompt`: Text prompt describing the video to generate (default: predefined example prompt)
- `--negative_prompt`: Text describing what to avoid in the generated video (default: predefined negative prompt)

Output parameters:
- `--save_path`: Path to save the generated video (default: "output/generated_video.mp4")

Generation parameters:
- `--guidance`: Classifier-free guidance scale (default: 7.0)
- `--seed`: Random seed for reproducibility (default: 0)
- `--use_cuda_graphs`: Use CUDA Graphs for inference acceleration (for Text2Image phase)
- `--num_gpus`: Number of GPUs to use for context parallel inference in the video generation phase (default: 1)

Multi-GPU inference:
- For multi-GPU inference, use `torchrun --nproc_per_node=$NUM_GPUS examples/text2world.py ...`
- Both `--nproc_per_node` (for torchrun) and `--num_gpus` (for the script) must be set to the same value
- Setting the `NUM_GPUS` environment variable and using it for both parameters ensures they stay synchronized

Batch processing:
- `--batch_input_json`: Path to JSON file containing batch inputs, where each entry should have 'prompt' and 'output_video' fields

Content safety and controls:
- `--disable_guardrail`: Disable guardrail checks on prompts (by default, guardrails are enabled to filter harmful content)
- `--disable_prompt_refiner`: Disable prompt refiner that enhances short prompts (by default, the prompt refiner is enabled)

## Prompt Engineering Tips

For best results with Text2World models, prompts should describe both what should appear in the scene and how things should move or change over time:

1. **Scene description**: Include details about objects, lighting, materials, and spatial relationships
2. **Motion description**: Describe how elements should move, interact, or change during the video
3. **Temporal progression**: Use words like "gradually," "suddenly," or "transitions to" to guide how the scene evolves
4. **Physical dynamics**: Describe physical effects like "water splashing," "leaves rustling," or "smoke billowing"
5. **Cinematography terms**: Include camera movements like "panning across," "zooming in," or "tracking shot"

Example of a good prompt:
```
A tranquil lakeside at sunset. Golden light reflects off the calm water surface, gradually rippling as a gentle breeze passes through. Tall pine trees along the shore sway slightly, their shadows lengthening across the water. A small wooden dock extends into the lake, where a rowboat gently bobs with the subtle movements of the water.
```

This prompt includes both static scene elements and suggestions for motion that the Video2World model can interpret and animate.

## Related Documentation

- [Text2Image Inference Guide](inference_text2image.md) - Generate still images from text prompts
- [Video2World Inference Guide](inference_video2world.md) - Generate videos from text and visual inputs
- [Setup Guide](setup.md) - Environment setup and checkpoint download instructions
- [Performance Guide](performance.md) - Hardware requirements and optimization recommendations
