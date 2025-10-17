# Text2Image Inference Guide

This guide provides instructions on running inference with Cosmos-Predict2 Text2Image models.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Overview](#overview)
- [Examples](#examples)
  - [Single Image Generation](#single-image-generation)
  - [Batch Image Generation](#batch-image-generation)
- [API Documentation](#api-documentation)
- [Prompt Engineering Tips](#prompt-engineering-tips)

## Prerequisites

Before running inference:

1. **Environment setup**: Follow the [Setup guide](setup.md) for installation instructions.
2. **Model checkpoints**: Download required model weights following the [Downloading Checkpoints](setup.md#downloading-checkpoints) section in the Setup guide.
3. **Hardware considerations**: Review the [Performance guide](performance.md) for GPU requirements and model selection recommendations.

## Overview

Cosmos-Predict2 provides two models for text-to-image generation: `Cosmos-Predict2-2B-Text2Image` and `Cosmos-Predict2-14B-Text2Image`. These models can transform natural language descriptions into high-quality images through progressive diffusion guided by the text prompt.

The inference script is `examples/text2image.py`.
It requires the input argument `--prompt` (text input).
To see the complete list of available arguments, run:

```bash
!python examples/text2image.py --help
```

## Examples

### Single Image Generation

This is a basic example for running inference on the 2B model with a single prompt.
The output is saved to `output/text2image_2b.jpg`.

```bash
# Set the input prompt
!PROMPT_="A well-worn broom sweeps across a dusty wooden floor, its bristles gathering crumbs and flecks of debris in swift, rhythmic strokes. Dust motes dance in the sunbeams filtering through the window, glowing momentarily before settling. The quiet swish of straw brushing wood is interrupted only by the occasional creak of old floorboards. With each pass, the floor grows cleaner, restoring a sense of quiet order to the humble room."
# Run text2image generation
!python -m examples.text2image \
    --prompt "${PROMPT_}" \
    --model_size 2B \
    --save_path output/text2image_2b.jpg
```

The 0.6B or 14B model can be run similarly by changing the model size parameter.

#### Single Image Generation with fast image tokenizer
```bash
python -m examples.text2image \
    --prompt "${PROMPT_}" \
    --model_size 0.6B \
    --save_path output/text2image_0p6b_tokenizer_fast.jpg \
    --use_fast_tokenizer
```
Fast tokenizer is available only for 0.6B model.


### Batch Image Generation

For generating multiple images with different prompts, you can use a JSON file with batch inputs. The JSON file should contain an array of objects, where each object has:

- `prompt`: The text prompt describing the desired image (required)
- `output_image`: The path where the generated image should be saved (required)

An example can be found in `assets/text2image/batch_example.json`:

```json
[
  {
    "prompt": "A well-worn broom sweeps across a dusty wooden floor, its bristles gathering crumbs and flecks of debris in swift, rhythmic strokes. Dust motes dance in the sunbeams filtering through the window, glowing momentarily before settling. The quiet swish of straw brushing wood is interrupted only by the occasional creak of old floorboards. With each pass, the floor grows cleaner, restoring a sense of quiet order to the humble room.",
    "output_image": "output/sweeping-broom-sunlit-floor.jpg"
  },
  {
    "prompt": "A laundry machine whirs to life, tumbling colorful clothes behind the foggy glass door. Suds begin to form in a frothy dance, clinging to fabric as the drum spins. The gentle thud of shifting clothes creates a steady rhythm, like a heartbeat of the home. Outside the machine, a quiet calm fills the room, anticipation building for the softness and warmth of freshly laundered garments.",
    "output_image": "output/laundry-machine-spinning-clothes.jpg"
  },
  {
    "prompt": "A robotic arm tightens a bolt beneath the hood of a car, its tool head rotating with practiced torque. The metal-on-metal sound clicks into place, and the arm pauses briefly before retracting with a soft hydraulic hiss. Overhead lights reflect off the glossy vehicle surface, while scattered tools and screens blink in the background—a garage scene reimagined through the lens of precision engineering.",
    "output_image": "output/robotic-arm-car-assembly.jpg"
  }
]
```

Specify the input via the `--batch_input_json` argument:

```bash
# Run batch text2image generation
python -m examples.text2image \
    --model_size 2B \
    --batch_input_json assets/text2image/batch_example.json
```

This will generate three separate images according to the prompts specified in the JSON file, with each output saved to its corresponding path.

## API Documentation

The `text2image.py` script supports the following command-line arguments:

Input and output parameters:

- `--prompt`: Text prompt describing the image to generate (default: predefined example prompt)
- `--negative_prompt`: Text describing what to avoid in the generated image (default: empty)
- `--aspect_ratio`: Aspect ratio of the generated output (width:height) (choices: "1:1", "4:3", "3:4", "16:9", "9:16", default: "16:9")
- `--save_path`: Path to save the generated image (default: "output/generated_image.jpg")
- `--batch_input_json`: Path to JSON file containing batch inputs, where each entry should have 'prompt' and 'output_image' fields

Model selection:

- `--model_size`: Size of the model to use (choices: "0.6B", "2B", "14B", default: "2B")
- `--dit_path`: Custom path to the DiT model checkpoint for post-trained models (default: uses standard checkpoint path based on model_size)
- `--load_ema`: Whether to use EMA weights from the post-trained DIT model checkpoint for generation.
- `--use_fast_tokenizer`: Whether to use fast image tokenizer for generation. This option is available only for 0.6B model.

Generation parameters:

- `--seed`: Random seed for reproducible results (default: 0)

Performance optimization parameters:

- `--use_cuda_graphs`: Use CUDA Graphs to accelerate DiT inference.
- `--benchmark`: Run in benchmark mode to measure average generation time.

Content safety:

- `--disable_guardrail`: Disable guardrail checks on prompts (by default, guardrails are enabled to filter harmful content)

> **Note**: Text2Image runs on a single GPU and does not support multi-GPU inference. For multi-GPU video generation, use [Text2World](inference_text2world.md) or [Video2World](inference_video2world.md) pipelines.

## Prompt Engineering Tips

For best results with Cosmos models, create detailed prompts that emphasize physical realism, natural laws, and real-world behaviors. Describe specific objects, materials, lighting conditions, and spatial relationships while maintaining logical consistency throughout the scene.

Incorporate photography terminology like composition, lighting setups, and camera settings. Use concrete terms like "natural lighting" or "wide-angle lens" rather than abstract descriptions, unless intentionally aiming for surrealism. Include negative prompts to explicitly specify undesired elements.

The more grounded a prompt is in real-world physics and natural phenomena, the more physically plausible and realistic the generated image will be.

## Related Documentation

- [Text2World Inference Guide](inference_text2world.md) - Generate videos directly from text prompts
- [Video2World Inference Guide](inference_video2world.md) - Generate videos from text and visual inputs
- [Setup Guide](setup.md) - Environment setup and checkpoint download instructions
- [Performance Guide](performance.md) - Hardware requirements and optimization recommendations
