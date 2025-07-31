# Video2World Inference Guide

This guide provides instructions on running inference with Cosmos-Predict2 Video2World models.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Overview](#overview)
- [Examples](#examples)
  - [Single Video Generation](#single-video-generation)
  - [Batch Video Generation](#batch-video-generation)
  - [Multi-Frame Video Conditioning](#multi-frame-video-conditioning)
  - [Using the Prompt Refiner](#using-the-prompt-refiner)
  - [Multi-GPU Inference](#multi-gpu-inference)
  - [Rejection Sampling for Quality Improvement](#rejection-sampling-for-quality-improvement)
- [API Documentation](#api-documentation)
- [Prompt Engineering Tips](#prompt-engineering-tips)

## Prerequisites

Before running inference:

1. **Environment setup**: Follow the [Setup guide](setup.md) for installation instructions.
2. **Model checkpoints**: Download required model weights following the [Downloading Checkpoints](setup.md#downloading-checkpoints) section in the Setup guide.
3. **Hardware considerations**: Review the [Performance guide](performance.md) for GPU requirements and model selection recommendations.

## Overview

Cosmos-Predict2 provides two models for generating videos from a combination of text and visual inputs: `Cosmos-Predict2-2B-Video2World` and `Cosmos-Predict2-14B-Video2World`. These models can transform a still image or video clip into a longer, animated sequence guided by the text description.

The inference script is located at `examples/video2world.py`.
It requires input arguments:
- `--input_path`: input image or video
- `--prompt`: text prompt

By default the checkpoint you downloaded from the [Downloading Checkpoints](setup.md#downloading-checkpoints) section in the Setup guide are for 720P and 16FPS. If you instead want to change the behavior to, say, 480P and 10FPS, you need to download the corresponding checkpoint and pass `--fps 10 --resolution 480`.

For a complete list of available arguments and options:
```bash
python -m examples.video2world --help
```

## Examples

### Single Video Generation

#### Using the 2B model

This is a basic example for running inference on the 2B model with a single image.
The output is saved to `output/video2world_2b.mp4`.
The corresponding input prompt is saved to `output/video2world_2b.txt`.

```bash
# Set the input prompt
PROMPT_="A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."
# Run video2world generation
python -m examples.video2world \
    --model_size 2B \
    --input_path assets/video2world/input0.jpg \
    --num_conditional_frames 1 \
    --prompt "${PROMPT_}" \
    --save_path output/video2world_2b.mp4
```

#### Using the 14B model

The 14B model can be run similarly by changing the model size parameter. For GPUs with lower memory limit, it may also make sense to offload guardrail and prompt refiner models.

```bash
# Set the input prompt
PROMPT_="A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene."
# Run video2world generation
python -m examples.video2world \
    --model_size 14B \
    --input_path assets/video2world/input0.jpg \
    --num_conditional_frames 1 \
    --prompt "${PROMPT_}" \
    --save_path output/video2world_14b.mp4 \
    --offload_guardrail \
    --offload_prompt_refiner
```

The 14B model requires significant GPU memory, so it is recommended to offload the prompt refiner or guardrail models to CPU to conserve GPU memory.

### Batch Video Generation

For generating multiple videos with different inputs and prompts, you can use a JSON file with batch inputs. The JSON file should contain an array of objects, where each object has:
- `input_video`: The path to the input image or video (required)
- `prompt`: The text prompt describing the desired video (required)
- `output_video`: The path where the generated video should be saved (required)

An example can be found in `assets/video2world/batch_example.json`:
```json
[
  {
    "input_video": "assets/video2world/input0.jpg",
    "prompt": "A nighttime city bus terminal gradually shifts from stillness to subtle movement. At first, multiple double-decker buses are parked under the glow of overhead lights, with a central bus labeled '87D' facing forward and stationary. As the video progresses, the bus in the middle moves ahead slowly, its headlights brightening the surrounding area and casting reflections onto adjacent vehicles. The motion creates space in the lineup, signaling activity within the otherwise quiet station. It then comes to a smooth stop, resuming its position in line. Overhead signage in Chinese characters remains illuminated, enhancing the vibrant, urban night scene.",
    "output_video": "output/bus-terminal-night-movement.mp4"
  },
  {
    "input_video": "assets/video2world/input1.jpg",
    "prompt": "As the red light shifts to green, the red bus at the intersection begins to move forward, its headlights cutting through the falling snow. The snowy tire tracks deepen as the vehicle inches ahead, casting fresh lines onto the slushy road. Around it, streetlights glow warmer, illuminating the drifting flakes and wet reflections on the asphalt. Other cars behind start to edge forward, their beams joining the scene. The stillness of the urban street transitions into motion as the quiet snowfall is punctuated by the slow advance of traffic through the frosty city corridor.",
    "output_video": "output/snowy-intersection-traffic.mp4"
  },
  {
    "input_video": "assets/video2world/input2.jpg",
    "prompt": "In the later moments of the video, the female worker in the front, dressed in a white coat and hairnet, performs a repetitive yet precise task. She scoops golden granular material from a wide jar and steadily pours it into the next empty glass bottle on the conveyor belt. Her hand moves with practiced control as she aligns the scoop over each container, ensuring an even fill. The sequence highlights her focused attention and consistent motion, capturing the shift from preparation to active material handling as the production line advances bottle by bottle.",
    "output_video": "output/factory-worker-bottle-filling.mp4"
  }
]
```

Specify the input via the `--batch_input_json` argument:
```bash
# Run batch video2world generation
python -m examples.video2world \
    --model_size 2B \
    --batch_input_json assets/video2world/batch_example.json
```

This will generate three separate videos according to the inputs and prompts specified in the JSON file, with each output saved to its corresponding path.

### Multi-Frame Video Conditioning

Video2World models support two types of conditioning on visual input:

1. **Single-frame conditioning (default)**: Uses 1 frame from an image or video for conditioning
2. **Multi-frame conditioning**: Uses the last 5 consecutive frames from a video for enhanced temporal consistency

Using multiple frames as conditioning input can provide better temporal coherence in the generated video by giving the model more context about the motion present in the original sequence.

Multi-frame conditioning is particularly effective when:
- Preservation of specific motion patterns from the input video is desired
- The input contains complex or distinctive movements that should be maintained
- Stronger visual coherence between the input and output videos is needed
- Extending or transforming an existing video clip is the goal

For 5-frame conditioning, the input must be a video file, not a still image. Specify the number of conditional frames with the `--num_conditional_frames 5` argument:

```bash
# Set the input prompt
PROMPT_="A point-of-view video shot from inside a vehicle, capturing a quiet suburban street bathed in bright sunlight. The road is lined with parked cars on both sides, and buildings, likely residential or small businesses, are visible across the street. A STOP sign is prominently displayed near the center of the intersection. The sky is clear and blue, with the sun shining brightly overhead, casting long shadows on the pavement. On the left side of the street, several vehicles are parked, including a van with some text on its side. Across the street, a white van is parked near two trash bins, and a red SUV is parked further down. The buildings on either side have a mix of architectural styles, with some featuring flat roofs and others with sloped roofs. Overhead, numerous power lines stretch across the street, and a few trees are visible in the background, partially obscuring the view of the buildings. As the video progresses, a white car truck makes a right turn into the adjacent opposite lane. The ego vehicle slows down and comes to a stop, waiting until the car fully enters the opposite lane before proceeding. The pedestrian keeps walking on the street. The other vehicles remain stationary, parked along the curb. The scene remains static otherwise, with no significant changes in the environment or additional objects entering the frame. By the end of the video, the white car truck has moved out of the camera view, the rest of the scene remains largely unchanged, maintaining the same composition and lighting conditions as the beginning."
# Run video2world generation with 5-frame conditioning
python -m examples.video2world \
    --model_size 2B \
    --input_path assets/video2world/input3.mp4 \
    --num_conditional_frames 5 \
    --prompt "${PROMPT_}" \
    --save_path output/video2world_2b_5frames.mp4
```

Note that when using multi-frame conditioning in batch mode, all input files must be videos, not images.

Notes on multi-frame conditioning:
- Multi-frame conditioning requires video inputs with at least 5 frames
- The model will extract the last 5 frames from the input video

### Using the Prompt Refiner

The Cosmos-Predict2 models include a prompt refiner model using [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B) that automatically enhances short prompts with additional details. This is particularly useful when:
* Brief prompts need to be expanded into more detailed videos
* Additional descriptive elements would improve video quality
* Detailed prompt writing expertise is limited

The following example uses a short prompt that will be automatically expanded by the prompt refiner:
```bash
# Set the input short prompt
PROMPT_="A nighttime city bus terminal."
# Run video2world generation
python -m examples.video2world \
    --model_size 2B \
    --input_path assets/video2world/input0.jpg \
    --num_conditional_frames 1 \
    --prompt "${PROMPT_}" \
    --save_path output/video2world_2b_with_prompt_refiner.mp4
```

The prompt refiner is enabled by default. To disable it, use the `--disable_prompt_refiner` flag:
```bash
# Run video2world generation without prompt refinement
python -m examples.video2world \
    --model_size 2B \
    --input_path assets/video2world/input0.jpg \
    --prompt "${PROMPT_}" \
    --disable_prompt_refiner \
    --save_path output/video2world_2b_without_prompt_refiner.mp4
```

This configuration can be seen in the model's configuration:
```python
prompt_refiner_config=CosmosReason1Config(
    checkpoint_dir="checkpoints/nvidia/Cosmos-Reason1-7B",
    offload_model_to_cpu=True,
    enabled=True,  # Controls whether the refiner is used
)
```

### Multi-GPU Inference

For faster inference on high-resolution videos, Video2World supports context parallelism, which distributes the video frames across multiple GPUs. This can significantly reduce the inference time, especially for the larger 14B model.

To enable multi-GPU inference, set the `NUM_GPUS` environment variable and use `torchrun` to launch the script. Both `--nproc_per_node` and `--num_gpus` should be set to the same value:

```bash
# Set the number of GPUs to use
export NUM_GPUS=8

# Run video2world generation with context parallelism using torchrun
torchrun --nproc_per_node=${NUM_GPUS} examples/video2world.py \
    --model_size 2B \
    --input_path assets/video2world/input0.jpg \
    --prompt "${PROMPT_}" \
    --save_path output/video2world_2b_${NUM_GPUS}gpu.mp4 \
    --num_gpus ${NUM_GPUS}
```

This distributes the computation across multiple GPUs, with each GPU processing a subset of the video frames. The final video is automatically combined from the results of all GPUs.

If using the 14B model, it is recommended to offload the prompt refiner model or guardrail models to CPU to save GPU memory (see [Using the 14B Model](#using-the-14b-model) for reference).

> **Note:** Both parameters are required: `--nproc_per_node` tells PyTorch how many processes to launch, while `--num_gpus` tells the model how to distribute the workload. Using the same environment variable for both ensures they are synchronized.

Important considerations for multi-GPU inference:
- The number of GPUs should ideally be a divisor of the number of frames in the video
- All GPUs should have the same model capacity and memory
- For best results, use context parallelism with the 14B model where memory constraints are significant
- Context parallelism works with both single-frame and multi-frame conditioning
- Requires NCCL support and proper GPU interconnect for efficient communication

### Rejection Sampling for Quality Improvement

Video quality can be further improved by generating multiple variations and selecting the best one based on automatic quality assessment using [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B) as the critic model. This approach, known as rejection sampling, can significantly enhance the visual quality of the generated videos.

```bash
# Set the input prompt
PROMPT_="A nighttime city bus terminal gradually shifts from stillness to subtle movement. Multiple double-decker buses are parked under overhead lights, with a central bus labeled '87D' facing forward."
# Run video2world generation with rejection sampling
python -m examples.video2world_bestofn \
    --model_size 2B \
    --input_path assets/video2world/input0.jpg \
    --prompt "${PROMPT_}" \
    --num_generations 4 \
    --num_critic_trials 5 \
    --disable_guardrail \
    --save_path output/rejection_sampling_demo
```

This command:
1. Generates 5 different videos from the same input and prompt
2. Evaluates each video 3 times using the Cosmos-Reason1 critic model
3. Saves all videos with quality scores in their filenames (from 000 to 100)
4. Creates HTML reports with detailed analysis for each video

The highest-scored video represents the best generation from the batch. For batch processing with existing videos:

```bash
# Run critic on existing videos without generation
python -m examples.video2world_bestofn \
    --skip_generation \
    --save_path output/my_existing_videos
```

### Long Video Generation

In a single forward pass of the Video2World model, we only generate one chunk of video. To generate longer videos of multiple chunks, we support long video generation in an auto-regressive inference manner. The idea is to generate the first chunk, then iteratively taking the last `num_conditional_frames` frames of the previous chunk as input condition of next chunk.

Since long video generation calls the whole denoising process of Video2World model for `num_chunks` times, it's much slower than single-chunk video generation. We hence highly recommend using multi-GPU inference to boost the speed.

```bash
# Set the input prompt
PROMPT_="The video opens with a view inside a well-lit warehouse or retail store aisle, characterized by high ceilings and industrial shelving units stocked with various products. The shelves are neatly organized with items such as canned goods, packaged foods, and cleaning supplies, all displayed in bright packaging that catches the eye. The surrounding environment includes additional shelving units filled with similar products. The scene concludes with the forklift still in motion, ensuring the pallet is securely placed on the shelf."

# Set the number of GPUs to use
export NUM_GPUS=8

# Run video2world long video generation of 6 chunks
PYTHONPATH=. torchrun --nproc_per_node=${NUM_GPUS} examples/video2world_lvg.py \
    --model_size 14B \
    --num_chunks 6 \
    --input_path assets/video2world_lvg/example_input.jpg \
    --prompt "${PROMPT_}" \
    --save_path output/video2world_2b_lvg_example1.mp4 \
    --num_gpus ${NUM_GPUS} \
    --disable_guardrail \
    --disable_prompt_refiner
```

Example output is included at `assets/video2world_lvg/example_output.mp4`.

If using the 14B model, it is recommended to offload the prompt refiner model or guardrail models to CPU to save GPU memory (see [Using the 14B Model](#using-the-14b-model) for reference).

### Faster inference with Sparse Attention
If you're targeting 720p generation, and you're using a Hopper (compute capability 9.0) or
Blackwell datacenter-class (compute capability 10.0) GPU, you can optionally run
[Video2World + NATTEN](performance.md#sparse-attention-powered-by-natten) by using the `--natten`
flag.

```bash
python -m examples.video2world \
    --model_size 2B \
    --input_path $INPUT_PATH \
    --prompt "${PROMPT_}" \
    --natten \
    --save_path output/video2world_2b_with_natten.mp4
```

Running with NATTEN can bring you anywhere from 1.7X to 2.6X end-to-end speedup over the base model,
depending on variant, frame rate, and hardware. In terms of quality, we've observed that in many
domains the sparse attention variants are comparable with the base models.

## API Documentation

The `video2world.py` script supports the following command-line arguments:

Model selection:
- `--model_size`: Size of the model to use (choices: "2B", "14B", default: "2B")
- `--dit_path`: Custom path to the DiT model checkpoint for post-trained models (default: uses standard checkpoint path based on model_size)
- `--load_ema`: Whether to use EMA weights from the post-trained DIT model checkpoint for generation.
- `--fps`: FPS of the model to use for video-to-world generation (choices: 10, 16, default: 16)
- `--resolution`: Resolution of the model to use for video-to-world generation (choices: 480, 720, default: 720)

By default a 720P + 16FPS model is used for `model_size` size model. If you want to use another config, download the corresponding checkpoint and pass either `--fps` or `--resolution` or both.

Input parameters:
- `--prompt`: Text prompt describing the video to generate (default: empty string)
- `--negative_prompt`: Text describing what to avoid in the generated video (default: predefined negative prompt)
- `--aspect_ratio`: Aspect ratio of the generated output (width:height) (choices: "1:1", "4:3", "3:4", "16:9", "9:16", default: "16:9")
- `--input_path`: Path to input image or video for conditioning (default: "assets/video2world/input0.jpg")
- `--num_conditional_frames`: Number of frames to condition on (choices: 1, 5, default: 1)

If the shape of the input image/video is different from the target resolution & aspect ratio, first, the input will be resized to equal or larger lengths in height & width dimensions. Then, it will be center-cropped to match the predefined resolution for the corresponding aspect ratio.

Output parameters:
- `--save_path`: Path to save the generated video (default: "output/generated_video.mp4")

Generation parameters:
- `--guidance`: Classifier-free guidance scale (default: 7.0)
- `--seed`: Random seed for reproducibility (default: 0)
- `--num_gpus`: Number of GPUs to use for context parallel inference in the video generation phase (default: 1)

Performance parameters:
- `--use_cuda_graphs`: Use CUDA Graphs to accelerate DiT inference.
- `--natten`: Use sparse attention variants built with [NATTEN](https://natten.org). This feature is
    only available with 720p resolution, and on Hopper and specific Blackwell datacenter cards
    (B200 and GB200) for now. [Learn more](performance.md).
- `--benchmark`: Run in benchmark mode to measure average generation time.

Multi-GPU inference:
- For multi-GPU inference, use `torchrun --nproc_per_node=$NUM_GPUS examples/video2world.py ...`
- Both `--nproc_per_node` (for torchrun) and `--num_gpus` (for the script) must be set to the same value
- Setting the `NUM_GPUS` environment variable and using it for both parameters ensures they stay synchronized

Batch processing:
- `--batch_input_json`: Path to JSON file containing batch inputs, where each entry should have 'input_video', 'prompt', and 'output_video' fields

Content safety and controls:
- `--disable_guardrail`: Disable guardrail checks on prompts (by default, guardrails are enabled to filter harmful content)
- `--disable_prompt_refiner`: Disable prompt refiner that enhances short prompts (by default, the prompt refiner is enabled)

GPU memory controls:
- `--offload_guardrail`: Offload guardrail to CPU to save GPU memory
- `--offload_prompt_refiner`: Offload prompt refiner to CPU to save GPU memory
## Specialized Scripts

In addition to the main `video2world.py` script, there are specialized variants for specific use cases:

### Rejection Sampling (video2world_bestofn.py)

The `video2world_bestofn.py` script extends the standard Video2World capabilities with rejection sampling to improve video quality. It supports all the standard Video2World parameters plus:

- `--num_generations`: Number of different videos to generate from the same input (default: 2)
- `--num_critic_trials`: Number of times to evaluate each video with the critic model (default: 5)
- `--skip_generation`: Flag to run critic only on existing videos without generation
- `--save_path`: Directory to save the generated videos and HTML reports (default: "output/best-of-n")

For more details, see the [Rejection Sampling for Quality Improvement](#rejection-sampling-for-quality-improvement) section.

## Prompt Engineering Tips

For best results with Video2World models, create detailed prompts that emphasize:

1. **Physical realism**: Describe how objects interact with the environment following natural laws of physics
2. **Motion details**: Specify how elements in the scene should move over time
3. **Visual consistency**: Maintain logical relationships between objects throughout the video
4. **Cinematography terminology**: Use terms like "tracking shot," "pan," or "zoom" to guide camera movement
5. **Temporal progression**: Describe how the scene evolves (e.g., "gradually," "suddenly," "transitions to")
6. **Cinematography terms**: Include camera movements like "panning across," "zooming in," or "tracking shot"

Include negative prompts to explicitly specify undesired elements, such as jittery motion, visual artifacts, or unrealistic physics.

The more grounded a prompt is in real-world physics and natural temporal progression, the more physically plausible and realistic the generated video will be.

Example of a good prompt:
```
A tranquil lakeside at sunset. Golden light reflects off the calm water surface, gradually rippling as a gentle breeze passes through. Tall pine trees along the shore sway slightly, their shadows lengthening across the water. A small wooden dock extends into the lake, where a rowboat gently bobs with the subtle movements of the water.
```

This prompt includes both static scene elements and suggestions for motion that the Video2World model can interpret and animate.

## Related Documentation

- [Text2Image Inference Guide](inference_text2image.md) - Generate still images from text prompts
- [Text2World Inference Guide](inference_text2world.md) - Generate videos directly from text prompts
- [Setup Guide](setup.md) - Environment setup and checkpoint download instructions
- [Performance Guide](performance.md) - Hardware requirements and optimization recommendations
- [Training Cosmos-NeMo-Assets Guide](video2world_post-training_cosmos_nemo_assets.md) - Information on training on Cosmos-NeMo-Assets dataset.
