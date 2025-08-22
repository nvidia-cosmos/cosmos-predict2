<p align="center">
    <img src="assets/nvidia-cosmos-header.png" alt="NVIDIA Cosmos Header">
</p>

### Paper (coming soon!) | [Website](https://research.nvidia.com/labs/dir/cosmos-predict2/) | [Hugging Face](https://huggingface.co/collections/nvidia/cosmos-predict2-68028efc052239369a0f2959) | [PyPI](https://pypi.org/project/cosmos-predict2/)

Cosmos-Predict2 is a key branch of the [Cosmos World Foundation Models](https://www.nvidia.com/en-us/ai/cosmos) (WFMs) ecosystem for Physical AI, specializing in future state prediction through advanced world modeling. It offers two powerful capabilities: text-to-image generation for creating high-quality images from text descriptions, and video-to-world generation for producing visual simulations from video inputs.

We visualize the architecture of Cosmos-Predict2 in the following figure.

<p align="center">
    <img src="assets/cosmos-predict-diagram.png" alt="Cosmos-Predict Architecture Diagram" width=80%>
</p>

## News
* 2025-08-22: Cosmos-Predict2 is now available on [PyPI](https://pypi.org/project/cosmos-predict2/)! See [Getting Started](#getting-started) for usage.
* 2025-08-21: Cosmos-Predict2 now has pre-built dependencies! See [Setup Guide](documentations/setup.md).
* 2025-08-15: We released the [0.6B Text2Image](documentations/inference_text2image.md) model with fast tokenizer support!
* 2025-07-10: We released [Predict2 + NATTEN](documentations/performance.md#sparse-attention-powered-by-natten), bringing up to 2.6X end-to-end inference speedup with sparse attention ([Video](https://www.youtube.com/watch?v=o396JZsz4V4)).
* 2025-06-11: We released post-training and inference code, along with model weights. For a code walkthrough, please see this [video](https://www.youtube.com/watch?v=ibnVm6hPtxA).

## Models

* [Cosmos-Predict2-2B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image): Text-to-image generation
* [Cosmos-Predict2-14B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Text2Image): Text-to-image generation
* [Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World): Video + Text based future visual world generation
* [Cosmos-Predict2-14B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Video2World): Video + Text based future visual world generation
* [Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1): Video + Text based future visual world generation, post-trained on GR00T Dreams GR1 dataset
* [Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID](https://huggingface.co/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID): Video + Text based future visual world generation, post-trained on GR00T Dreams DROID dataset
* [Cosmos-Predict2-2B-Sample-Action-Conditioned](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned): Video + Action based future visual world generation, post-trained on Bridge dataset
---

## Getting Started

System Requirements:

* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* NVIDIA driver compatible with CUDA 12.6
* Linux x86-64
* glibc>=2.31 (e.g Ubuntu >=22.04)
* Python 3.10

We **HIGHLY** recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/).

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Cosmos-Predict2 can be installed with `pip` (requires `python==3.10`):

```shell
uv venv --python 3.10 --allow-existing
uv pip install -U "cosmos-predict2[cu126]" --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/cu126_torch260/simple
```

[Example Project](examples/project/README.md)

To run the repository examples below, please follow the [Setup Guide](documentations/setup.md)

## Diffusers

Cosmos-Predict2 is included in [`diffusers>=0.34.0`](https://huggingface.co/docs/transformers/en/index).

Run example inference scripts:

* [Text2Image](scripts/hf_text2image.py)

  ```shell
  ./scripts/hf_text2image.py output/hf_text2image --prompt "assets/text2image/example_prompt.txt" -v
  ```

* [Video2World](scripts/hf_video2world.py)

  ```shell
  ./scripts/hf_video2world.py output/hf_video2world --prompt "assets/video2world/example_prompt.txt" --image "assets/video2world/example_input.jpg" -v
  ```

## Quick Start

Here is a quick example demonstrating how to use Cosmos-Predict2-2B-Video2World for video generation:

```python
import torch
from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

# Create the video generation pipeline.
pipe = Video2WorldPipeline.from_config(
    config=get_cosmos_predict2_video2world_pipeline(model_size="2B"),
    dit_path=get_cosmos_predict2_video2world_checkpoint(model_size="2B"),
)

# Specify the input image path and text prompt.
image_path = "assets/video2world/example_input.jpg"
prompt = "A high-definition video captures the precision of robotic welding in an industrial setting. The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation."

# Run the video generation pipeline.
video = pipe(input_path=image_path, prompt=prompt)

# Save the resulting output video.
save_image_or_video(video, "output/test.mp4", fps=16)
```

**Input prompt:**
> A high-definition video captures the precision of robotic welding in an industrial setting. The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure. The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues. A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity. The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery, indicating a busy and functional industrial workspace. As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left. The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward. The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance, with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation.

| Input image | Output video |
|-------------|--------------|
| ![Input Image](assets/video2world/example_input.jpg) | <video width="512" src="https://github-production-user-asset-6210df.s3.amazonaws.com/8789158/454153937-f015a579-1a8c-4c7f-8683-de2913e1c2f4.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20250611%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250611T235940Z&X-Amz-Expires=300&X-Amz-Signature=97df8ab45e8de22d94d10b01eb6591fcdc234bf3faed4f3fe50f50d92722af21&X-Amz-SignedHeaders=host"></video> |

---

## User Guide
Our [setup guide](documentations/setup.md) provides complete information on
* [System requirements](documentations/setup.md#system-requirements): Detailed hardware and software prerequisites
* [Installation](documentations/setup.md#installation): Step-by-step setup with both Conda and Docker options
* [Downloading checkpoints](documentations/setup.md#downloading-checkpoints): Instructions for obtaining model weights
* [Troubleshooting](documentations/setup.md#troubleshooting): Solutions for common installation and CUDA compatibility issues

For inference examples and usage
* **[Text2Image Inference](documentations/inference_text2image.md)**: Guide for generating high-quality images from text prompts
* **[Video2World Inference](documentations/inference_video2world.md)**: Guide for generating videos from images/videos with text prompts, including:
  * Single and batch processing
  * Multi-frame conditioning
  * Multi-GPU inference for faster generation
  * Using the prompt refiner
  * Rejection sampling for quality improvement
* **[Text2World Inference](documentations/inference_text2world.md)**: Guide for generating videos directly from text prompts, including:
  * Single and batch processing
  * Multi-GPU inference for faster generation

For post-training customization
* **[Video2World Post-training guide](documentations/post-training_video2world.md)**: General guide to the video2world training system in the codebase.
* **[Video2World Post-training on Cosmos-NeMo-Assets](documentations/post-training_video2world_cosmos_nemo_assets.md)**: Case study for post-training on Cosmos-NeMo-Assets data
* **[Video2World Post-training on fisheye-view AgiBotWorld-Alpha dataset](documentations/post-training_video2world_agibot_fisheye.md)**: Case study for post-training on fisheye-view robot videos from AgiBotWorld-Alpha dataset.
* **[Video2World Post-training on GR00T Dreams GR1 and DROID datasets](documentations/post-training_video2world_gr00t.md)**: Case study for post-training on GR00T Dreams GR1 and DROID datasets.
* **[Video2World Action-conditioned Post-training on Bridge dataset](documentations/post-training_video2world_action.md)**: Case study for action-conditioned post-training on Bridge dataset.
* **[Text2Image Post-training guide](documentations/post-training_text2image.md)**: General guide to the text2image training system in the codebase.
* **[Text2Image Post-training on Cosmos-NeMo-Assets](documentations/post-training_text2image_cosmos_nemo_assets.md)**: Case study for post-training on Cosmos-NeMo-Assets image data.

Our [performance guide](documentations/performance.md) includes
* [Hardware requirements](documentations/performance.md#hardware-requirements): Recommended GPU configurations and memory requirements
* [Performance benchmarks](documentations/performance.md#performance-benchmarks): Detailed speed and quality comparisons across different GPU architectures
* [Model selection guide](documentations/performance.md#model-selection-guide): Practical advice for choosing between 2B and 14B variants based on your needs

---

## Contributing

We thrive on community collaboration! [NVIDIA-Cosmos](https://github.com/nvidia-cosmos/) wouldn't be where it is without contributions from developers like you. Check out our [Contributing Guide](CONTRIBUTING.md) to get started, and share your feedback through issues.

Big thanks 🙏 to everyone helping us push the boundaries of open-source physical AI!

---

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

This model includes safety and content moderation features powered by Llama Guard 3. Llama Guard 3 is used solely as a content input filter and is subject to its own license.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
