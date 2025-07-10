# Performance Guide

Cosmos models come in different sizes and variants, each with different hardware requirements and performance characteristics. This guide will help you choose the right model for your needs.

## Hardware Requirements

The following table shows the GPU memory requirements for different Cosmos models:

| Model | Required GPU VRAM |
|-------|-------------------|
| Cosmos-Predict2-2B-Text2Image | 26.02 GB |
| Cosmos-Predict2-14B-Text2Image | 48.93 GB |
| Cosmos-Predict2-2B-Video2World | 32.54 GB |
| Cosmos-Predict2-14B-Video2World | 56.38 GB |

For optimal performance, we recommend:
* NVIDIA GPUs with Ampere architecture (RTX 30 Series, A100) or newer
* At least 32GB of GPU VRAM for 2B models
* At least 64GB of GPU VRAM for 14B models

## Performance Benchmarks

### Inference performance

The following table shows generation times across different NVIDIA GPU hardware:

| GPU Hardware | 2B-Text2Image | 14B-Text2Image | 2B-Video2World | 14B-Video2World |
|--------------|---------------|----------------|----------------|-----------------|
| NVIDIA GB200 | 3.39 sec | 8.5 sec | 25.61 sec | 85.26 sec |
| NVIDIA B200 | 3.24 sec | 8.68 sec | 30.7 sec | 92.59 sec |
| NVIDIA RTX PRO 6000 | 5.59 sec | 24.16 sec | 82.43 sec | 321.9 sec |
| NVIDIA DGX Spark | 24.87 sec | 138.94 sec | 344.64 sec | 1902.26 sec |
| NVIDIA H200 SXM | 9.02 sec | 15.96 sec | 50.2 sec | 176.19 sec |
| NVIDIA H200 NVL | 6.34 sec | 16.95 sec | 54.01 sec | 203.56 sec |
| NVIDIA H100 PCIe | 11.12 sec | 23.83 sec | 79.87 sec | 286.46 sec |
| NVIDIA H100 NVL | 5.05 sec | 23.97 sec | 87.32 sec | 377.67 sec |
| NVIDIA H20 | 11.47 sec | 59.59 sec | 179.69 sec | 852.64 sec |
| NVIDIA L40S | 8.9 sec | (OOM) | 127.49 sec | 1036.24 sec |
| NVIDIA RTX 6000 Ada | 11.94 sec | 167.86 sec | 180.99 sec | 876.68 sec |

Note: (OOM) indicates "Out of Memory" - the model is too large to run on that GPU.

Note: Video2World was run with 480p resolution and at 16 FPS.

### Post-training performance

Review the [AgiBot-Fisheye](post-training_video2world_agibot_fisheye.md) post-training example, which contains performance numbers on different GPUs.

## Model Selection Guide

It is recommended to use the 2B models for
- faster inference times and lower latency
- limited GPU memory (requires ~26-33GB VRAM)
- simpler scenes and compositions
- rapid prototyping or testing
- processing large batches of images/videos efficiently

It is recommended to use the 14B models for
- higher quality and more detailed outputs
- sufficient GPU resources (requires ~49-57GB VRAM)
- complex scenes with intricate details
- quality is prioritized over generation speed
- final production assets

The 14B models generally produce higher fidelity results with better coherence and detail, but come with increased computational costs. The 2B models offer a good balance of quality and performance for many practical applications while being more resource-efficient.

For most development and testing scenarios, starting with the 2B models is recommended. You can then scale up to 14B models when higher quality is needed and hardware resources permit.
