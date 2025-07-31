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

### Sparse Attention powered by [NATTEN](https://natten.org)
Video2World offers variants trained with sparse attention, which can accelerate inference up to 2.5X
on the Hopper and Blackwell architectures with comparable quality.
This feature is only available for 720p inference, and only on NVIDIA GPUs with compute capability
9.0 or 10.0.

Since many concurrent works in sparse attention for video generation report performance numbers with
Flash Attention V2 as baseline, we note that our baseline models run with SOTA attention kernels
for the Hopper ([Flash Attention V3](https://arxiv.org/abs/2407.08608)) and Blackwell (cuDNN
Attention) architectures.

NATTEN's [Hopper](https://natten.org/backends/#hopper-fna-fmha) and
[Blackwell FNA](https://natten.org/backends/#blackwell-fna-fmha) kernels can deliver speedups
**proportional to reduction in FLOPs** over FAv3 and cuDNN's Blackwell FMHA.

The following table shows generation times (720p, 16fps) with and without sparsity across supported NVIDIA GPUs:

| GPU Hardware     | 2B-Video2World | 2B-Video2World + [NATTEN](https://natten.org) | 14B-Video2World | 14B-Video2World + [NATTEN](https://natten.org) |
|------------------|----------------|------------------|----------------|------------------|
| NVIDIA B200      | 123.9 sec      | 54.0 sec (2.3X)  | 439.4 sec      | 223.1 sec (2.0X) |
| NVIDIA H200 SXM  | 221.7 sec      | 89.4 sec (2.5X)  | 836.9 sec      | 412.9 sec (2.0X) |
| NVIDIA H200 NVL  | 267.2 sec      | 104.3 sec (2.6X) | 1006.7 sec     | 489.5 sec (2.1X) |
| NVIDIA H100 PCIe | 378.5 sec      | 149.6 sec (2.5X) | 1425.4 sec     | 706.9 sec (2.0X) |
| NVIDIA H100 NVL  | 355.7 sec      | 138.7 sec (2.6X) | 1348.6 sec     | 677.0 sec (2.0X) |
| NVIDIA H100 SXM  | 228.8 sec      | 94.2 sec (2.4X)  | 856.9 sec      | 426.0 sec (2.0X) |

The following table shows generation times (720p, 10fps) with and without sparsity across supported NVIDIA GPUs:

| GPU Hardware     | 2B-Video2World | 2B-Video2World + [NATTEN](https://natten.org) | 14B-Video2World | 14B-Video2World + [NATTEN](https://natten.org) |
|------------------|----------------|------------------|----------------|------------------|
| NVIDIA B200      | 62.4 sec       | 32.6 sec (1.9X)  | 230.0 sec      | 136.5 sec (1.7X) |
| NVIDIA H200 SXM  | 111.1 sec      | 52.9 sec (2.1X)  | 436.7 sec      | 252.1 sec (1.7X) |
| NVIDIA H200 NVL  | 133.1 sec      | 60.7 sec (2.2X)  | 519.3 sec      | 296.6 sec (1.8X) |
| NVIDIA H100 PCIe | 187.9 sec      | 87.4 sec (2.1X)  | 749.2 sec      | 439.3 sec (1.7X) |
| NVIDIA H100 NVL  | 175.5 sec      | 79.0 sec (2.2X)  | 711.5 sec      | 418.0 sec (1.7X) |
| NVIDIA H100 SXM  | 115.1 sec      | 56.0 sec (2.0X)  | 447.9 sec      | 260.0 sec (1.7X) |

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

If you have a Hopper (compute capability 9.0) or Blackwell datacenter-class
(compute capability 10.0) GPU, you can also experiment with the Sparse Attention variants. Sparse
variants are comparable in terms of visual quality with their base counterparts across various
domains.
