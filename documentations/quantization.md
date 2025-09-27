# 4-bit Quantization for Single GPU Inference

## Overview
This guide enables running Cosmos-Predict2 2B on a **single RTX 4090** (24GB) using 4-bit quantization.

## Hardware Requirements
- **Minimum**: 1x RTX 4090 (24GB VRAM)

## Quick Start

### Single RTX 4090:
```bash
python examples/video2world.py \
  --model_size 2B \
  --quantization 4bit \
  --quantization_double_quant \
  --input_path "image.jpg" \
  --prompt "Your creative prompt" \
  --resolution 480 \
  --disable_guardrail \
  --disable_prompt_refiner
```

### Multi-GPU (when available):
```bash
export CUDA_VISIBLE_DEVICES=0,1
python examples/video2world.py \
  --quantization 4bit \
  [other args...]
```

## Technical Details
- Uses BitsAndBytes NF4 quantization
- Environment variable based configuration
- Backward compatible (no quantization by default)
- Proper device mapping for multi-GPU setups