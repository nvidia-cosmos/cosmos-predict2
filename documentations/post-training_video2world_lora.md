# Predict2 Video2World LoRA Post-Training Guide

This guide provides instructions on running LoRA (Low-Rank Adaptation) post-training with Cosmos-Predict2 Video2World models.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Overview](#overview)
- [Post-training Guide](#post-training-guide)

## Prerequisites

Before running LoRA post-training:

1. **Environment setup**: Follow the [Setup guide](setup.md) for installation instructions.
2. **Model checkpoints**: Download required model weights following the [Downloading Checkpoints](setup.md#downloading-checkpoints) section in the Setup guide.
3. **Hardware considerations**: Review the [Performance guide](performance.md) for GPU requirements and model selection recommendations.
4. **PEFT library**: The LoRA training uses the PEFT (Parameter-Efficient Fine-Tuning) library which should be installed as part of the environment setup.

## Overview

Cosmos-Predict2 provides two models for generating videos from a combination of text and visual inputs: `Cosmos-Predict2-2B-Video2World` and `Cosmos-Predict2-14B-Video2World`. These models can transform a still image or video clip into a longer, animated sequence guided by the text description.

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that allows you to adapt large pre-trained models to specific domains or tasks by training only a small number of additional parameters. This approach offers several advantages:

### Key Benefits of LoRA Post-Training

- **Memory Efficiency**: Only trains a small subset of parameters (typically < 1% of total model parameters)
- **Faster Training**: Significantly reduced training time compared to full fine-tuning
- **Storage Efficiency**: LoRA checkpoints are much smaller than full model checkpoints
- **Flexibility**: Can maintain multiple LoRA adapters for different domains
- **Preserved Base Capabilities**: Retains the original model's capabilities while adding domain-specific improvements

We support LoRA post-training with example datasets:
- [post-training_video2world_cosmos_nemo_assets](/documentations/post-training_video2world_cosmos_nemo_assets.md)
  - Basic examples with a small 4 videos dataset (can be adapted for LoRA)
- [post-training_video2world_agibot_fisheye](/documentations/post-training_video2world_agibot_fisheye.md)
  - Examples with fisheye-view dataset (can be adapted for LoRA)
- [post-training_video2world_gr00t](/documentations/post-training_video2world_gr00t.md)
  - Examples with GR00T-dreams datasets (can be adapted for LoRA)

## Post-training Guide

### 1. Preparing Data

The LoRA post-training data preparation follows the same format as standard post-training.

Dataset folder format:
```
datasets/custom_video2world_lora_dataset/
├── metas/
│   ├── *.txt
├── videos/
│   ├── *.mp4
```

`metas` folder contains `.txt` files containing prompts describing the video content.
`videos` folder contains the corresponding `.mp4` video files.

After preparing `metas` and `videos` folders, run the following command to pre-compute T5-XXL embeddings:
```bash
python -m scripts.get_t5_embeddings --dataset_path datasets/custom_video2world_lora_dataset/
```

This script will create `t5_xxl` folder under the dataset root where the T5-XXL embeddings are saved as `.pickle` files:
```
datasets/custom_video2world_lora_dataset/
├── metas/
│   ├── *.txt
├── videos/
│   ├── *.mp4
├── t5_xxl/
│   ├── *.pickle
```

### 2. Creating Configs for LoRA Training

Define dataloader from the prepared dataset with LoRA-specific configurations.

For example:
```python
# custom LoRA dataset example
example_video_lora_dataset = L(Dataset)(
    dataset_dir="datasets/custom_video2world_lora_dataset",
    num_frames=93,
    video_size=(704, 1280),  # 720 resolution, 16:9 aspect ratio
)

dataloader_video_train_lora = L(DataLoader)(
    dataset=example_video_lora_dataset,
    sampler=L(get_sampler)(dataset=example_video_lora_dataset),
    batch_size=2,  # Can use larger batch size due to memory efficiency
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)
```

With the `dataloader_video_train_lora`, create a config for a LoRA training job.
Here's a LoRA post-training example for video2world 2B model:

```python
predict2_video2world_lora_training_2b_custom_data = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world_lora",
        name="2b_custom_data",
    ),
    model=dict(
        config=dict(
            # Enable LoRA training
            train_architecture="lora",
            # LoRA configuration parameters
            lora_rank=16,
            lora_alpha=16,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
            pipe_config=dict(
                ema=dict(enabled=True),     # enable EMA during training
                prompt_refiner_config=dict(enabled=False),  # disable prompt refiner during training
                guardrail_config=dict(enabled=False),   # disable guardrail during training
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=2,            # context parallelism size
    ),
    dataloader_train=dataloader_video_train_lora,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=2000,                      # LoRA typically needs more iterations but trains faster
    ),
    checkpoint=dict(
        save_iter=500,                      # checkpoints will be saved every 500 iterations.
    ),
    optimizer=dict(
        lr=2 ** (-10),                      # LoRA typically uses higher learning rates
    ),
    scheduler=dict(
        warm_up_steps=[0],
        cycle_lengths=[2_000],              # adjust considering max_iter
        f_max=[0.6],
        f_min=[0.0],
    ),
)
```

Here's a LoRA post-training example for video2world 14B model:

```python
predict2_video2world_lora_training_14b_custom_data = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_14b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world_lora",
        name="14b_custom_data",
    ),
    model=dict(
        config=dict(
            # Enable LoRA training
            train_architecture="lora",
            # LoRA configuration parameters
            lora_rank=32,                    # Higher rank for larger model
            lora_alpha=32,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
            pipe_config=dict(
                ema=dict(enabled=True),     # enable EMA during training
                prompt_refiner_config=dict(enabled=False),  # disable prompt refiner during training
                guardrail_config=dict(enabled=False),   # disable guardrail during training
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=4,            # Higher context parallelism for larger model
    ),
    dataloader_train=dataloader_video_train_lora,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=1500,                      # Fewer iterations for larger model
    ),
    checkpoint=dict(
        save_iter=300,                      # More frequent checkpoints
    ),
    optimizer=dict(
        lr=2 ** (-11),                      # Lower learning rate for larger model
    ),
    scheduler=dict(
        warm_up_steps=[0],
        cycle_lengths=[1_500],
        f_max=[0.6],
        f_min=[0.0],
    ),
)
```

The config should be registered to ConfigStore:
```python
for _item in [
    # 2b, custom LoRA data
    predict2_video2world_lora_training_2b_custom_data,
    # 14b, custom LoRA data
    predict2_video2world_lora_training_14b_custom_data,
]:
    # Get the experiment name from the global variable.
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
```

### 2.1. LoRA Configuration Parameters

The key LoRA-specific parameters in the config are:

| Parameter | Description | Default | Recommended Values |
|-----------|-------------|---------|-------------------|
| `train_architecture` | Set to "lora" to enable LoRA training | "full" | **"lora"** (required) |
| `lora_rank` | Rank of LoRA adaptation matrices | 16 | 8-32 (2B), 16-64 (14B) |
| `lora_alpha` | LoRA scaling parameter | 16 | Usually equals lora_rank |
| `lora_target_modules` | Target modules for LoRA adaptation | "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2" | Comma-separated module names |
| `init_lora_weights` | Initialize LoRA weights properly | True | Keep as True |

#### LoRA Parameter Guidelines

**For 2B Model:**
- **Conservative**: rank=8, alpha=8, lr=2^(-12)
- **Balanced**: rank=16, alpha=16, lr=2^(-10) 
- **Aggressive**: rank=32, alpha=32, lr=2^(-9)

**For 14B Model:**
- **Conservative**: rank=16, alpha=16, lr=2^(-12)
- **Balanced**: rank=32, alpha=32, lr=2^(-11)
- **Aggressive**: rank=64, alpha=64, lr=2^(-10)

### 2.2. Config System

In the above config example, it starts by overriding from the registered configs:
```python
    {"override /model": "predict2_video2world_fsdp_2b"},
    {"override /optimizer": "fusedadamw"},
    {"override /scheduler": "lambdalinear"},
    {"override /ckpt_type": "standard"},
    {"override /dataloader_val": "mock"},
```

The configuration system is organized as follows:

```
cosmos_predict2/configs/base/
├── config.py                   # Main configuration class definition
├── defaults/                   # Default configuration groups
│   ├── callbacks.py            # Training callbacks configurations
│   ├── checkpoint.py           # Checkpoint saving/loading configurations
│   ├── data.py                 # Dataset and dataloader configurations
│   ├── ema.py                  # Exponential Moving Average configurations
│   ├── model.py                # Model architecture configurations
│   ├── optimizer.py            # Optimizer configurations
│   └── scheduler.py            # Learning rate scheduler configurations
└── experiment/                 # Experiment-specific configurations
    ├── cosmos_nemo_assets.py   # Experiments with cosmos_nemo_assets
    ├── agibot_head_center_fisheye_color.py  # Experiments with agibot_head_center_fisheye_color
    ├── groot.py                # Experiments with groot
    └── utils.py                # Utility functions for experiments
```

The system provides several pre-defined configuration groups that can be mixed and matched:

#### Model Configurations (`defaults/model.py`)
- `predict2_video2world_fsdp_2b`: 2B parameter Video2World model with FSDP
- `predict2_video2world_fsdp_14b`: 14B parameter Video2World model with FSDP

Both can be used with LoRA by setting `train_architecture="lora"` in the model config.

#### Optimizer Configurations (`defaults/optimizer.py`)
- `fusedadamw`: FusedAdamW optimizer with standard settings
- Custom optimizer configurations for different training scenarios

LoRA typically uses higher learning rates than full fine-tuning.

#### Scheduler Configurations (`defaults/scheduler.py`)
- `constant`: Constant learning rate
- `lambdalinear`: Linearly warming-up learning rate
- Various learning rate scheduling strategies

#### Data Configurations (`defaults/data.py`)
- Training and validation dataset configurations

#### Checkpoint Configurations (`defaults/checkpoint.py`)
- `standard`: Standard local checkpoint handling

#### Callback Configurations (`defaults/callbacks.py`)
- `basic`: Essential training callbacks
- Performance monitoring and logging callbacks

In addition to the overridden values, the rest of the config setup overwrites or adds the other config details.

### 3. Run a LoRA Training Job

Run the following command to execute a LoRA post-training job with the custom data:

#### For 2B Model:
```bash
EXP=predict2_video2world_lora_training_2b_custom_data
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py \
    -- experiment=${EXP} \
    model.config.train_architecture=lora
```

#### For 14B Model:
```bash
EXP=predict2_video2world_lora_training_14b_custom_data
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py \
    -- experiment=${EXP} \
    model.config.train_architecture=lora
```

The `model.config.train_architecture=lora` parameter explicitly enables LoRA training mode.

#### Training Progress Monitoring

During LoRA training, you'll see parameter statistics like:
```
Total parameters: 2,345,678,901
Trainable LoRA parameters: 1,234,567
LoRA parameter ratio: 0.05%
```

This confirms that only a small fraction of parameters are being trained.

The LoRA checkpoints will be saved to `checkpoints/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `video2world_lora`, `NAME` is `2b_custom_data`.

```
checkpoints/posttraining/video2world_lora/2b_custom_data/checkpoints/
├── model/
│   ├── iter_{NUMBER}.pt  # Contains base model + LoRA parameters
├── optim/
├── scheduler/
├── trainer/
├── latest_checkpoint.txt
```

### 4. Run Inference on LoRA Post-trained Checkpoints

LoRA post-trained checkpoints require special handling during inference. Use the dedicated `video2world_lora.py` script:

#### Cosmos-Predict2-2B-Video2World with LoRA

For example, if a LoRA post-trained checkpoint with 1000 iterations is to be used, run the following command:

```bash
export NUM_GPUS=8
export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=${NUM_GPUS} examples/video2world_lora.py \
    --model_size 2B \
    --dit_path "checkpoints/posttraining/video2world_lora/2b_custom_data/checkpoints/model/iter_000001000.pt" \
    --input_path "assets/video2world/input0.jpg" \
    --prompt "A descriptive prompt for physical AI adapted to your custom domain." \
    --save_path output/lora_custom_data/generated_video_from_lora_post-training.mp4 \
    --num_gpus ${NUM_GPUS} \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2" \
    --offload_guardrail \
    --offload_prompt_refiner
```

#### Cosmos-Predict2-14B-Video2World with LoRA

The 14B model can be run similarly by changing the `--model_size` and using the appropriate LoRA parameters:

```bash
export NUM_GPUS=8
export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=${NUM_GPUS} examples/video2world_lora.py \
    --model_size 14B \
    --dit_path "checkpoints/posttraining/video2world_lora/14b_custom_data/checkpoints/model/iter_000001000.pt" \
    --input_path "assets/video2world/input0.jpg" \
    --prompt "A descriptive prompt for physical AI adapted to your custom domain." \
    --save_path output/lora_custom_data/generated_video_from_14b_lora_post-training.mp4 \
    --num_gpus ${NUM_GPUS} \
    --use_lora \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_target_modules "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2" \
    --offload_guardrail \
    --offload_prompt_refiner
```

#### Critical Requirements for LoRA Inference

**Important**: The LoRA parameters used during inference **must match** those used during training:

1. **Use the `--use_lora` flag**: This is required to enable LoRA inference mode
2. **Match LoRA parameters**: `--lora_rank`, `--lora_alpha`, and `--lora_target_modules` must match training config
3. **Use the LoRA inference script**: Use `video2world_lora.py` instead of `video2world.py`

See [documentations/inference_video2world_lora.md](documentations/inference_video2world_lora.md) for detailed inference run instructions.

### 5. Advanced LoRA Training Configurations

#### Domain-Specific LoRA Training

For different domains, you can adjust LoRA parameters:

**Robotics/Physical AI Domain:**
```python
model=dict(
    config=dict(
        train_architecture="lora",
        lora_rank=24,
        lora_alpha=24,
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        # Domain-specific settings
        max_iter=3000,
        lr=2 ** (-9.5),
    )
)
```

**Surveillance/Security Domain:**
```python
model=dict(
    config=dict(
        train_architecture="lora",
        lora_rank=16,
        lora_alpha=16,
        lora_target_modules="q_proj,k_proj,v_proj,output_proj",  # Attention-only for subtle changes
        # Domain-specific settings
        max_iter=2000,
        lr=2 ** (-10.5),
    )
)
```

**Creative/Artistic Domain:**
```python
model=dict(
    config=dict(
        train_architecture="lora",
        lora_rank=32,
        lora_alpha=48,  # Higher alpha for stronger adaptation
        lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2,norm1,norm2",
        # Domain-specific settings
        max_iter=4000,
        lr=2 ** (-9),
    )
)
```

#### Multi-Dataset LoRA Training

For training on multiple datasets, create combined dataloaders:

```python
# Multiple datasets
metropolis_dataset = L(Dataset)(
    dataset_dir="datasets/metropolis_lora",
    num_frames=93,
    video_size=(704, 1280),
)

robotics_dataset = L(Dataset)(
    dataset_dir="datasets/robotics_lora",
    num_frames=93,
    video_size=(704, 1280),
)

# Combined dataloader
combined_dataloader = L(DataLoader)(
    dataset=L(ConcatDataset)([metropolis_dataset, robotics_dataset]),
    sampler=L(get_sampler)(dataset=L(ConcatDataset)([metropolis_dataset, robotics_dataset])),
    batch_size=2,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)
```

#### LoRA with Different Target Modules

**Attention-Only (Lightweight):**
```python
lora_target_modules="q_proj,k_proj,v_proj,output_proj"
```

**Attention + MLP (Recommended):**
```python
lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
```

**Comprehensive (Maximum Adaptation):**
```python
lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2,norm1,norm2"
```

### 6. Troubleshooting LoRA Training

#### Common Issues and Solutions

1. **LoRA Parameters Not Found**
   ```
   Error: No parameters found for LoRA adaptation
   ```
   **Solution**: Ensure `model.config.train_architecture=lora` is set in the training command.

2. **Memory Issues**
   ```
   CUDA out of memory
   ```
   **Solution**: Reduce batch size or increase context parallelism size.

3. **Slow Convergence**
   ```
   Loss not decreasing after many iterations
   ```
   **Solution**: Try higher learning rate (2^(-9) to 2^(-8)) or higher LoRA rank.

4. **Overfitting**
   ```
   Training loss decreases but validation loss increases
   ```
   **Solution**: Reduce LoRA rank, add data augmentation, or early stopping.
