# Predict2 Video2World Post-Training Guide

This guide provides instructions on running post-training with Cosmos-Predict2 Video2World models.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Overview](#overview)
- [Post-training Guide](#post-training-guide)

## Prerequisites

Before running training:

1. **Environment setup**: Follow the [Setup guide](setup.md) for installation instructions.
2. **Model checkpoints**: Download required model weights following the [Downloading Checkpoints](setup.md#downloading-checkpoints) section in the Setup guide.
3. **Hardware considerations**: Review the [Performance guide](performance.md) for GPU requirements and model selection recommendations.

## Overview

Cosmos-Predict2 provides two models for generating videos from a combination of text and visual inputs: `Cosmos-Predict2-2B-Video2World` and `Cosmos-Predict2-14B-Video2World`. These models can transform a still image or video clip into a longer, animated sequence guided by the text description.

We support post-training the models with example datasets.
- [post-training_video2world_cosmos_nemo_assets](/documentations/post-training_video2world_cosmos_nemo_assets.md)
  - Basic examples with a small 4 videos dataset
- [post-training_video2world_agibot_fisheye](/documentations/post-training_video2world_agibot_fisheye.md)
  - Examples with fisheye-view dataset
  - Post-training from checkpoints with resolution & fps choices
- [post-training_video2world_gr00t](/documentations/post-training_video2world_gr00t.md)
  - Examples with GR00T-dreams datasets
## Post-training Guide

### 1. Preparing Data

The post-training data is expected to contain paired prompt and video files.
For example, a custom dataset can be saved in a following structure.

Dataset folder format:
```
datasets/custom_video2world_dataset/
├── metas/
│   ├── *.txt
├── videos/
│   ├── *.mp4
```

`metas` folder contains `.txt` files containing prompts describing the video content.
`videow` folder contains the corresponding `.mp4` video files.

After preparing `metas` and `videos` folders, run the following command to pre-compute T5-XXL embeddings.
```bash
python -m scripts.get_t5_embeddings --dataset_path datasets/custom_video2world_dataset/
```
This script will create `t5_xxl` folder under the dataset root where the T5-XXL embeddings are saved as `.pickle` files.
```
datasets/custom_video2world_dataset/
├── metas/
│   ├── *.txt
├── videos/
│   ├── *.mp4
├── t5_xxl/
│   ├── *.pickle
```

### 2. Creating Configs for Training

Define dataloader from the prepared dataset.

For example,
```python
# custom dataset example
example_video_dataset = L(Dataset)(
    dataset_dir="datasets/custom_video2world_dataset",
    num_frames=93,
    video_size=(704, 1280),  # 720 resolution, 16:9 aspect ratio
)

dataloader_video_train = L(DataLoader)(
    dataset=example_video_dataset,
    sampler=L(get_sampler)(dataset=example_video_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)
```

With the `dataloader_video_train`, create a config for a training job.
Here's a post-training example for video2world 2B model.
```python
predict2_video2world_training_2b_custom_data = dict(
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
        group="video2world",
        name="2b_custom_data",
    ),
    model=dict(
        config=dict(
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
    dataloader_train=dataloader_video_train,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=1000,                      # maximum number of iterations
    ),
    checkpoint=dict(
        save_iter=500,                      # checkpoints will be saved every 500 iterations.
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
    ),
    scheduler=dict(
        warm_up_steps=[0],
        cycle_lengths=[1_000],              # adjust considering max_iter
        f_max=[0.6],
        f_min=[0.0],
    ),
)
```

The config should be registered to ConfigStore.
```python
for _item in [
    # 2b, custom data
    predict2_video2world_training_2b_custom_data,
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

### 2.1. Config System

In the above config example, it starts by overriding from the registered configs.
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

#### Optimizer Configurations (`defaults/optimizer.py`)
- `fusedadamw`: FusedAdamW optimizer with standard settings
- Custom optimizer configurations for different training scenarios

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


In addition to the overrided values, the rest of the config setup overwrites or addes the other config details.

### 3. Run a Training Job.

Run the following command to execute an example post-training job with the custom data.
```bash
EXP=predict2_video2world_training_2b_custom_data
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

The above command will train the entire model. If you are interested in training with LoRA, attach `model.config.train_architecture=lora` to the training command.

The checkpoints will be saved to `checkpoints/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `video2world`, `NAME` is `2b_custom_data`.

```
checkpoints/posttraining/video2world/2b_custom_data/checkpoints/
├── model/
│   ├── iter_{NUMBER}.pt
├── optim/
├── scheduler/
├── trainer/
├── latest_checkpoint.txt
```

### 4. Run Inference on Post-trained Checkpoints

##### Cosmos-Predict2-2B-Video2World

For example, if a posttrained checkpoint with 1000 iterations is to be used, run the following command.
Use `--dit_path` argument to specify the path to the post-trained checkpoint.

```bash
python examples/video2world.py \
  --model_size 2B \
  --dit_path "checkpoints/posttraining/video2world/2b_custom_data/checkpoints/model/iter_000001000.pt" \
  --prompt "A descriptive prompt for physical AI." \
  --input_path "assets/video2world_cosmos_nemo_assets/output_Digit_Lift_movie.jpg" \
  --save_path output/cosmos_nemo_assets/generated_video_from_post-training.mp4
```

To load EMA weights from the post-trained checkpoint, add argument `--load_ema`.
```bash
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python examples/video2world.py \
  --model_size 2B \
  --dit_path "checkpoints/posttraining/video2world/2b_custom_data/checkpoints/model/iter_000001000.pt" \
  --load_ema \
  --prompt "A descriptive prompt for physical AI." \
  --input_path "assets/video2world_cosmos_nemo_assets/output_Digit_Lift_movie.jpg" \
  --save_path output/cosmos_nemo_assets/generated_video_from_post-training.mp4
```

See [documentations/inference_video2world.md](documentations/inference_video2world.md) for inference run details.

##### Cosmos-Predict2-14B-Video2World

The 14B model can be run similarly by changing the `--model_size` and `--dit_path` arguments.
