# Predict2 Text2Image Post-Training Guide

This guide provides instructions on running post-training with Cosmos-Predict2 Text2Image models.

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

Cosmos-Predict2 provides two models for generating videos from a combination of text and visual inputs: `Cosmos-Predict2-2B-Text2Image` and `Cosmos-Predict2-14B-Text2Image`. These models can transform a still image or video clip into a longer, animated sequence guided by the text description.

We support post-training the models with example datasets.
- [post-training_text2image_cosmos_nemo_assets](/documentations/post-training_text2image_cosmos_nemo_assets.md)
  - Basic examples with a small 4 videos dataset

## Post-training Guide

### 1. Preparing Data

The post-training data is expected to contain paired prompt and video files.
For example, a custom dataset can be saved in a following structure.

Dataset folder format:
```
datasets/custom_text2image_dataset/
├── metas/
│   ├── *.txt
├── images/
│   ├── *.jpg
```

`metas` folder contains `.txt` files containing prompts describing the video content.
`videow` folder contains the corresponding `.mp4` video files.

After preparing `metas` and `images` folders, run the following command to pre-compute T5-XXL embeddings.
```bash
python -m scripts.get_t5_embeddings --dataset_path datasets/custom_text2image_dataset/
```
This script will create `t5_xxl` folder under the dataset root where the T5-XXL embeddings are saved as `.pickle` files.
```
datasets/custom_text2image_dataset/
├── metas/
│   ├── *.txt
├── images/
│   ├── *.jpg
├── t5_xxl/
│   ├── *.pickle
```

### 2. Creating Configs for Training

Define dataloader from the prepared dataset.

For example,
```python
# custom dataset example
example_image_dataset = L(Dataset)(
    dataset_dir="datasets/custom_text2image_dataset",
    image_size=(768, 1360),  # 1024 resolution, 16:9 aspect ratio
)

dataloader_image_train = L(DataLoader)(
    dataset=example_image_dataset,
    sampler=L(get_sampler)(dataset=example_image_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)
```

With the `dataloader_image_train`, create a config for a training job.
Here's a post-training example for text2image 2B model.
```python
predict2_text2image_training_2b_custom_data = dict(
    defaults=[
        {"override /model": "predict2_text2image_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock_image"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="text2image",
        name="2b_custom_data",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                ema=dict(enabled=True),     # enable EMA during training
                guardrail_config=dict(enabled=False),   # disable guardrail during training
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=1,            # context parallelism size
    ),
    dataloader_train=dataloader_image_train,
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
        weight_decay=0.2,
    ),
    scheduler=dict(
        warm_up_steps=[0],
        cycle_lengths=[1_000],              # adjust considering max_iter
        f_max=[0.4],
        f_min=[0.0],
    ),
)
```

The config should be registered to ConfigStore.
```python
for _item in [
    # 2b, custom data
    predict2_text2image_training_2b_custom_data,
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
    {"override /model": "predict2_text2image_fsdp_2b"},
    {"override /optimizer": "fusedadamw"},
    {"override /scheduler": "lambdalinear"},
    {"override /ckpt_type": "standard"},
    {"override /dataloader_val": "mock_image"},
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
    └── utils.py                # Utility functions for experiments
```


The system provides several pre-defined configuration groups that can be mixed and matched:

#### Model Configurations (`defaults/model.py`)
- `predict2_text2image_fsdp_2b`: 2B parameter Text2Image model with FSDP
- `predict2_text2image_fsdp_14b`: 14B parameter Text2Image model with FSDP

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
EXP=predict2_text2image_training_2b_custom_data
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

The above command will train the entire model. If you are interested in training with LoRA, attach `model.config.train_architecture=lora` to the training command.

The checkpoints will be saved to `checkpoints/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `text2image`, `NAME` is `2b_custom_data`.

```
checkpoints/posttraining/text2image/2b_custom_data/checkpoints/
├── model/
│   ├── iter_{NUMBER}.pt
├── optim/
├── scheduler/
├── trainer/
├── latest_checkpoint.txt
```

### 4. Run Inference on Post-trained Checkpoints

##### Cosmos-Predict2-2B-Text2Image

For example, if a posttrained checkpoint with 1000 iterations is to be used, run the following command.
Use `--dit_path` argument to specify the path to the post-trained checkpoint.

```bash
python examples/text2image.py \
  --model_size 2B \
  --dit_path "checkpoints/posttraining/text2image/2b_custom_data/checkpoints/model/iter_000001000.pt" \
  --prompt "A descriptive prompt for physical AI." \
  --save_path output/cosmos_nemo_assets/generated_image_from_post-training.mp4
```

To load EMA weights from the post-trained checkpoint, add argument `--load_ema`.
```bash
python examples/text2image.py \
  --model_size 2B \
  --dit_path "checkpoints/posttraining/text2image/2b_custom_data/checkpoints/model/iter_000001000.pt" \
  --load_ema \
  --prompt "A descriptive prompt for physical AI." \
  --save_path output/cosmos_nemo_assets/generated_image_from_post-training.mp4
```

See [documentations/inference_text2image.md](documentations/inference_text2image.md) for inference run details.

##### Cosmos-Predict2-14B-Text2Image

The 14B model can be run similarly by changing the `--model_size` and `--dit_path` arguments.
