# Video2World Post-training for AgiBotWorld-Alpha

This guide provides instructions on running post-training with Cosmos-Predict2 Video2World models.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Preparing Data](#1-preparing-data)
- [Post-training](#2-post-training)
- [Inference with the Post-trained checkpoint](#3-inference-with-the-post-trained-checkpoint)

## Prerequisites

Before running training:

1. **Environment setup**: Follow the [Setup guide](setup.md) for installation instructions.
2. **Model checkpoints**: Download required model weights following the [Downloading Checkpoints](setup.md#downloading-checkpoints) section in the Setup guide.
3. **Hardware considerations**: Review the [Performance guide](performance.md) for GPU requirements and model selection recommendations.


## 1. Preparing Data
### 1.1 Downloading & Pre-Processing AgiBotWorld-Alpha dataset

We take a subset of [AgiBotWorld-Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) to provide an example post-training job.
1. Get a [Hugging Face](https://huggingface.co/settings/tokens) access token with `Read` permission
2. Login: `huggingface-cli login`
3. The [AgiBot World COMMUNITY LICENSE AGREEMENT](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) must be submitted. It is required before AgiBotWorld-Alpha can be downloaded.
4. Download task 327 from [AgiBotWorld-Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) dataset:
```bash
# Download, extract, and clean (default behavior)
python scripts/prepare_agibot_fisheye_data.py

# Clean existing data
python scripts/prepare_agibot_fisheye_data.py --delete-only

# Split videos into (5-second) windows
# In this example, we use task_id 327, episode_id 685393 as validation data
python scripts/prepare_agibot_fisheye_data.py --split-only --val_episode_ids 685393

# (Optional) Remove the source data
rm -rf datasets/agibot
```

Expect to use ~100 GB storage in the data preparation steps.
After the processing is done, there will be ~2 GB data remaining in `datasets/agibot_head_center_fisheye_color` folder.

Dataset folder format:
```
datasets/agibot_head_center_fisheye_color/
├── train/
│   ├── metas/
│   │   ├── *.txt
│   ├── videos/
│   │   ├── *.mp4
├── val/
│   ├── metas/
│   │   ├── *.txt
│   ├── videos/
│   │   ├── *.mp4
```


### 1.2 Preprocessing the Data

Run the following command to pre-compute T5-XXL embeddings for the video caption used for post-training:
```bash
# The script will use the provided prompt from the dataset, save the T5-XXL embeddings in pickle format.
python scripts/get_t5_embeddings.py --dataset_path datasets/agibot_head_center_fisheye_color/train
python scripts/get_t5_embeddings.py --dataset_path datasets/agibot_head_center_fisheye_color/val
```

Dataset folder format:
```
datasets/agibot_head_center_fisheye_color/
├── train/
│   ├── metas/
│   │   ├── *.txt
│   ├── videos/
│   │   ├── *.mp4
│   ├── t5_xxl/
│   │   ├── *.pickle
├── val/
│   ├── metas/
│   │   ├── *.txt
│   ├── videos/
│   │   ├── *.mp4
│   ├── t5_xxl/
│   │   ├── *.pickle
```

## 2. Post-training
### 2.1. Post-training on AgiBotWorld-Alpha dataset
#### Cosmos-Predict2-2B-Video2World

Run the following command to execute an example post-training job with `agibot_head_center_fisheye_color` data.
```bash
EXP=predict2_video2world_training_2b_agibot_head_center_fisheye_color
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

The model will be post-trained using the agibot_head_center_fisheye_color dataset.
See the config `predict2_video2world_training_2b_agibot_head_center_fisheye_color` defined in `cosmos_predict2/configs/base/experiment/agibot_head_center_fisheye_color.py` to understand how the dataloader is defined.
```python
# agibot_head_center_fisheye_color example
example_video_dataset_agibot_head_center_fisheye_color = L(Dataset)(
    dataset_dir="datasets/agibot_head_center_fisheye_color",
    num_frames=93,
    video_size=(704, 1280),
)

dataloader_train_agibot_head_center_fisheye_color = L(DataLoader)(
    dataset=example_video_dataset_agibot_head_center_fisheye_color,
    sampler=L(get_sampler)(dataset=example_video_dataset_agibot_head_center_fisheye_color),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)
```

The checkpoints will be saved to `checkpoints/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `video2world`, `NAME` is `2b_agibot_head_center_fisheye_color`.

See the job config to understand how they are determined.
```python
predict2_video2world_training_2b_agibot_head_center_fisheye_color = dict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="video2world",
            name="2b_agibot_head_center_fisheye_color",
        ),
        ...
    )
)
```

The checkpoints will be saved in the below structure.
```
checkpoints/posttraining/video2world/2b_agibot_head_center_fisheye_color/checkpoints/
├── model/
│   ├── iter_{NUMBER}.pt
├── optim/
├── scheduler/
├── trainer/
├── latest_checkpoint.txt
```

##### Resolution, FPS variants

Post-training can be done from the provided checkpoints with resolution - [480p, 720p] and fps - [10fps, 16fps] choices.
The corresponding config names are with `_{RESOLUTION}p_{FPS}fps` appended to the default config name.
The default config without those postfixes is with 720p 16fps.
```bash
# 480p, 10fps
EXP=predict2_video2world_training_2b_agibot_head_center_fisheye_color_480p_10fps
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}

# 480p, 16fps
EXP=predict2_video2world_training_2b_agibot_head_center_fisheye_color_480p_16fps
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}

# 720p, 10fps
EXP=predict2_video2world_training_2b_agibot_head_center_fisheye_color_720p_10fps
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}

# 720p, 16fps
EXP=predict2_video2world_training_2b_agibot_head_center_fisheye_color_720p_16fps
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

#### Cosmos-Predict2-14B-Video2World

Run the following command to execute an example post-training job with `agibot_head_center_fisheye_color` data with 4 nodes with 8 GPUs.
```bash
EXP=predict2_video2world_training_14b_agibot_head_center_fisheye_color
torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 \
-m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

The above command will train the entire model. If you are interested in training with [LoRA](https://arxiv.org/abs/2106.09685), attach `model.config.train_architecture=lora` to the training command.

The checkpoints will be saved in the below structure.
```
checkpoints/posttraining/video2world/14b_agibot_head_center_fisheye_color/checkpoints/
├── model/
│   ├── iter_{NUMBER}.pt
├── optim/
├── scheduler/
├── trainer/
├── latest_checkpoint.txt
```

##### Resolution, FPS variants

Like 2B models, post-training can be done from the provided checkpoints with resolution - [480p, 720p] and fps - [10fps, 16fps] choices.
The corresponding config names are with `_{RESOLUTION}p_{FPS}fps` appended to the default config name.
The default config without those postfixes is with 720p 16fps.
```bash
# 480p, 10fps
EXP=predict2_video2world_training_14b_agibot_head_center_fisheye_color_480p_10fps
EXP=predict2_video2world_training_14b_agibot_head_center_fisheye_color
torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 \
-m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}

# 480p, 16fps
EXP=predict2_video2world_training_14b_agibot_head_center_fisheye_color_480p_16fps
EXP=predict2_video2world_training_14b_agibot_head_center_fisheye_color
torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 \
-m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}

# 720p, 10fps
EXP=predict2_video2world_training_14b_agibot_head_center_fisheye_color_720p_10fps
EXP=predict2_video2world_training_14b_agibot_head_center_fisheye_color
torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 \
-m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}

# 720p, 16fps
EXP=predict2_video2world_training_14b_agibot_head_center_fisheye_color_720p_16fps
EXP=predict2_video2world_training_14b_agibot_head_center_fisheye_color
torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 \
-m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

### 2.2 Post-training performance

The following table shows the expected iteration speed for 2B and 14B Video2World model training on different GPUs.
Note that 2B model uses 8 GPUs, while 14B model uses 32 GPUs. 14B model also has 4x lower global batch size, as it uses Context-Parallelism of size 8, while 2B model uses Context-Parallelism of size 2.

| GPU Hardware    | 2B-Video2World | 14B-Video2World |
|-----------------|----------------|-----------------|
| NVIDIA B200     | 6.05 sec       | 6.27 sec        |
| NVIDIA H100 NVL | 10.07 sec      | 8.72 sec        |
| NVIDIA A100     | 22.5 sec       | 22.14 sec       |

## 3. Inference with the Post-trained checkpoint
### 3.1 Inference
##### Cosmos-Predict2-2B-Video2World

For example, if a posttrained checkpoint with 1000 iterations is to be used, run the following command.
Use `--dit_path` argument to specify the path to the post-trained checkpoint.

```bash
PROMPT_="The video captures a humanoid robot positioned in front of a fruit stand in a supermarket environment. The robot's right arm extends downward, reaching for a shiitake mushroom on the shelf. The arm carefully grasps the mushroom, lifting it towards the robot's body. The surrounding environment includes a shopping cart with a clear plastic bag and a red handle, as well as various fruits and vegetables displayed on the shelves. The robot's task is to retrieve items from the supermarket shelves, and this frame shows the initial step of picking up a shiitake mushroom."

python examples/video2world.py \
  --model_size 2B \
  --dit_path "checkpoints/posttraining/video2world/2b_agibot_head_center_fisheye_color/checkpoints/model/iter_000001000.pt" \
  --prompt "${PROMPT_}" \
  --input_path "datasets/agibot_head_center_fisheye_color/val/videos/task_327_episode_685393_window_0_frame_0-149.mp4" \
  --num_conditional_frames 1 \
  --save_path output/generated_video_2b_agibot_fisheye.mp4
```

To load EMA weights from the post-trained checkpoint, add argument `--load_ema`.
```bash
python examples/video2world.py \
  --model_size 2B \
  --dit_path "checkpoints/posttraining/video2world/2b_agibot_head_center_fisheye_color/checkpoints/model/iter_000001000.pt" \
  --prompt "${PROMPT_}" \
  --input_path "datasets/agibot_head_center_fisheye_color/val/videos/task_327_episode_685393_window_0_frame_0-149.mp4" \
  --num_conditional_frames 1 \
  --load_ema \
  --save_path output/generated_video_2b_agibot_fisheye_ema.mp4
```

See [documentations/inference_video2world.md](./inference_video2world.md) for inference run details.

##### Cosmos-Predict2-14B-Video2World

The 14B model can be run similarly by changing the `--model_size` and `--dit_path` arguments.
