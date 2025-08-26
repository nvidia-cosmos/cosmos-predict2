# Video2World Post-training for Cosmos-NeMo-Assets

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
### 1.1 Downloading Cosmos-NeMo-Assets

The first step is downloading a dataset with videos.

You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p. These videos should focus on the subject throughout the entire video so that each video chunk contains the subject.

You can use [nvidia/Cosmos-NeMo-Assets](https://huggingface.co/datasets/nvidia/Cosmos-NeMo-Assets) for post-training.

```bash
mkdir -p datasets/cosmos_nemo_assets/

# This command will download the videos for physical AI
huggingface-cli download nvidia/Cosmos-NeMo-Assets --repo-type dataset --local-dir datasets/cosmos_nemo_assets/ --include "*.mp4*"

mv datasets/cosmos_nemo_assets/nemo_diffusion_example_data datasets/cosmos_nemo_assets/videos
```

### 1.2 Preprocessing the Data

Cosmos-NeMo-Assets comes with a single caption for 4 long videos.
Run the following command to pre-compute T5-XXL embeddings for the video caption used for post-training:
```bash
# The script will use the provided prompt, save the T5-XXL embeddings in pickle format.
python scripts/get_t5_embeddings_from_cosmos_nemo_assets.py --dataset_path datasets/cosmos_nemo_assets --prompt "A video of sks teal robot."
```

Dataset folder format:
```
datasets/cosmos_nemo_assets/
├── metas/
│   ├── *.txt
├── videos/
│   ├── *.mp4
├── t5_xxl/
│   ├── *.pickle
```

## 2. Post-training
### 2.1. Post-training on Cosmos-NeMo-Assets dataset
#### Cosmos-Predict2-2B-Video2World

Run the following command to execute an example post-training job with `cosmos_nemo_assets` data.
```bash
EXP=predict2_video2world_training_2b_cosmos_nemo_assets
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

The model will be post-trained using the cosmos_nemo_assets dataset.
See the config `predict2_video2world_training_2b_cosmos_nemo_assets` defined in `cosmos_predict2/configs/base/experiment/cosmos_nemo_assets.py` to understand how the dataloader is defined.
```python
# Cosmos-NeMo-Assets example
example_video_dataset_cosmos_nemo_assets = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    num_frames=93,
    video_size=(704, 1280),
)

dataloader_train_cosmos_nemo_assets = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)
```

The checkpoints will be saved to `checkpoints/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `video2world`, `NAME` is `2b_cosmos_nemo_assets`.

See the job config to understand how they are determined.
```python
predict2_video2world_training_2b_cosmos_nemo_assets = dict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="video2world",
            name="2b_cosmos_nemo_assets",
        ),
        ...
    )
)
```

The checkpoints will be saved in the below structure.
```
checkpoints/posttraining/video2world/2b_cosmos_nemo_assets/checkpoints/
├── model/
│   ├── iter_{NUMBER}.pt
├── optim/
├── scheduler/
├── trainer/
├── latest_checkpoint.txt
```

#### Cosmos-Predict2-14B-Video2World

Run the following command to execute an example post-training job with `cosmos_nemo_assets` data with 4 nodes with 8 GPUs.
```bash
EXP=predict2_video2world_training_14b_cosmos_nemo_assets
torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 \
-m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

The above command will train the entire model. If you are interested in training with [LoRA](https://arxiv.org/abs/2106.09685), attach `model.config.train_architecture=lora` to the training command.

The checkpoints will be saved in the below structure.
```
checkpoints/posttraining/video2world/14b_cosmos_nemo_assets/checkpoints/
├── model/
│   ├── iter_{NUMBER}.pt
├── optim/
├── scheduler/
├── trainer/
├── latest_checkpoint.txt
```


## 3. Inference with the Post-trained checkpoint
### 3.1 Inference
##### Cosmos-Predict2-2B-Video2World

For example, if a posttrained checkpoint with 1000 iterations is to be used, run the following command.
Use `--dit_path` argument to specify the path to the post-trained checkpoint.

```bash
python examples/video2world.py \
  --model_size 2B \
  --dit_path "checkpoints/posttraining/video2world/2b_cosmos_nemo_assets/checkpoints/model/iter_000001000.pt" \
  --prompt "A video of sks teal robot." \
  --input_path "assets/video2world_cosmos_nemo_assets/output_Digit_Lift_movie.jpg" \
  --save_path output/cosmos_nemo_assets/generated_video_teal_robot.mp4
```

See [documentations/inference_video2world.md](./inference_video2world.md) for inference run details.

##### Cosmos-Predict2-14B-Video2World

The 14B model can be run similarly by changing the `--model_size` and `--dit_path` arguments.
