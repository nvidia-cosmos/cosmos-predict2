# Multiview Post-training for Waymo

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
### 1.1 Downloading & Pre-Processing Waymo Dataset

You can use [Waymo Open Dataset](https://waymo.com/open/) for post-training.

Please follow the [instruction](https://github.com/nv-tlabs/cosmos-av-sample-toolkits?tab=readme-ov-file#convert-public-datasets) in [cosmos-av-sample-toolkits](https://github.com/nv-tlabs/cosmos-av-sample-toolkits) to download and convert the Waymo Open Dataset.

```bash
mkdir -p datasets/waymo/

# Run this command in cosmos-av-sample-toolkits to process the Waymo videos
python convert_waymo_to_rds_hq.py -i <WAYMO_TFRECORDS_FOLDER> -o datasets/waymo/videos -n 32
```

#### 2. Preprocessing the Data

Run the following command to pre-compute T5-XXL embeddings for the video captions used for post-training:

```bash
# The script will use the provided prompt, save the T5-XXL embeddings in pickle format.
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/get_t5_embeddings_from_waymo.py --dataset_path datasets/waymo --prompt "A video of car driving on the road."
```

Dataset folder format:
```
datasets/waymo/
├── cache/
│   ├── prefix_t5_embeddings_pinhole_front.pickle
│   ├── prefix_t5_embeddings_pinhole_front_left.pickle
│   ├── prefix_t5_embeddings_pinhole_front_right.pickle
│   ├── prefix_t5_embeddings_pinhole_side_left.pickle
│   └── prefix_t5_embeddings_pinhole_side_right.pickle
├── metas/
│   ├── pinhole_front
│       ├── *.txt
│   ├── pinhole_front_left
│   ├── pinhole_front_right
│   ├── pinhole_side_left
│   └── pinhole_side_right
├── videos/
│   ├── pinhole_front
│       ├── *.mp4
│   ├── pinhole_front_left
│   ├── pinhole_front_right
│   ├── pinhole_side_left
│   └── pinhole_side_right
├── t5_xxl/
│   ├── pinhole_front
│       ├── *.pickle
│   ├── pinhole_front_left
│   ├── pinhole_front_right
│   ├── pinhole_side_left
│   └── pinhole_side_right
```

## 2. Post-training
### 2.1. Post-training on Waymo dataset
#### Cosmos-Predict2-2B-Multiview

Run the following command to execute an example post-training job with Waymo data
```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_multiview_training_2b_720p_10fps_7views_29frames_waymo5views
```

The model will be post-trained using the Waymo dataset. See the config `predict2_multiview_training_2b_720p_10fps_7views_29frames_waymo5views` defined in `cosmos_predict2/configs/base/experiment/waymo.py` to understand how the dataloader is defined.
```python
example_video_dataset_waymo5views_720p_train = L(MultiviewDataset)(
    dataset_dir="datasets/waymo",
    state_t=8,
    num_frames=29,
    sequence_interval=1,
    camera_keys=camera_keys,
    video_size=(704, 1280),
    front_camera_key="pinhole_front",
    camera_to_view_id=camera_to_view_id,
    front_view_caption_only=True,
    is_train=True,
)
```

The checkpoints will be saved to `checkpoints/PROJECT/GROUP/NAME`.
In the above example, `PROJECT` is `posttraining`, `GROUP` is `multiview`, `NAME` is `2b_720p_10fps_7views_29frames_waymo5views"`.

See the job config to understand how they are determined.
```python
predict2_multiview_training_2b_720p_10fps_7views_29frames_waymo5views = dict(
    dict(
        ...
        job=dict(
            project="posttraining",
            group="multiview",
            name="2b_720p_10fps_7views_29frames_waymo5views",
        ),
        ...
    )
)
```

The checkpoints will be saved in the below structure.
```
checkpoints/posttraining/multiview/2b_720p_10fps_7views_29frames_waymo5views/checkpoints/
├── model/
│   ├── iter_{NUMBER}.pt
├── optim/
├── scheduler/
├── trainer/
├── latest_checkpoint.txt
```

##### LoRA Post-Training

Post-training can be done using LoRA. Run the following command to execute an example LoRA multiview post-training job
```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_multiview_lora_training_2b_720p_10fps_7views_29frames_waymo5views

```


### 2.2 Post-training performance

The following table shows the expected iteration speed for 2B Multiview Training variations on 8 H100 GPUs with Context-Parallelism of size 8. Note that the 480p configurations can also run on as few as 2 H100 GPUs with Context-Parallelism of size 2.

| GPU Hardware    | 2B-Multiview-720p-7views-29frames | 2B-Multiview-720p-5views-29frames | 2B-Multiview-480p-5views-29frames |
|-----------------|-----------------------------------|-----------------------------------|-----------------------------------|
| NVIDIA B200     | -                                 | -                                 |-                                  |
| NVIDIA H100 NVL | ~8.90s sec                        | ~6.5 sec                          | ~3.5 sec                            |
| NVIDIA A100     | -                                 | -                                 |-                                  |

## 3. Inference with the Post-trained checkpoint
### 3.1 Inference
##### Cosmos-Predict2-2B-Multiview

For example, if a posttrained checkpoint with 1000 iterations is to be used, run the following command.
Use `--dit_path` argument to specify the path to the post-trained checkpoint.

```bash
export NUM_GPUS=8
torchrun --nproc_per_node=${NUM_GPUS} examples/multiview.py  \
  --model_size 2B  \
  --num_gpus $NUM_GPUS  \
  --dit_path "checkpoints/posttraining/multiview/2b_720p_10fps_7views_29frames_waymo5views/checkpoints/model/iter_000001000.pt" \
  --input_path  assets/multiview/sample1/video.mp4  \
  --prompt assets/multiview/sample1/caption.txt \
  --num_conditional_frames 1 \
  --fps 10    \
  --n_views 7  \
  --guidance 7.0 \
  --disable_prompt_refiner \
  --save_path output/multiview_postrain_2b_sample1_cond1.mp4

```

To load EMA weights from the post-trained checkpoint, add argument `--load_ema`.
```bash
export NUM_GPUS=8
torchrun --nproc_per_node=${NUM_GPUS} examples/multiview.py  \
  --model_size 2B  \
  --num_gpus $NUM_GPUS  \
  --dit_path "checkpoints/posttraining/multiview/2b_720p_10fps_7views_29frames_waymo5views//checkpoints/model/iter_000001000.pt" \
  --input_path  assets/multiview/sample1/video.mp4  \
  --prompt assets/multiview/sample1/caption.txt \\
  --num_conditional_frames 1 \
  --fps 10    \
  --n_views 7  \
  --guidance 7.0 \
  --disable_prompt_refiner \
  --save_path output/multiview_postrain_2b_sample1_cond1.mp4
```

See [documentations/inference_multiview.md](documentations/inference_multiview.md) for inference run details.