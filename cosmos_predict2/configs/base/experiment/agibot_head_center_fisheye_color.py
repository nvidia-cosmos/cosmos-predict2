# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_video import Dataset
from imaginaire.lazy_config import LazyCall as L


def get_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()

# agibot_head_center_fisheye_color example
example_video_dataset_agibot_head_center_fisheye_color_train = L(Dataset)(
    dataset_dir="datasets/agibot_head_center_fisheye_color/train",
    num_frames=93,
    video_size=(704, 1280),
)
example_video_dataset_agibot_head_center_fisheye_color_val = L(Dataset)(
    dataset_dir="datasets/agibot_head_center_fisheye_color/val",
    num_frames=93,
    video_size=(704, 1280),
)

dataloader_train_agibot_head_center_fisheye_color = L(DataLoader)(
    dataset=example_video_dataset_agibot_head_center_fisheye_color_train,
    sampler=L(get_sampler)(dataset=example_video_dataset_agibot_head_center_fisheye_color_train),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
dataloader_val_agibot_head_center_fisheye_color = L(DataLoader)(
    dataset=example_video_dataset_agibot_head_center_fisheye_color_val,
    sampler=L(get_sampler)(dataset=example_video_dataset_agibot_head_center_fisheye_color_val),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# agibot_head_center_fisheye_color example by the resolution
# dataset - 480p
example_video_dataset_agibot_head_center_fisheye_color_480p_train = L(Dataset)(
    dataset_dir="datasets/agibot_head_center_fisheye_color/train",
    num_frames=93,
    video_size=(432, 768),
)
example_video_dataset_agibot_head_center_fisheye_color_480p_val = L(Dataset)(
    dataset_dir="datasets/agibot_head_center_fisheye_color/val",
    num_frames=93,
    video_size=(432, 768),
)
# dataset - 720p
example_video_dataset_agibot_head_center_fisheye_color_720p_train = L(Dataset)(
    dataset_dir="datasets/agibot_head_center_fisheye_color/train",
    num_frames=93,
    video_size=(704, 1280),
)
example_video_dataset_agibot_head_center_fisheye_color_720p_val = L(Dataset)(
    dataset_dir="datasets/agibot_head_center_fisheye_color/val",
    num_frames=93,
    video_size=(704, 1280),
)

# dataloader - 480p
dataloader_train_agibot_head_center_fisheye_color_480p = L(DataLoader)(
    dataset=example_video_dataset_agibot_head_center_fisheye_color_480p_train,
    sampler=L(get_sampler)(dataset=example_video_dataset_agibot_head_center_fisheye_color_480p_train),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
dataloader_val_agibot_head_center_fisheye_color_480p = L(DataLoader)(
    dataset=example_video_dataset_agibot_head_center_fisheye_color_480p_val,
    sampler=L(get_sampler)(dataset=example_video_dataset_agibot_head_center_fisheye_color_480p_val),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
# dataloader - 720p
dataloader_train_agibot_head_center_fisheye_color_720p = L(DataLoader)(
    dataset=example_video_dataset_agibot_head_center_fisheye_color_720p_train,
    sampler=L(get_sampler)(dataset=example_video_dataset_agibot_head_center_fisheye_color_720p_train),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
dataloader_val_agibot_head_center_fisheye_color_720p = L(DataLoader)(
    dataset=example_video_dataset_agibot_head_center_fisheye_color_720p_val,
    sampler=L(get_sampler)(dataset=example_video_dataset_agibot_head_center_fisheye_color_720p_val),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)

# 2B model training configs
# default config, 720p 16fps
# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_agibot_head_center_fisheye_color
predict2_video2world_training_2b_agibot_head_center_fisheye_color = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="2b_agibot_head_center_fisheye_color",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
                max_num_conditional_frames=1,
                min_num_conditional_frames=1,
                net=dict(
                    rope_h_extrapolation_ratio=2.0,
                    rope_t_extrapolation_ratio=1.0,
                    rope_w_extrapolation_ratio=2.0,
                ),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=2,
    ),
    dataloader_train=dataloader_train_agibot_head_center_fisheye_color,
    dataloader_val=dataloader_val_agibot_head_center_fisheye_color,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=100000,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=2 ** (-15.5),
    ),
    scheduler=dict(
        warm_up_steps=[2_000],
        cycle_lengths=[400_000],
        f_max=[0.99],
        f_min=[0.0],
    ),
)

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_agibot_head_center_fisheye_color_480p_10fps
predict2_video2world_training_2b_agibot_head_center_fisheye_color_480p_10fps = dict(
    defaults=[
        "/experiment/predict2_video2world_training_2b_agibot_head_center_fisheye_color",
        {"override /model": "predict2_video2world_fsdp_2b_480p_10fps"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="2b_agibot_head_center_fisheye_color_480p_10fps",
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_agibot_head_center_fisheye_color_480p,
    dataloader_val=dataloader_val_agibot_head_center_fisheye_color_480p,
)

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_agibot_head_center_fisheye_color_480p_16fps
predict2_video2world_training_2b_agibot_head_center_fisheye_color_480p_16fps = dict(
    defaults=[
        "/experiment/predict2_video2world_training_2b_agibot_head_center_fisheye_color",
        {"override /model": "predict2_video2world_fsdp_2b_480p_16fps"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="2b_agibot_head_center_fisheye_color_480p_16fps",
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_agibot_head_center_fisheye_color_480p,
    dataloader_val=dataloader_val_agibot_head_center_fisheye_color_480p,
)

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_agibot_head_center_fisheye_color_720p_10fps
predict2_video2world_training_2b_agibot_head_center_fisheye_color_720p_10fps = dict(
    defaults=[
        "/experiment/predict2_video2world_training_2b_agibot_head_center_fisheye_color",
        {"override /model": "predict2_video2world_fsdp_2b_720p_10fps"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="2b_agibot_head_center_fisheye_color_720p_10fps",
    ),
    dataloader_train=dataloader_train_agibot_head_center_fisheye_color_720p,
    dataloader_val=dataloader_val_agibot_head_center_fisheye_color_720p,
)

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_agibot_head_center_fisheye_color_720p_16fps
predict2_video2world_training_2b_agibot_head_center_fisheye_color_720p_16fps = dict(
    defaults=[
        "/experiment/predict2_video2world_training_2b_agibot_head_center_fisheye_color",
        {"override /model": "predict2_video2world_fsdp_2b_720p_16fps"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="2b_agibot_head_center_fisheye_color_720p_16fps",
    ),
    dataloader_train=dataloader_train_agibot_head_center_fisheye_color_720p,
    dataloader_val=dataloader_val_agibot_head_center_fisheye_color_720p,
)

# 14B model training configs
# default config, 720p 16fps
# torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_14b_agibot_head_center_fisheye_color
predict2_video2world_training_14b_agibot_head_center_fisheye_color = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_14b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="14b_agibot_head_center_fisheye_color",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
                max_num_conditional_frames=1,
                min_num_conditional_frames=1,
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=8,
    ),
    dataloader_train=dataloader_train_agibot_head_center_fisheye_color,
    dataloader_val=dataloader_val_agibot_head_center_fisheye_color,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=100000,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=2 ** (-15.5),
        weight_decay=0.2,
    ),
    scheduler=dict(
        warm_up_steps=[2_000],
        cycle_lengths=[50_000],
        f_max=[0.2],
        f_min=[0.0],
    ),
)

# torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_14b_agibot_head_center_fisheye_color_480p_10fps
predict2_video2world_training_14b_agibot_head_center_fisheye_color_480p_10fps = dict(
    defaults=[
        "/experiment/predict2_video2world_training_14b_agibot_head_center_fisheye_color",
        {"override /model": "predict2_video2world_fsdp_14b_480p_10fps"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="14b_agibot_head_center_fisheye_color_480p_10fps",
    ),
    dataloader_train=dataloader_train_agibot_head_center_fisheye_color_480p,
    dataloader_val=dataloader_val_agibot_head_center_fisheye_color_480p,
)

# torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_14b_agibot_head_center_fisheye_color_480p_16fps
predict2_video2world_training_14b_agibot_head_center_fisheye_color_480p_16fps = dict(
    defaults=[
        "/experiment/predict2_video2world_training_14b_agibot_head_center_fisheye_color",
        {"override /model": "predict2_video2world_fsdp_14b_480p_16fps"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="14b_agibot_head_center_fisheye_color_480p_16fps",
    ),
    dataloader_train=dataloader_train_agibot_head_center_fisheye_color_480p,
    dataloader_val=dataloader_val_agibot_head_center_fisheye_color_480p,
)

# torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_14b_agibot_head_center_fisheye_color_720p_10fps
predict2_video2world_training_14b_agibot_head_center_fisheye_color_720p_10fps = dict(
    defaults=[
        "/experiment/predict2_video2world_training_14b_agibot_head_center_fisheye_color",
        {"override /model": "predict2_video2world_fsdp_14b_720p_10fps"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="14b_agibot_head_center_fisheye_color_720p_10fps",
    ),
    dataloader_train=dataloader_train_agibot_head_center_fisheye_color_720p,
    dataloader_val=dataloader_val_agibot_head_center_fisheye_color_720p,
)

# torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_14b_agibot_head_center_fisheye_color_720p_16fps
predict2_video2world_training_14b_agibot_head_center_fisheye_color_720p_16fps = dict(
    defaults=[
        "/experiment/predict2_video2world_training_14b_agibot_head_center_fisheye_color",
        {"override /model": "predict2_video2world_fsdp_14b_720p_16fps"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="14b_agibot_head_center_fisheye_color_720p_16fps",
    ),
    dataloader_train=dataloader_train_agibot_head_center_fisheye_color_720p,
    dataloader_val=dataloader_val_agibot_head_center_fisheye_color_720p,
)

for _item in [
    # 2b, agibot_head_center_fisheye_color
    predict2_video2world_training_2b_agibot_head_center_fisheye_color,  # 720p 16fps
    predict2_video2world_training_2b_agibot_head_center_fisheye_color_480p_10fps,
    predict2_video2world_training_2b_agibot_head_center_fisheye_color_480p_16fps,
    predict2_video2world_training_2b_agibot_head_center_fisheye_color_720p_10fps,
    predict2_video2world_training_2b_agibot_head_center_fisheye_color_720p_16fps,
    # 14b, agibot_head_center_fisheye_color
    predict2_video2world_training_14b_agibot_head_center_fisheye_color,  # 720p 16fps
    predict2_video2world_training_14b_agibot_head_center_fisheye_color_480p_10fps,
    predict2_video2world_training_14b_agibot_head_center_fisheye_color_480p_16fps,
    predict2_video2world_training_14b_agibot_head_center_fisheye_color_720p_10fps,
    predict2_video2world_training_14b_agibot_head_center_fisheye_color_720p_16fps,
]:
    # Get the experiment name from the global variable.
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
