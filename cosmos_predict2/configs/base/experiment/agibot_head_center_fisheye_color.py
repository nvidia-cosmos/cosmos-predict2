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
    video_size=(720, 1280),
)

example_video_dataset_agibot_head_center_fisheye_color_val = L(Dataset)(
    dataset_dir="datasets/agibot_head_center_fisheye_color/val",
    num_frames=93,
    video_size=(720, 1280),
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

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_agibot_head_center_fisheye_color
predict2_video2world_training_2b_agibot_head_center_fisheye_color = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="2b_agibot_head_center_fisheye_color",
    ),
    model=dict(
        config=dict(
            fsdp_shard_size=8,
            pipe_config=dict(
                ema=dict(enabled=True),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=2,
    ),
    dataloader_train=dataloader_train_agibot_head_center_fisheye_color,
    # dataloader_val=dataloader_val_agibot_head_center_fisheye_color,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=1000,
    ),
    checkpoint=dict(
        save_iter=200,
    ),
)

# torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_14b_agibot_head_center_fisheye_color
predict2_video2world_training_14b_agibot_head_center_fisheye_color = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_14b"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="14b_agibot_head_center_fisheye_color",
    ),
    model=dict(
        config=dict(
            fsdp_shard_size=8,
            pipe_config=dict(
                ema=dict(enabled=True),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=8,
    ),
    dataloader_train=dataloader_train_agibot_head_center_fisheye_color,
    # dataloader_val=dataloader_val_agibot_head_center_fisheye_color,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=1000,
    ),
    checkpoint=dict(
        save_iter=200,
    ),
)

for _item in [
    # 2b, agibot_head_center_fisheye_color
    predict2_video2world_training_2b_agibot_head_center_fisheye_color,
    # 14b, agibot_head_center_fisheye_color
    predict2_video2world_training_14b_agibot_head_center_fisheye_color,
]:
    # Get the experiment name from the global variable.
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
