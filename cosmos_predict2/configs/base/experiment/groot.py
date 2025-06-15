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


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()

# GROOT example
example_video_dataset_gr1 = L(Dataset)(
    dataset_dir="datasets/benchmark_train/gr1",
    num_frames=93,
    video_size=(432, 768),
)

dataloader_train_gr1 = L(DataLoader)(
    dataset=example_video_dataset_gr1,
    sampler=L(get_sampler)(dataset=example_video_dataset_gr1),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)


# NVTE_FUSED_ATTN=0 torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_groot_gr1_480
predict2_video2world_training_2b_groot_gr1_480 = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /data_val": "mock"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=8,
            pipe_config=dict(guardrail_config=dict(enabled=False)),
        )
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
    ),
    scheduler=dict(
        f_max=[0.2],
        f_min=[0.1],
        warm_up_steps=[1_000],
        cycle_lengths=[100_000],
    ),
    job=dict(
        project="posttraining",
        group="video2world",
        name="2b_groot_gr1_480",
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_gr1,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=200),
        ),
    ),
    checkpoint=dict(
        save_iter=200,
    ),
)

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_14b_groot_gr1_480
predict2_video2world_training_14b_groot_gr1_480 = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_14b"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /data_val": "mock"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=dict(
        config=dict(
            fsdp_shard_size=32,
            pipe_config=dict(guardrail_config=dict(enabled=False)),
        )
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
    ),
    scheduler=dict(
        f_max=[0.2],
        f_min=[0.1],
        warm_up_steps=[1_000],
        cycle_lengths=[100_000],
    ),
    model_parallel=dict(
        context_parallel_size=4,
    ),
    job=dict(
        project="posttraining",
        group="video2world",
        name="14b_groot_gr1_480",
    ),
    dataloader_train=dataloader_train_gr1,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=200),
        ),
    ),
    checkpoint=dict(
        save_iter=200,
    ),
)


for _item in [
    # 2b, gr1
    predict2_video2world_training_2b_groot_gr1_480,
    # 14b, gr1
    predict2_video2world_training_14b_groot_gr1_480,
]:
    # Get the experiment name from the global variable, e.g. exp01_wan_lora -> experiment_name = "exp01_wan_lora"
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
