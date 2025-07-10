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

from cosmos_predict2.data.dataset_image import ImageDataset
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

# Cosmos-NeMo-Assets text2image example
example_image_dataset_cosmos_nemo_assets_images = L(ImageDataset)(
    dataset_dir="datasets/cosmos_nemo_assets_images",
    image_size=(768, 1360),  # 1024 resolution, 16:9 aspect ratio
)

dataloader_train_cosmos_nemo_assets_images = L(DataLoader)(
    dataset=example_image_dataset_cosmos_nemo_assets_images,
    sampler=L(get_sampler)(dataset=example_image_dataset_cosmos_nemo_assets_images),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

# torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_text2image_training_2b_cosmos_nemo_assets
predict2_text2image_training_2b_cosmos_nemo_assets = dict(
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
        name="2b_cosmos_nemo_assets",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                ema=dict(enabled=True),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets_images,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=1000,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=2 ** (-15),
    ),
    scheduler=dict(
        warm_up_steps=[0],
        cycle_lengths=[1_000],
        f_max=[0.17],
        f_min=[0.0],
    ),
)

# torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_text2image_training_14b_cosmos_nemo_assets
predict2_text2image_training_14b_cosmos_nemo_assets = dict(
    defaults=[
        {"override /model": "predict2_text2image_fsdp_14b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mock_image"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="text2image",
        name="14b_cosmos_nemo_assets",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                ema=dict(enabled=True),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=1,
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets_images,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=1000,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.2,
    ),
    scheduler=dict(
        warm_up_steps=[0],
        cycle_lengths=[1_000],
        f_max=[0.4],
        f_min=[0.0],
    ),
)

# Cosmos-NeMo-Assets video2world example
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

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_cosmos_nemo_assets
predict2_video2world_training_2b_cosmos_nemo_assets = dict(
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
        name="2b_cosmos_nemo_assets",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=2,
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=1000,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
    ),
    scheduler=dict(
        warm_up_steps=[0],
        cycle_lengths=[1_000],
        f_max=[0.6],
        f_min=[0.0],
    ),
)

# torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_14b_cosmos_nemo_assets
predict2_video2world_training_14b_cosmos_nemo_assets = dict(
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
        group="video2world",
        name="14b_cosmos_nemo_assets",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=8,
    ),
    dataloader_train=dataloader_train_cosmos_nemo_assets,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=1000,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.2,
    ),
    scheduler=dict(
        warm_up_steps=[0],
        cycle_lengths=[1_000],
        f_max=[0.25],
        f_min=[0.0],
    ),
)

for _item in [
    # 2b, cosmos_nemo_assets
    predict2_text2image_training_2b_cosmos_nemo_assets,
    # 14b, cosmos_nemo_assets
    predict2_text2image_training_14b_cosmos_nemo_assets,
    # 2b, cosmos_nemo_assets
    predict2_video2world_training_2b_cosmos_nemo_assets,
    # 14b, cosmos_nemo_assets
    predict2_video2world_training_14b_cosmos_nemo_assets,
]:
    # Get the experiment name from the global variable.
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
