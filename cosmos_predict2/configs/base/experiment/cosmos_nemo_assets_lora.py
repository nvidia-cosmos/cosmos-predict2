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



# Cosmos-NeMo-Assets LoRA video2world example
example_video_dataset_cosmos_nemo_assets_lora = L(Dataset)(
    dataset_dir="datasets/cosmos_nemo_assets",
    num_frames=93,
    video_size=(704, 1280),
)

dataloader_train_cosmos_nemo_assets_lora = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets_lora,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets_lora),
    batch_size=2,  # Can use larger batch size due to LoRA memory efficiency
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_lora_training_2b_cosmos_nemo_assets model.config.train_architecture=lora
predict2_video2world_lora_training_2b_cosmos_nemo_assets = dict(
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
        name="2b_cosmos_nemo_assets",
    ),
    model=dict(
        config=dict(
            # Enable LoRA training
            train_architecture="lora",
            # LoRA configuration parameters for video2world
            lora_rank=16,
            lora_alpha=16,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
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
    dataloader_train=dataloader_train_cosmos_nemo_assets_lora,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=2000,  # LoRA typically needs more iterations but trains faster
    ),
    checkpoint=dict(
        save_iter=100,  # More frequent checkpoints for LoRA
    ),
    optimizer=dict(
        lr=2 ** (-10),  # Higher learning rate for LoRA
    ),
    scheduler=dict(
        warm_up_steps=[0],
        cycle_lengths=[2_000],
        f_max=[0.6],
        f_min=[0.0],
    ),
)

# torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_lora_training_14b_cosmos_nemo_assets model.config.train_architecture=lora
predict2_video2world_lora_training_14b_cosmos_nemo_assets = dict(
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
        name="14b_cosmos_nemo_assets",
    ),
    model=dict(
        config=dict(
            # Enable LoRA training
            train_architecture="lora",
            # LoRA configuration parameters for video2world 14B
            lora_rank=32,  # Higher rank for larger model
            lora_alpha=32,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
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
    dataloader_train=dataloader_train_cosmos_nemo_assets_lora,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
        ),
        max_iter=1500,  # Fewer iterations for larger model
    ),
    checkpoint=dict(
        save_iter=300,  # More frequent checkpoints
    ),
    optimizer=dict(
        lr=2 ** (-11),  # Lower learning rate for larger model
        weight_decay=0.2,
    ),
    scheduler=dict(
        warm_up_steps=[0],
        cycle_lengths=[1_500],
        f_max=[0.25],
        f_min=[0.0],
    ),
)

for _item in [
    # 2b, cosmos_nemo_assets video2world LoRA
    predict2_video2world_lora_training_2b_cosmos_nemo_assets,
    # 14b, cosmos_nemo_assets video2world LoRA
    predict2_video2world_lora_training_14b_cosmos_nemo_assets,
]:
    # Get the experiment name from the global variable.
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    ) 