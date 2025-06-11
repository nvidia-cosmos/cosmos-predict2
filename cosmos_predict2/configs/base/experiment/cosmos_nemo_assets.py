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


# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_cosmos_nemo_assets
predict2_video2world_training_2b_cosmos_nemo_assets = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /data_val": "mock"},
        {"override /data_train": "cosmos_nemo_assets"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="2b_cosmos_nemo_assets",
    ),
    model=dict(
        config=dict(
            num_video_frames=77,
            resolution="720",
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

# torchrun --nproc_per_node=8 --nnodes=4 --rdzv_id 123 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:1234 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_14b_cosmos_nemo_assets
predict2_video2world_training_14b_cosmos_nemo_assets = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_14b"},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /data_val": "mock"},
        {"override /data_train": "cosmos_nemo_assets"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world",
        name="14b_cosmos_nemo_assets",
    ),
    model=dict(
        config=dict(
            num_video_frames=77,
            resolution="720",
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
    # 2b, cosmos_nemo_assets
    predict2_video2world_training_2b_cosmos_nemo_assets,
    # 14b, cosmos_nemo_assets
    predict2_video2world_training_14b_cosmos_nemo_assets,
]:
    # Get the experiment name from the global variable.
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]
    
    cs = ConfigStore.instance()
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
