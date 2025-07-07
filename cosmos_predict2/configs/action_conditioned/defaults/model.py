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

from cosmos_predict2.configs.action_conditioned.config import (
    PREDICT2_VIDEO2WORLD_PIPELINE_2B_ACTION_CONDITIONED,
)
from cosmos_predict2.models.video2world_action_model import Predict2Video2WorldActionConditionedModel
from cosmos_predict2.models.video2world_model import Predict2ModelManagerConfig, Predict2Video2WorldModelConfig
from imaginaire.lazy_config import LazyCall as L

PREDICT2_V2W_2B_ACTION_CONDITIONED_FSDP_CONFIG = dict(
    trainer=dict(
        distributed_parallelism="fsdp",
    ),
    model=L(Predict2Video2WorldActionConditionedModel)(
        config=Predict2Video2WorldModelConfig(
            pipe_config=PREDICT2_VIDEO2WORLD_PIPELINE_2B_ACTION_CONDITIONED,
            model_manager_config=L(Predict2ModelManagerConfig)(
                dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt",
                text_encoder_path="",  # Do not load text encoder for training.
            ),
            fsdp_shard_size=-1,
        ),
        _recursive_=False,
    ),
)


def register_model_action_conditioned() -> None:
    cs = ConfigStore.instance()
    # predict2 v2w 2b model
    cs.store(
        group="model",
        package="_global_",
        name="predict2_v2w_2b_action_conditioned_fsdp",
        node=PREDICT2_V2W_2B_ACTION_CONDITIONED_FSDP_CONFIG,
    )
