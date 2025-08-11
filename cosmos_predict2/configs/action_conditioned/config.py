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

import dataclasses

from cosmos_predict2.conditioner import ActionConditioner, BooleanFlag, ReMapkey, TextAttr
from cosmos_predict2.configs.base.config_video2world import (
    ConditioningStrategy,
    CosmosGuardrailConfig,
    CosmosReason1Config,
    Video2WorldPipelineConfig,
)
from cosmos_predict2.configs.base.defaults.ema import EMAConfig
from cosmos_predict2.models.text2image_dit import SACConfig
from cosmos_predict2.models.video2world_action_dit import ActionConditionedMinimalV1LVGDiT
from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface
from imaginaire.constants import (
    CHECKPOINTS_DIR,
    COSMOS_REASON1_MODEL_DIR,
    CosmosPredict2ActionConditionedFPS,
    CosmosPredict2ActionConditionedModelSize,
    CosmosPredict2ActionConditionedResolution,
    get_cosmos_predict2_video2world_tokenizer,
)
from imaginaire.lazy_config import LazyCall as L

# Cosmos Predict2 Video2World 2B
_PREDICT2_ACTION_CONDITIONED_NET_2B = L(ActionConditionedMinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    # attention settings
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    atten_backend="minimal_a2a",
    # positional embedding settings
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
    sac_config=L(SACConfig)(
        every_n_blocks=1,
        mode="predict2_2b_720",
    ),
    # NOTE: add action dimension
    action_dim=7 * 12,
)

_PREDICT2_ACTION_CONDITIONED_PIPELINE_2B = Video2WorldPipelineConfig(
    adjust_video_noise=True,
    conditioner=L(ActionConditioner)(
        fps=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="fps",
            output_key="fps",
        ),
        padding_mask=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="padding_mask",
            output_key="padding_mask",
        ),
        text=L(TextAttr)(
            dropout_rate=0.2,
            input_key=["t5_text_embeddings"],
        ),
        use_video_condition=L(BooleanFlag)(
            dropout_rate=0.0,
            input_key="fps",
            output_key="use_video_condition",
        ),
        # NOTE: add additional action as condition
        action=L(ReMapkey)(
            input_key="action",
            output_key="action",
            dropout_rate=0.0,
            dtype=None,
        ),
    ),
    conditioning_strategy=str(ConditioningStrategy.FRAME_REPLACE),
    min_num_conditional_frames=1,
    max_num_conditional_frames=1,
    net=_PREDICT2_ACTION_CONDITIONED_NET_2B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    rectified_flow_loss_weight_uniform=True,
    resize_online=True,
    resolution="720",
    ema=L(EMAConfig)(enabled=False),  # defaults to inference
    sigma_conditional=0.0001,
    sigma_data=1.0,
    state_ch=16,
    state_t=4,
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        load_mean_std=False,
        name="tokenizer",
        vae_pth=get_cosmos_predict2_video2world_tokenizer(model_size="2B"),
    ),
    # disable prompt refiner and guardrail for action conditional
    prompt_refiner_config=CosmosReason1Config(
        checkpoint_dir=COSMOS_REASON1_MODEL_DIR,
        offload_model_to_cpu=True,
        enabled=False,
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir=CHECKPOINTS_DIR,
        offload_model_to_cpu=True,
        enabled=False,
    ),
)


@dataclasses.dataclass(frozen=True)
class _ActionConditionedPipelineConfig:
    model_size: CosmosPredict2ActionConditionedModelSize


_PREDICT2_ACTION_CONDITIONED_PIPELINES: dict[_ActionConditionedPipelineConfig, Video2WorldPipelineConfig] = {
    _ActionConditionedPipelineConfig("2B"): _PREDICT2_ACTION_CONDITIONED_PIPELINE_2B,
}


def get_cosmos_predict2_action_conditioned_pipeline(
    *,
    model_size: CosmosPredict2ActionConditionedModelSize,
    resolution: CosmosPredict2ActionConditionedResolution,
    fps: CosmosPredict2ActionConditionedFPS,
) -> Video2WorldPipelineConfig:
    key = _ActionConditionedPipelineConfig(model_size)
    return _PREDICT2_ACTION_CONDITIONED_PIPELINES[key]
