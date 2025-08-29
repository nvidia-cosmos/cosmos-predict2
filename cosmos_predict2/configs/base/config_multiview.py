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
from copy import deepcopy

import attrs

from cosmos_predict2.conditioner import BooleanFlag, ConditionLocation, MultiViewConditioner, ReMapkey, TextAttr
from cosmos_predict2.configs.base.config_video2world import (
    ConditioningStrategy,
    CosmosGuardrailConfig,
    CosmosReason1Config,
    EMAConfig,
    SolverTimestampConfig,
    TokenizerInterface,
)
from cosmos_predict2.models.multiview_dit import MultiViewDiT
from cosmos_predict2.models.text2image_dit import SACConfig
from imaginaire.auxiliary.text_encoder import (
    CosmosTextEncoderConfig,
)
from imaginaire.config import LazyDict, make_freezable
from imaginaire.constants import (
    CHECKPOINTS_DIR,
    COSMOS_REASON1_MODEL_DIR,
    CosmosPredict2MultiviewFPS,
    CosmosPredict2MultiviewFrames,
    CosmosPredict2MultiviewResolution,
    CosmosPredict2MultiviewViews,
    CosmosPredict2Video2WorldModelSize,
    get_cosmos_predict2_video2world_tokenizer,
)
from imaginaire.lazy_config import LazyCall as L


@make_freezable
@attrs.define(slots=False)
class MultiviewPipelineConfig:
    adjust_video_noise: bool
    conditioner: LazyDict[MultiViewConditioner]
    conditioning_strategy: str
    min_num_conditional_frames_per_view: int
    max_num_conditional_frames_per_view: int
    condition_locations: list[ConditionLocation]
    # concat_view_embedding: bool
    # view_condition_dim: int
    # n_cameras_emb: int
    sigma_conditional: float
    net: LazyDict[MultiViewDiT]
    tokenizer: LazyDict[TokenizerInterface]
    prompt_refiner_config: CosmosReason1Config
    guardrail_config: CosmosGuardrailConfig
    precision: str
    rectified_flow_t_scaling_factor: float
    resize_online: bool
    resolution: str
    ema: EMAConfig
    sigma_data: float = 1.0
    state_ch: int = 16
    state_t: int = 24
    text_encoder: CosmosTextEncoderConfig = attrs.field(factory=CosmosTextEncoderConfig)
    input_video_key: str = "video"
    input_image_key: str = "images"
    timestamps: SolverTimestampConfig = L(SolverTimestampConfig)(  # noqa: RUF009
        nfe=35,
        t_min=0.01,
        t_max=200.0,
        order=7.0,
        is_forward=False,
    )


_PREDICT2_MULTIVIEW_NET_2B_10FPS_7VIEWS_29FRAMES = L(MultiViewDiT)(
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
    pos_emb_learnable=False,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=3.0,
    rope_w_extrapolation_ratio=3.0,
    rope_t_extrapolation_ratio=8.0 / 24.0,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
    sac_config=L(SACConfig)(
        every_n_blocks=1,
        mode="predict2_2b_720",
    ),
    state_t=8,
    n_cameras_emb=7,
    view_condition_dim=7,
    concat_view_embedding=True,
)

_PREDICT2_MULTIVIEW_PIPELINE_2B_10FPS_7VIEWS_29FRAMES = MultiviewPipelineConfig(
    adjust_video_noise=False,
    conditioner=L(MultiViewConditioner)(
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
            dropout_rate=0.0,
            input_key=["t5_text_embeddings"],
        ),
        use_video_condition=L(BooleanFlag)(
            dropout_rate=0.0,
            input_key="fps",
            output_key="use_video_condition",
        ),
        view_indices_B_T=L(ReMapkey)(
            input_key="latent_view_indices_B_T",
            output_key="view_indices_B_T",
            dropout_rate=0.0,
            dtype=None,
        ),
        ref_cam_view_idx_sample_position=L(ReMapkey)(
            input_key="ref_cam_view_idx_sample_position",
            output_key="ref_cam_view_idx_sample_position",
            dropout_rate=0.0,
            dtype=None,
        ),
    ),
    conditioning_strategy=str(ConditioningStrategy.FRAME_REPLACE),
    min_num_conditional_frames_per_view=0,
    max_num_conditional_frames_per_view=1,
    condition_locations=[ConditionLocation.FIRST_RANDOM_N],
    net=_PREDICT2_MULTIVIEW_NET_2B_10FPS_7VIEWS_29FRAMES,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    resize_online=True,
    resolution="720",
    ema=L(EMAConfig)(enabled=False),  # defaults to inference
    sigma_conditional=0.0001,
    sigma_data=1.0,
    state_ch=16,
    state_t=8,
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        temporal_window=16,
        load_mean_std=False,
        name="tokenizer",
        vae_pth=get_cosmos_predict2_video2world_tokenizer(model_size="2B"),
    ),
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

_PREDICT2_MULTIVIEW_PIPELINE_2B_720P_10FPS_7VIEWS_29FRAMES = deepcopy(
    _PREDICT2_MULTIVIEW_PIPELINE_2B_10FPS_7VIEWS_29FRAMES
)


@dataclasses.dataclass(frozen=True)
class _MultiviewPipelineConfig:
    model_size: CosmosPredict2Video2WorldModelSize
    resolution: CosmosPredict2MultiviewResolution
    fps: CosmosPredict2MultiviewFPS
    views: dataclasses.field(default=CosmosPredict2MultiviewViews, kw_only=True)
    frames: dataclasses.field(default=CosmosPredict2MultiviewFrames, kw_only=True)


_PREDICT2_MULTIVIEW_PIPELINES: dict[
    _MultiviewPipelineConfig,
    MultiviewPipelineConfig,
] = {
    _MultiviewPipelineConfig(
        "2B", "720", 10, views=7, frames=29
    ): _PREDICT2_MULTIVIEW_PIPELINE_2B_720P_10FPS_7VIEWS_29FRAMES,
}


def get_cosmos_predict2_multiview_pipeline(
    *,
    model_size: CosmosPredict2Video2WorldModelSize,
    views: CosmosPredict2MultiviewViews,
    frames: CosmosPredict2MultiviewFrames,
    resolution: CosmosPredict2MultiviewResolution = "720",
    fps: CosmosPredict2MultiviewFPS = 16,
) -> MultiviewPipelineConfig:
    key = _MultiviewPipelineConfig(model_size, resolution, fps, views=views, frames=frames)
    return _PREDICT2_MULTIVIEW_PIPELINES[key]
