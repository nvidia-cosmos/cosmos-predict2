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

from typing import List, Optional, Tuple

import torch

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.text2image_dit import MiniTrainDIT
from imaginaire.utils import log


class MinimalV1LVGDiT(MiniTrainDIT):
    def __init__(self, *args, **kwargs):
        assert "in_channels" in kwargs, "in_channels must be provided"
        kwargs["in_channels"] += 1  # Add 1 for the condition mask
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        use_cuda_graphs: bool = False,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        # Save inputs if path is set (for auto_deploy compilation)
        if self.save_input_path:
            log.info(f"=== ACTUAL DiT INPUT ANALYSIS ===")
            log.info(f"x_B_C_T_H_W shape: {x_B_C_T_H_W.shape}")
            log.info(f"timesteps_B_T shape: {timesteps_B_T.shape}")
            log.info(f"crossattn_emb shape: {crossattn_emb.shape if crossattn_emb is not None else None}")
            log.info(f"fps shape: {fps.shape if fps is not None else None}")
            log.info(f"padding_mask shape: {padding_mask.shape if padding_mask is not None else None}")
            log.info(f"condition_video_input_mask_B_C_T_H_W shape: {condition_video_input_mask_B_C_T_H_W.shape if condition_video_input_mask_B_C_T_H_W is not None else None}")
            
            # Analyze the main input tensor
            B, C, T, H, W = x_B_C_T_H_W.shape
            patch_dim = 2  # spatial patch size
            temporal_patch = 1  # temporal patch size
            flattened_dim = C * patch_dim * patch_dim * temporal_patch
            log.info(f"Main input analysis: B={B}, C={C}, T={T}, H={H}, W={W}")
            log.info(f"Patch embedding input: {C} * {patch_dim} * {patch_dim} * {temporal_patch} = {flattened_dim}")
            log.info(f"Expected error format: [{H//patch_dim * W//patch_dim * T}*s0, {flattened_dim}] X [72, 2048]")
            log.info(f"=== END ANALYSIS ===")
            
            # Clear the save path to avoid repeated logs
            self.save_input_path = None

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )
        return super().forward(
            x_B_C_T_H_W=x_B_C_T_H_W,
            timesteps_B_T=timesteps_B_T,
            crossattn_emb=crossattn_emb,
            fps=fps,
            padding_mask=padding_mask,
            data_type=data_type,
            use_cuda_graphs=use_cuda_graphs,
        )
