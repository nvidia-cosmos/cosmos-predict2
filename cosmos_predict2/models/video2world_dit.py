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
        self.save_input_path = None  # For auto_deploy input capture

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
        
        # Save inputs if path is set (for auto_deploy compilation)
        if self.save_input_path:
            # Save core arguments AND critical kwargs for accurate video conditioning
            inputs_to_save = {
                "x_B_C_T_H_W": x_B_C_T_H_W.detach().cpu(),
                "timesteps_B_T": timesteps_B_T.detach().cpu(),
                "crossattn_emb": crossattn_emb.detach().cpu() if crossattn_emb is not None else None,
                "fps": fps.detach().cpu() if fps is not None else None,
                "padding_mask": padding_mask.detach().cpu() if padding_mask is not None else None,
                "condition_video_input_mask_B_C_T_H_W": condition_video_input_mask_B_C_T_H_W.detach().cpu() if condition_video_input_mask_B_C_T_H_W is not None else None,
                "data_type": data_type,
                "use_cuda_graphs": use_cuda_graphs,
                # Include critical video conditioning parameters
                "gt_frames": kwargs.get("gt_frames").detach().cpu() if kwargs.get("gt_frames") is not None else None,
                "use_video_condition": kwargs.get("use_video_condition"),
            }
            torch.save(inputs_to_save, self.save_input_path)
            self.save_input_path = None

        del kwargs

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
