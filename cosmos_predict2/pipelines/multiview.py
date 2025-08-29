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

import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange
from megatron.core import parallel_state
from torch.distributed import get_process_group_ranks
from tqdm import tqdm

from cosmos_predict2.auxiliary.cosmos_reason1 import CosmosReason1
from cosmos_predict2.conditioner import DataType, TextCondition
from cosmos_predict2.configs.base.config_multiview import (
    MultiviewPipelineConfig,
)
from cosmos_predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_predict2.models.utils import init_weights_on_device, load_state_dict
from cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.schedulers.rectified_flow_scheduler import (
    RectifiedFlowAB2Scheduler,
)
from cosmos_predict2.utils.context_parallel import (
    broadcast_split_tensor,
    cat_outputs_cp,
    split_inputs_cp,
)
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig, get_cosmos_text_encoder
from imaginaire.lazy_config import instantiate
from imaginaire.utils import log, misc
from imaginaire.utils.easy_io import easy_io
from imaginaire.utils.ema import FastEmaModelUpdater

_VIDEO_EXTENSIONS = [".mp4"]
NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"
SAMPLE_N_VIEWS_KEY: str = "sample_n_views"


def read_and_process_multiview_video(
    video_path: str,
    resolution: list[int],
    num_video_frames: int,
    num_latent_conditional_frames: int = 2,
    resize: bool = True,
    n_views: int = 1,
):
    """
    Reads a video, processes it for model input.

    The video is loaded using easy_io, and uses the last 4x(num_latent_conditional_frames - 1) + 1 from the video.
    If the video is shorter than num_video_frames, it pads with the last frame repeated.
    The first num_latent_conditional_frames are marked as conditioning frames.

    Args:
        video_path (str): Path to the input video file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int): Number of frames needed by the model (should equal model.tokenizer.get_pixel_num_frames(model.config.state_t)).
        num_latent_conditional_frames (int): Number of latent conditional frames from the input video (1 or 2).
        resize (bool, optional): Whether to resize the video to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W) where T equals num_video_frames.

    Raises:
        ValueError: If the video extension is not supported or other validation errors.

    Note:
        Uses the last 4x(num_latent_conditional_frames - 1) + 1 frames from the video. If video is shorter, pads with last frame repeated.
    """
    ext = os.path.splitext(video_path)[1]
    if ext.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Invalid video extension: {ext}")

    # Load video using easy_io
    try:
        video_frames, video_metadata = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
        log.info(f"Loaded video with shape {video_frames.shape}, metadata: {video_metadata}")
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")  # noqa: B904

    # Convert numpy array to tensor and rearrange dimensions
    video_tensor = torch.from_numpy(video_frames).float() / 255.0  # Convert to [0, 1] range
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    C, T, H, W = video_tensor.shape

    available_frames_per_view = video_tensor.shape[1] // n_views

    # Calculate how many frames to extract from input video
    frames_to_extract_per_view = 4 * (num_latent_conditional_frames - 1) + 1 if num_latent_conditional_frames > 0 else 0
    log.info(
        f"Will extract the last {frames_to_extract_per_view} frames from input video and pad to {num_video_frames}"
    )

    # Validate num_latent_conditional_frames
    if num_latent_conditional_frames not in [0, 1, 2]:
        raise ValueError(f"num_latent_conditional_frames must be 0, 1 or 2, but got {num_latent_conditional_frames}")

    if available_frames_per_view < frames_to_extract_per_view:
        raise ValueError(
            f"Video has only {available_frames_per_view} frames per view but needs at least {frames_to_extract_per_view} frames for num_latent_conditional_frames={num_latent_conditional_frames}"
        )

    # Pre-allocate tensor with target dimensions and directly in uint8 format
    # This avoids allocating a large float tensor that would be converted later
    full_video = torch.zeros((num_video_frames, C, H, W), dtype=torch.uint8, device=video_tensor.device)

    if num_latent_conditional_frames > 0:
        # Extract the last frames_to_extract from each input video view
        video_tensor = rearrange(video_tensor, "c (v t) h w -> c v t h w", v=n_views)
        start_idx = available_frames_per_view - frames_to_extract_per_view
        extracted_frames = video_tensor[:, :, start_idx:, :, :]  # (C, V, frames_to_extract_per_view, H, W)
        extracted_frames = rearrange(
            extracted_frames, "c v t h w -> c (v t) h w"
        )  # (C, V*frames_to_extract_per_view, H, W)

        # Convert to (V*frames_to_extract, C, H, W) for resize
        extracted_frames = extracted_frames.permute(1, 0, 2, 3)  # (V*frames_to_extract, C, H, W)
        log.info(f"extracted_frames.shape: {extracted_frames.shape}")

        # Resize the extracted frames if needed (more efficient than resizing full tensor later)
        if resize:
            C, H, W = extracted_frames.shape[1], extracted_frames.shape[2], extracted_frames.shape[3]
            target_h, target_w = resolution

            # Calculate scaling based on aspect ratio
            scaling_ratio = max((target_w / W), (target_h / H))
            resizing_shape = (int(math.ceil(scaling_ratio * H)), int(math.ceil(scaling_ratio * W)))  # noqa: RUF046

            # Resize and crop the extracted frames
            extracted_frames = torchvision.transforms.functional.resize(extracted_frames, resizing_shape)
            extracted_frames = torchvision.transforms.functional.center_crop(extracted_frames, resolution)

        # Get final dimensions
        C, H, W = extracted_frames.shape[1], extracted_frames.shape[2], extracted_frames.shape[3]

        full_video_v_t_c_h_w = rearrange(full_video, "(v t) c h w -> v t c h w", v=n_views)
        extracted_frames_v_t_c_h_w = rearrange(extracted_frames, "(v t) c h w -> v t c h w", v=n_views)

        # Set extracted frames and convert to uint8
        full_video_v_t_c_h_w[:, :frames_to_extract_per_view] = (extracted_frames_v_t_c_h_w * 255.0).to(torch.uint8)
        full_video = rearrange(full_video_v_t_c_h_w, "v t c h w -> (v t) c h w")

    # Add batch dimension and permute in one operation to final format
    # [T, C, H, W] -> [1, C, T, H, W]
    full_video = full_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return full_video


class MultiviewPipeline(Video2WorldPipeline):
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)

    @staticmethod
    def from_config(
        config: MultiviewPipelineConfig,
        dit_path: str = "",
        use_text_encoder: bool = True,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_ema_to_reg: bool = False,
        load_prompt_refiner: bool = False,
    ) -> Any:
        # Create a pipe
        pipe = MultiviewPipeline(device=device, torch_dtype=torch_dtype)
        pipe.config = config
        pipe.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        pipe.tensor_kwargs = {"device": "cuda", "dtype": pipe.precision}
        log.warning(f"precision {pipe.precision}")

        # 1. set data keys and data information
        pipe.sigma_data = config.sigma_data
        pipe.setup_data_key()

        log.info(f"config.timestamps: {config.timestamps}")
        # 2. setup up diffusion processing and scaling~(pre-condition)
        pipe.scheduler = RectifiedFlowAB2Scheduler(
            sigma_min=config.timestamps.t_min,
            sigma_max=config.timestamps.t_max,
            order=config.timestamps.order,
            t_scaling_factor=config.rectified_flow_t_scaling_factor,
        )

        pipe.scaling = RectifiedFlowScaling(pipe.sigma_data, config.rectified_flow_t_scaling_factor)

        # 3. Set up tokenizer
        pipe.tokenizer = instantiate(config.tokenizer)
        assert pipe.tokenizer.latent_ch == pipe.config.state_ch, (
            f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {pipe.config.state_ch}"
        )

        # 4. Load text encoder
        if use_text_encoder:
            pipe.text_encoder = get_cosmos_text_encoder(config=config.text_encoder, device=device)
        else:
            pipe.text_encoder = None

        # 5. Initialize conditioner
        pipe.conditioner = instantiate(config.conditioner)
        assert sum(p.numel() for p in pipe.conditioner.parameters() if p.requires_grad) == 0, (
            "conditioner should not have learnable parameters"
        )

        if load_prompt_refiner:
            pipe.prompt_refiner = CosmosReason1(
                checkpoint_dir=config.prompt_refiner_config.checkpoint_dir,
                offload_model_to_cpu=config.prompt_refiner_config.offload_model_to_cpu,
                enabled=config.prompt_refiner_config.enabled,
            )

        if config.guardrail_config.enabled:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            pipe.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
            )
            pipe.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
                config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
            )
        else:
            pipe.text_guardrail_runner = None
            pipe.video_guardrail_runner = None

        # 6. Set up DiT
        if dit_path:
            log.info(f"Loading DiT from {dit_path}")
        else:
            log.warning("dit_path not provided, initializing DiT with random weights")
        with init_weights_on_device():
            dit_config = config.net
            pipe.dit = instantiate(dit_config).eval()  # inference

        if dit_path:
            # load weights - already materializes tensors
            state_dict = load_state_dict(dit_path)
            prefix_to_load = "net_ema." if load_ema_to_reg else "net."
            # drop net. prefix
            state_dict_dit_compatible = dict()
            for k, v in state_dict.items():
                if k.startswith(prefix_to_load):
                    state_dict_dit_compatible[k[len(prefix_to_load) :]] = v
                else:
                    state_dict_dit_compatible[k] = v
            pipe.dit.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
            del state_dict, state_dict_dit_compatible
            log.success(f"Successfully loaded DiT from {dit_path}")
        else:
            pipe.dit = pipe.dit.to(device=device, dtype=torch_dtype)

        # 6-2. Handle EMA
        if config.ema.enabled:
            pipe.dit_ema = instantiate(dit_config).eval()
            pipe.dit_ema.requires_grad_(False)

            pipe.dit_ema_worker = FastEmaModelUpdater()  # default when not using FSDP

            s = config.ema.rate
            pipe.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
            # copying is only necessary when starting the training at iteration 0.
            # Actual state_dict should be loaded after the pipe is created.
            pipe.dit_ema_worker.copy_to(src_model=pipe.dit, tgt_model=pipe.dit_ema)

        pipe.dit = pipe.dit.to(device=device, dtype=torch_dtype)
        torch.cuda.empty_cache()

        # 7. training states
        if parallel_state.is_initialized():
            pipe.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            pipe.data_parallel_size = 1

        return pipe

    def _get_data_batch_input(
        self,
        video: torch.Tensor,
        prompt: str,
        negative_prompt: str = "",
        num_latent_conditional_frames: int = 1,
        n_views: int = 1,
        fps: int = 30,
    ):
        """
        Prepares the input data batch for the diffusion model.

        Constructs a dictionary containing the video tensor, text embeddings,
        and other necessary metadata required by the model's forward pass.
        Optionally includes negative text embeddings.

        Args:
            video (torch.Tensor): The input video tensor (B, C, T, H, W).
            prompt (str): The text prompt for conditioning.
            negative_prompt (str): Negative prompt.
            num_latent_conditional_frames (int, optional): The number of latent conditional frames. Defaults to 1.

        Returns:
            dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
        """
        B, C, T, H, W = video.shape
        t5_text_embeddings = torch.zeros(
            B, n_views * CosmosTextEncoderConfig.NUM_TOKENS, CosmosTextEncoderConfig.EMBED_DIM, dtype=self.torch_dtype
        ).to(self.device)
        if prompt.endswith(".txt"):
            prompts = open(prompt).read().splitlines()
            assert len(prompts) == n_views, (
                f"Number of prompts {len(prompts)} should be equal to number of views {n_views}"
            )
            for i, prompt in enumerate(prompts):
                if i != 0:
                    log.info(f"prompt for view {i} will not be used, skipping")
                    continue
                log.info(f"{i}. encode prompt: {prompt}")
                t5_text_embeddings[
                    :, i * CosmosTextEncoderConfig.NUM_TOKENS : (i + 1) * CosmosTextEncoderConfig.NUM_TOKENS
                ] = self.encode_prompt(prompt).to(dtype=self.torch_dtype).to(self.device)
        elif prompt.endswith(".pt"):
            t5_text_embeddings = torch.load(prompt)
            assert t5_text_embeddings.shape[1] == n_views * CosmosTextEncoderConfig.NUM_TOKENS, (
                f"t5_text_embeddings.shape[1] {t5_text_embeddings.shape[1]} should be {n_views * CosmosTextEncoderConfig.NUM_TOKENS}"
            )
        else:
            t5_text_embeddings[:, 0 : CosmosTextEncoderConfig.NUM_TOKENS] = (
                self.encode_prompt(prompt).to(dtype=self.torch_dtype).to(self.device)
            )
        latent_view_indices_T = torch.repeat_interleave(torch.arange(n_views), self.config.state_t)
        latent_view_indices_B_T = latent_view_indices_T.unsqueeze(0).expand(B, -1).to(self.device)

        data_batch = {
            "sample_n_views": n_views,
            "latent_view_indices_B_T": latent_view_indices_B_T,
            "ref_cam_view_idx_sample_position": torch.tensor([-1]).to(self.device),
            "dataset_name": "video_data",
            "video": video,
            "t5_text_embeddings": t5_text_embeddings,
            "fps": fps * torch.ones(B).to(self.device),
            "padding_mask": torch.zeros(B, 1, H, W).to(self.device),
            "num_conditional_frames": num_latent_conditional_frames,
        }

        # Handle negative prompts for classifier-free guidance
        if negative_prompt:
            log.warning("Negative prompt is only applied to the first view")
            neg_t5_text_embeddings = torch.zeros(
                B,
                n_views * CosmosTextEncoderConfig.NUM_TOKENS,
                CosmosTextEncoderConfig.EMBED_DIM,
                dtype=self.torch_dtype,
            ).to(self.device)
            neg_t5_text_embeddings[:, 0 : CosmosTextEncoderConfig.NUM_TOKENS] = self.encode_prompt(negative_prompt).to(
                dtype=self.torch_dtype
            )
            data_batch["neg_t5_text_embeddings"] = neg_t5_text_embeddings

        # Move tensors to GPU and convert to bfloat16 if they are floating point
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

        return data_batch

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.config.state_t)
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if n_views > 4 and cp_size > 1 and n_views <= cp_size:
            return self.encode_cp(state)
        state = rearrange(state, "B C (V T) H W -> (B V) C T H W", V=n_views)
        encoded_state = super().encode(state)
        encoded_state = rearrange(encoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return encoded_state

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        n_views = latent.shape[2] // self.config.state_t
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if n_views > 4 and cp_size > 1 and n_views <= cp_size:
            return self.decode_cp(latent)
        latent = rearrange(latent, "B C (V T) H W -> (B V) C T H W", V=n_views)
        decoded_state = super().decode(latent)
        decoded_state = rearrange(decoded_state, "(B V) C T H W -> B C (V T) H W", V=n_views)
        return decoded_state

    @torch.no_grad()
    def encode_cp(self, state: torch.Tensor) -> torch.Tensor:
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        n_views = state.shape[2] // self.tokenizer.get_pixel_num_frames(self.config.state_t)
        assert n_views < cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
        state_V_B_C_T_H_W = rearrange(state, "B C (V T) H W -> V B C T H W", V=n_views)
        state_input = torch.zeros((cp_size, *state_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        state_input[0:n_views] = state_V_B_C_T_H_W
        local_state_V_B_C_T_H_W = broadcast_split_tensor(state_input, seq_dim=0, process_group=cp_group)
        local_state = rearrange(local_state_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        encoded_state = super().encode(local_state)
        encoded_state_list = [torch.empty_like(encoded_state) for _ in range(cp_size)]
        dist.all_gather(encoded_state_list, encoded_state, group=cp_group)
        encoded_state = torch.cat(encoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return encoded_state

    @torch.no_grad()
    def decode_cp(self, latent: torch.Tensor) -> torch.Tensor:
        cp_size = len(get_process_group_ranks(parallel_state.get_context_parallel_group()))
        cp_group = parallel_state.get_context_parallel_group()
        log.info(f"latent.shape: {latent.shape}")
        log.info(f"self.config.state_t: {self.config.state_t}")
        n_views = latent.shape[2] // self.config.state_t
        assert n_views < cp_size, f"n_views must be less than cp_size, got n_views={n_views} and cp_size={cp_size}"
        latent_V_B_C_T_H_W = rearrange(latent, "B C (V T) H W -> V B C T H W", V=n_views)
        latent_input = torch.zeros((cp_size, *latent_V_B_C_T_H_W.shape[1:]), **self.tensor_kwargs)
        latent_input[0:n_views] = latent_V_B_C_T_H_W
        local_latent_V_B_C_T_H_W = broadcast_split_tensor(latent_input, seq_dim=0, process_group=cp_group)
        local_latent = rearrange(local_latent_V_B_C_T_H_W, "V B C T H W -> (B V) C T H W")
        decoded_state = super().decode(local_latent)
        decoded_state_list = [torch.empty_like(decoded_state) for _ in range(cp_size)]
        dist.all_gather(decoded_state_list, decoded_state, group=cp_group)
        decoded_state = torch.cat(decoded_state_list[0:n_views], dim=2)  # [B, C, V * T, H, W]
        return decoded_state

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, torch.Tensor], input_key: str = None) -> None:  # noqa: RUF013
        input_key = self.input_video_key if input_key is None else input_key
        if input_key in data_batch:
            num_video_frames_per_view = self.tokenizer.get_pixel_num_frames(self.config.state_t)
            n_views = data_batch[input_key].shape[2] // num_video_frames_per_view
            data_batch[input_key] = rearrange(data_batch[input_key], "B C (V T) H W -> (B V) C T H W", V=n_views)
            super()._normalize_video_databatch_inplace(data_batch, input_key)
            data_batch[input_key] = rearrange(data_batch[input_key], "(B V) C T H W -> B C (V T) H W", V=n_views)

    def broadcast_split_for_model_parallelsim(
        self,
        x0_B_C_T_H_W: torch.Tensor,
        condition: torch.Tensor,
        epsilon_B_C_T_H_W: torch.Tensor,
        sigma_B_T: torch.Tensor,
    ):
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        n_views = x0_B_C_T_H_W.shape[2] // self.config.state_t
        if cp_size > 1 and n_views > 1:
            x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views).contiguous()
            if epsilon_B_C_T_H_W is not None:
                epsilon_B_C_T_H_W = rearrange(
                    epsilon_B_C_T_H_W, "B C (V T) H W -> (B V) C T H W", V=n_views
                ).contiguous()
            reshape_sigma_B_T = False
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if sigma_B_T.shape[-1] != 1:
                    assert sigma_B_T.shape[-1] % n_views == 0, (
                        f"sigma_B_T temporal dimension T must either be 1 or a multiple of sample_n_views. Got T={sigma_B_T.shape[-1]} and sample_n_views={n_views}"
                    )
                    sigma_B_T = rearrange(sigma_B_T, "B (V T) -> (B V) T", V=n_views).contiguous()
                    reshape_sigma_B_T = True
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = super().broadcast_split_for_model_parallelsim(
                x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
            )
            x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
            if epsilon_B_C_T_H_W is not None:
                epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "(B V) C T H W -> B C (V T) H W", V=n_views)
            if reshape_sigma_B_T:
                sigma_B_T = rearrange(sigma_B_T, "(B V) T -> B (V T)", V=n_views)
        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, TextCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Latent state
        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_video_key]
        latent_state = self.encode(raw_state).contiguous().float()
        # Condition
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        condition = condition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
        )
        return raw_state, latent_state, condition

    def get_x0_fn_from_batch(
        self,
        data_batch: dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
        use_cuda_graphs: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.

        This function first processes the input data batch through a conditioning workflow (`conditioner`) to obtain conditioned and unconditioned states. It then defines a nested function `x0_fn` which applies a denoising operation on an input `noise_x` at a given noise level `sigma` using both the conditioned and unconditioned states.

        Args:
        - data_batch (Dict): A batch of data used for conditioning. The format and content of this dictionary should align with the expectations of the `self.conditioner`
        - guidance (float, optional): A scalar value that modulates the influence of the conditioned state relative to the unconditioned state in the output. Defaults to 1.5.
        - is_negative_prompt (bool): use negative prompt t5 in uncondition if true

        Returns:
        - Callable: A function `x0_fn(noise_x, sigma)` that takes two arguments, `noise_x` and `sigma`, and return x0 predictoin

        The returned function is suitable for use in scenarios where a denoised state is required based on both conditioned and unconditioned inputs, with an adjustable level of guidance influence.
        """
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        _, x0, _ = self.get_data_and_condition(data_batch)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        uncondition = uncondition.set_video_condition(
            state_t=self.config.state_t,
            gt_frames=x0,
            condition_locations=self.config.condition_locations,
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames_per_view,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames_per_view,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        condition = condition.edit_for_inference(
            is_cfg_conditional=True,
            condition_locations=self.config.condition_locations,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False,
            condition_locations=self.config.condition_locations,
            num_conditional_frames_per_view=num_conditional_frames,
        )
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if not parallel_state.is_initialized():
            assert not self.dit.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def x0_fn(noise_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            cond_x0 = self.denoise(noise_x, sigma, condition).x0
            uncond_x0 = self.denoise(noise_x, sigma, uncondition).x0
            raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)

            if "guided_image" in data_batch:
                # replacement trick that enables inpainting with base model
                assert "guided_mask" in data_batch, "guided_mask should be in data_batch if guided_image is present"
                guide_image = data_batch["guided_image"]
                guide_mask = data_batch["guided_mask"]
                raw_x0 = guide_mask * guide_image + (1 - guide_mask) * raw_x0
            return raw_x0

        return x0_fn

    @torch.no_grad()
    def __call__(
        self,
        input_path: str,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        num_conditional_frames: int = 1,
        guidance: float = 7.0,
        n_views: int = 1,
        fps: int = 30,
        num_sampling_step: int = 35,
        seed: int = 0,
        use_cuda_graphs: bool = False,
        return_prompt: bool = False,
    ) -> torch.Tensor | None:
        # Parameter check
        width, height = VIDEO_RES_SIZE_INFO[self.config.resolution][aspect_ratio]
        height, width = self.check_resize_height_width(height, width)
        assert num_conditional_frames in [0, 1, 5], "num_conditional_frames must be 0, 1 or 5"
        num_latent_conditional_frames = (
            self.tokenizer.get_latent_num_frames(num_conditional_frames) if num_conditional_frames > 0 else 0
        )

        # Run text guardrail on the prompt
        if self.text_guardrail_runner is not None:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            log.info("Running guardrail check on prompt...")
            if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
                return None
            else:
                log.success("Passed guardrail on prompt")
        elif self.text_guardrail_runner is None:
            log.warning("Guardrail checks on prompt are disabled")

        # refine prompt only if prompt refiner is enabled
        if (
            hasattr(self, "prompt_refiner")
            and self.prompt_refiner is not None
            and getattr(self.config, "prompt_refiner_config", None)
            and getattr(self.config.prompt_refiner_config, "enabled", False)
        ):
            log.info("Starting prompt refinement...")
            prompt = self.prompt_refiner.refine_prompt(input_path, prompt)
            log.info("Finished prompt refinement")

            # Run text guardrail on the refined prompt
            if self.text_guardrail_runner is not None:
                log.info("Running guardrail check on refined prompt...")
                if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
                    return None
                else:
                    log.success("Passed guardrail on refined prompt")
            elif self.text_guardrail_runner is None:
                log.warning("Guardrail checks on refined prompt are disabled")
        elif (
            hasattr(self, "config")
            and hasattr(self.config, "prompt_refiner_config")
            and not self.config.prompt_refiner_config.enabled
        ):
            log.warning("Prompt refinement is disabled")

        log.info(f"{n_views=}")
        log.info(f"{self.tokenizer.get_pixel_num_frames(self.config.state_t)=}")
        num_video_frames = self.tokenizer.get_pixel_num_frames(self.config.state_t) * n_views
        log.info(f"{num_video_frames=}")

        # Detect file extension to determine appropriate reading function
        ext = os.path.splitext(input_path)[1].lower()
        if ext in _VIDEO_EXTENSIONS:
            # Always use video reading for video files, regardless of num_latent_conditional_frames
            vid_input = read_and_process_multiview_video(
                input_path,
                [height, width],
                num_video_frames,
                num_latent_conditional_frames,
                resize=True,
                n_views=n_views,
            )
        else:
            raise ValueError(f"Unsupported file extension: {ext}. Supported extensions are {_VIDEO_EXTENSIONS}")

        # Prepare the data batch with text embeddings
        data_batch = self._get_data_batch_input(
            vid_input,
            prompt,
            negative_prompt,
            num_latent_conditional_frames=num_latent_conditional_frames,
            n_views=n_views,
            fps=fps,
        )

        # preprocess
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_video_key
        n_sample = data_batch[input_key].shape[0]
        _T, _H, _W = data_batch[input_key].shape[-3:]
        state_shape = [
            self.config.state_ch,
            self.config.state_t * n_views,
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]

        x0_fn = self.get_x0_fn_from_batch(
            data_batch, guidance, is_negative_prompt=bool(negative_prompt), use_cuda_graphs=use_cuda_graphs
        )

        log.info("Starting video generation...")

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),  # noqa: RUF005
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * self.scheduler.config.sigma_max
        )

        # Split the input data and condition for model parallelism, if context parallelism is enabled.
        if self.dit.is_context_parallel_enabled:
            log.info("Splitting input data and condition for model parallelism, if context parallelism is enabled.")
            log.info(f"x_sigma_max.shape: {x_sigma_max.shape}")
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.get_context_parallel_group())

        # ------------------------------------------------------------------ #
        # Sampling loop driven by `RectifiedFlowAB2Scheduler`
        # ------------------------------------------------------------------ #
        scheduler = self.scheduler

        # Construct sigma schedule (L + 1 entries including simga_min) and timesteps
        scheduler.set_timesteps(num_sampling_step, device=x_sigma_max.device)

        # Bring the initial latent into the precision expected by the scheduler
        sample = x_sigma_max.to(dtype=torch.float32)

        x0_prev: torch.Tensor | None = None

        for i, _ in enumerate(tqdm(scheduler.timesteps, desc="Generating world", leave=False)):
            # Current noise level (sigma_t).
            sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)

            # `x0_fn` expects `sigma` as a tensor of shape [B] or [B, T]. We
            # pass a 1-D tensor broadcastable to any later shape handling.
            sigma_in = sigma_t.repeat(sample.shape[0])

            # x0 prediction with conditional and unconditional branches
            x0_pred = x0_fn(sample, sigma_in)

            # Scheduler step updates the noisy sample and returns the cached x0.
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=i,
                sample=sample,
                x0_prev=x0_prev,
            )

        # Final clean pass at sigma_min.
        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])
        samples = x0_fn(sample, sigma_in)

        # Merge context-parallel chunks back together if needed.
        if self.dit.is_context_parallel_enabled:
            cp_group = self.get_context_parallel_group()
            cp_size = 1 if cp_group is None else cp_group.size()
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=cp_group)
            if n_views > 1:
                samples = rearrange(
                    samples, "B C (c V T) H W -> B C (V c T) H W", c=cp_size, T=self.config.state_t // cp_size
                )
        # Decode
        video = self.decode(samples)  # shape: (B, C, T, H, W), possibly out of [-1, 1]

        # Run video guardrail on the generated video and apply postprocessing
        if self.video_guardrail_runner is not None:
            # Clamp to safe range before normalization
            video = video.clamp(-1.0, 1.0)
            video_normalized = (video + 1) / 2  # [0, 1]

            # Convert tensor to NumPy frames for guardrail processing
            video_squeezed = video_normalized.squeeze(0)  # (C, T, H, W)
            frames = (video_squeezed * 255).clamp(0, 255).to(torch.uint8)
            frames = frames.permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)

            # Run guardrail
            processed_frames = guardrail_presets.run_video_guardrail(frames, self.video_guardrail_runner)
            if processed_frames is None:
                return None
            else:
                log.success("Passed guardrail on generated video")

            # Convert processed frames back to tensor format
            processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
            processed_video = processed_video * 2 - 1  # back to [-1, 1]
            processed_video = processed_video.unsqueeze(0)

            video = processed_video.to(video.device, dtype=video.dtype)

        log.success("Video generation completed successfully")
        if return_prompt:
            return video, prompt
        else:
            return video

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch,
        guidance: float = 7.0,
        seed: int = 0,
        state_shape: tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        num_conditional_frames: int = 1,
    ) -> torch.Tensor | None:
        n_views = data_batch[SAMPLE_N_VIEWS_KEY]
        log.info(f"Generating {n_sample} samples with {num_conditional_frames=} and {n_views=}")
        # preprocess
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance, is_negative_prompt=is_negative_prompt)

        log.info("Starting video generation...")

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),  # noqa: RUF005
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * self.scheduler.config.sigma_max
        )

        # Split the input data and condition for model parallelism, if context parallelism is enabled.
        if self.dit.is_context_parallel_enabled:
            log.info("Splitting input data and condition for model parallelism, if context parallelism is enabled.")
            log.info(f"x_sigma_max.shape: {x_sigma_max.shape}")
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.get_context_parallel_group())

        # ------------------------------------------------------------------ #
        # Sampling loop driven by `RectifiedFlowAB2Scheduler`
        # ------------------------------------------------------------------ #
        scheduler = self.scheduler

        # Construct sigma schedule (L + 1 entries including simga_min) and timesteps
        scheduler.set_timesteps(num_steps, device=x_sigma_max.device)

        # Bring the initial latent into the precision expected by the scheduler
        sample = x_sigma_max.to(dtype=torch.float32)

        x0_prev: torch.Tensor | None = None

        for i, _ in enumerate(tqdm(scheduler.timesteps, desc="Generating world", leave=False)):
            # Current noise level (sigma_t).
            sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)

            # `x0_fn` expects `sigma` as a tensor of shape [B] or [B, T]. We
            # pass a 1-D tensor broadcastable to any later shape handling.
            sigma_in = sigma_t.repeat(sample.shape[0])

            # x0 prediction with conditional and unconditional branches
            x0_pred = x0_fn(sample, sigma_in)

            # Scheduler step updates the noisy sample and returns the cached x0.
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=i,
                sample=sample,
                x0_prev=x0_prev,
            )

        # Final clean pass at sigma_min.
        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])
        samples = x0_fn(sample, sigma_in)

        # Merge context-parallel chunks back together if needed.
        if self.dit.is_context_parallel_enabled:
            cp_group = self.get_context_parallel_group()
            cp_size = 1 if cp_group is None else cp_group.size()
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=cp_group)
            if n_views > 1:
                samples = rearrange(
                    samples, "B C (c V T) H W -> B C (V c T) H W", c=cp_size, T=self.config.state_t // cp_size
                )
        # Decode
        video = self.decode(samples)  # shape: (B, C, T, H, W), possibly out of [-1, 1]

        log.success("Video generation completed successfully")
        return video
