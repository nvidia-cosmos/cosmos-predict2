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
from typing import Any, List, Tuple, Union
import math
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from einops import rearrange
from megatron.core import parallel_state
from tqdm import tqdm

from imaginaire.utils import log, misc
from imaginaire.lazy_config import LazyDict, instantiate
from imaginaire.utils.easy_io import easy_io
from imaginaire.utils.ema import FastEmaModelUpdater

from cosmos_predict2.utils.context_parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict2.auxiliary.cosmos_reason1 import CosmosReason1
from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder
from cosmos_predict2.conditioner import DataType
from cosmos_predict2.datasets.utils import IMAGE_RES_SIZE_INFO, VIDEO_RES_SIZE_INFO
from cosmos_predict2.models.utils import init_weights_on_device, load_state_dict
from cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling
from cosmos_predict2.schedulers.rectified_flow_scheduler import RectifiedFlowAB2Scheduler
from cosmos_predict2.configs.base.config_video2world import Video2WorldPipelineConfig
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline, get_sample_batch
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline, read_and_process_video, read_and_process_image

_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", "webp"]
_VIDEO_EXTENSIONS = [".mp4"]

def process_video_first_frame(
    video_path: str,
    resolution: list[int],
    resize: bool = True,
):
    """
    Reads a video and extracts the first frame as a tensor.

    Args:
        video_path (str): Path to the input video file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        resize (bool, optional): Whether to resize the frame to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed frame tensor of shape (1, C, 1, H, W) - single frame with batch and time dimensions.

    Raises:
        ValueError: If the video extension is not supported or other validation errors.
    """
    ext = os.path.splitext(video_path)[1]
    if ext.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Invalid video extension: {ext}")

    # Load video using easy_io
    try:
        video_frames, video_metadata = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
        log.info(f"Loaded video with shape {video_frames.shape}, metadata: {video_metadata}")
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")

    # Extract only the first frame
    first_frame = video_frames[0]  # (H, W, C)
    
    # Convert numpy array to tensor and normalize to [0, 1] range
    frame_tensor = torch.from_numpy(first_frame).float() / 255.0  # (H, W, C)
    frame_tensor = frame_tensor.permute(2, 0, 1)  # (C, H, W)

    # Resize the frame if needed
    if resize:
        target_h, target_w = resolution
        H, W = frame_tensor.shape[1], frame_tensor.shape[2]

        # Calculate scaling based on aspect ratio
        scaling_ratio = max((target_w / W), (target_h / H))
        resizing_shape = (int(math.ceil(scaling_ratio * H)), int(math.ceil(scaling_ratio * W)))

        # Add batch dimension for resize operation
        frame_tensor = frame_tensor.unsqueeze(0)  # (1, C, H, W)
        
        # Resize and crop the frame
        frame_tensor = F.resize(frame_tensor, resizing_shape)
        frame_tensor = F.center_crop(frame_tensor, resolution)
        
        # Remove batch dimension
        frame_tensor = frame_tensor.squeeze(0)  # (C, H, W)

    # Add batch and time dimensions: (C, H, W) -> (1, C, 1, H, W)
    frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(2)  # (1, C, 1, H, W)
    
    return frame_tensor

def read_and_process_video_first_frames(
    video_path: str,
    resolution: list[int],
    num_video_frames: int,
    resize: bool = True,
):
    """
    Reads a video and extracts the first n frames, processes it for model input.

    The video is loaded using easy_io, and uses the first num_video_frames from the video.
    If the video is shorter than num_video_frames, it pads with the last frame repeated.

    Args:
        video_path (str): Path to the input video file.
        resolution (list[int]): Target resolution [H, W] for resizing.
        num_video_frames (int): Number of frames needed by the model (should equal model.tokenizer.get_pixel_num_frames(model.config.state_t)).
        resize (bool, optional): Whether to resize the video to the target resolution. Defaults to True.

    Returns:
        torch.Tensor: Processed video tensor of shape (1, C, T, H, W) where T equals num_video_frames.

    Raises:
        ValueError: If the video extension is not supported or other validation errors.

    Note:
        Uses the first num_video_frames frames from the video. If video is shorter, pads with last frame repeated.
    """
    ext = os.path.splitext(video_path)[1]
    if ext.lower() not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Invalid video extension: {ext}")

    # Load video using easy_io
    try:
        video_frames, video_metadata = easy_io.load(video_path)  # Returns (T, H, W, C) numpy array
        log.info(f"Loaded video with shape {video_frames.shape}, metadata: {video_metadata}")
    except Exception as e:
        raise ValueError(f"Failed to load video {video_path}: {e}")

    # Convert numpy array to tensor and rearrange dimensions
    video_tensor = torch.from_numpy(video_frames).float() / 255.0  # Convert to [0, 1] range
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)

    available_frames = video_tensor.shape[1]

    log.info(f"Will extract the first {num_video_frames} frames from input video")

    # Validate num_video_frames
    if num_video_frames <= 0:
        raise ValueError(f"num_video_frames must be positive, but got {num_video_frames}")

    # Extract the first num_video_frames from input video (or all available if less)
    frames_to_extract = min(available_frames, num_video_frames)
    extracted_frames = video_tensor[:, :frames_to_extract, :, :]  # (C, frames_to_extract, H, W)

    # Convert to (frames_to_extract, C, H, W) for resize
    extracted_frames = extracted_frames.permute(1, 0, 2, 3)  # (frames_to_extract, C, H, W)

    # Get last frame for padding
    last_frame = extracted_frames[-1]  # (C, H, W)

    # Resize the extracted frames if needed (more efficient than resizing full tensor later)
    if resize:
        C, H, W = extracted_frames.shape[1], extracted_frames.shape[2], extracted_frames.shape[3]
        target_h, target_w = resolution

        # Calculate scaling based on aspect ratio
        scaling_ratio = max((target_w / W), (target_h / H))
        resizing_shape = (int(math.ceil(scaling_ratio * H)), int(math.ceil(scaling_ratio * W)))

        # Resize and crop the extracted frames
        extracted_frames = torchvision.transforms.functional.resize(extracted_frames, resizing_shape)
        extracted_frames = torchvision.transforms.functional.center_crop(extracted_frames, resolution)

        # Resize and crop the last frame separately (for padding)
        last_frame = last_frame.unsqueeze(0)  # Add batch dim for resize
        last_frame = torchvision.transforms.functional.resize(last_frame, resizing_shape)
        last_frame = torchvision.transforms.functional.center_crop(last_frame, resolution)
        last_frame = last_frame.squeeze(0)  # Remove batch dim

    # Get final dimensions
    C, H, W = extracted_frames.shape[1], extracted_frames.shape[2], extracted_frames.shape[3]

    # Pre-allocate tensor with target dimensions and directly in uint8 format
    # This avoids allocating a large float tensor that would be converted later
    full_video = torch.zeros((num_video_frames, C, H, W), dtype=torch.uint8, device=extracted_frames.device)

    # Set extracted frames and convert to uint8
    full_video[:frames_to_extract] = (extracted_frames * 255.0).to(torch.uint8)

    # Pad remaining frames with the last frame (already resized if needed)
    if frames_to_extract < num_video_frames:
        last_frame_uint8 = (last_frame * 255.0).to(torch.uint8)
        for i in range(num_video_frames - frames_to_extract):
            full_video[frames_to_extract + i] = last_frame_uint8

    # Add batch dimension and permute in one operation to final format
    # [T, C, H, W] -> [1, C, T, H, W]
    full_video = full_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return full_video

class Text2ImageSDEditPipeline(Text2ImagePipeline):
    @staticmethod
    def from_config(
        config: LazyDict,
        dit_path: str = "",
        text_encoder_path: str = "",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_ema_to_reg: bool = False,
    ) -> Any:
        # Create a pipe
        pipe = Text2ImageSDEditPipeline(device=device, torch_dtype=torch_dtype)
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

        # 2. setup up diffusion processing and scaling~(pre-condition), sampler
        pipe.scheduler = RectifiedFlowAB2Scheduler(
            sigma_min=config.timestamps.t_min,
            sigma_max=config.timestamps.t_max,
            order=config.timestamps.order,
            t_scaling_factor=config.rectified_flow_t_scaling_factor,
        )
        pipe.scaling = RectifiedFlowScaling(
            pipe.sigma_data, config.rectified_flow_t_scaling_factor, config.rectified_flow_loss_weight_uniform
        )

        # 3. Set up tokenizer
        pipe.tokenizer = instantiate(config.tokenizer)
        assert (
            pipe.tokenizer.latent_ch == pipe.config.state_ch
        ), f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {pipe.config.state_ch}"

        # 4. Load text encoder
        if text_encoder_path:
            # inference
            pipe.text_encoder = CosmosT5TextEncoder(device=device, cache_dir=text_encoder_path)
            pipe.text_encoder.to(device)
        else:
            # training
            pipe.text_encoder = None

        # 5. Initialize conditioner
        pipe.conditioner = instantiate(config.conditioner)
        assert (
            sum(p.numel() for p in pipe.conditioner.parameters() if p.requires_grad) == 0
        ), "conditioner should not have learnable parameters"

        if config.guardrail_config.enabled:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            pipe.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
                config.guardrail_config.checkpoint_dir, config.guardrail_config.offload_model_to_cpu
            )
        else:
            pipe.text_guardrail_runner = None

        # 6. Set up DiT
        if dit_path:
            log.info(f"Loading DiT from {dit_path}")
        else:
            log.warning("dit_path not provided, initializing DiT with random weights")
        with init_weights_on_device():
            dit_config = config.net
            pipe.dit = instantiate(dit_config).eval()  # inference

        if dit_path:
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
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        input_video_path: str, 
        edit_strength: float = 0.5,
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        seed: int = 0,
        guidance: float = 4.0,
        num_sampling_step: int = 35,
        use_cuda_graphs: bool = False,
    ) -> torch.Tensor | None:
        # Parameter check
        width, height = IMAGE_RES_SIZE_INFO[self.config.resolution][aspect_ratio]
        height, width = self.check_resize_height_width(height, width)

        # Run text guardrail on the prompt
        if self.text_guardrail_runner is not None:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            log.info("Running guardrail check on prompt...")
            if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
                return None
            else:
                log.success("Passed guardrail on prompt")

        # get sample batch
        data_batch = get_sample_batch(resolution=self.config.resolution, aspect_ratio=aspect_ratio, batch_size=1)
        data_batch["t5_text_embeddings"] = self.encode_prompt(prompt).to(dtype=self.torch_dtype)
        data_batch["neg_t5_text_embeddings"] = self.encode_prompt(negative_prompt).to(dtype=self.torch_dtype)

        # preprocess
        self._augment_image_dim_inplace(data_batch)
        input_key = "images"
        n_sample = data_batch[input_key].shape[0]
        _T, _H, _W = data_batch[input_key].shape[-3:]
        state_shape = [
            self.config.state_ch,
            self.tokenizer.get_latent_num_frames(_T),
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]

        vid_first_frame = process_video_first_frame(
                input_video_path, [height, width], resize=True
            )
        data_batch["images"] = vid_first_frame.to(self.tensor_kwargs["device"], dtype=self.torch_dtype)

        # Obtains the latent state and condition.
        _, latent_state, _ = self.get_data_and_condition(data_batch)

        if negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        # Ensures we are conditioning on IMAGE data type.
        condition = condition.edit_data_type(DataType.IMAGE)
        uncondition = uncondition.edit_data_type(DataType.IMAGE)

        # Context parallelism is disabled for text2image
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(latent_state, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(latent_state, uncondition, None, None)

        log.info("Starting image generation...")

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            #* self.scheduler.config.sigma_max
        )

        # ------------------------------------------------------------------ #
        # Sampling loop driven by `RectifiedFlowAB2Scheduler`
        # ------------------------------------------------------------------ #
        scheduler = self.scheduler

        # Construct sigma schedule (L + 1 entries including simga_min) and timesteps
        scheduler.set_timesteps(num_sampling_step, device=x_sigma_max.device)

        # Bring the initial latent into the precision expected by the scheduler
        edit_step = int(num_sampling_step * (1 - edit_strength))
        sample = self.scheduler.add_noise(latent_state, x_sigma_max, self.scheduler.timesteps[edit_step:edit_step+1])
        sample = sample.to(dtype=torch.float32)
        timesteps = self.scheduler.timesteps[edit_step:]

        x0_prev: torch.Tensor | None = None

        for i, timestep in enumerate(tqdm(timesteps, desc="Generating image")):
            step = int(timestep)
            # Current noise level (sigma_t).
            sigma_t = scheduler.sigmas[step].to(sample.device, dtype=torch.float32)

            # `x0_fn` expects `sigma` as a tensor of shape [B] or [B, T]. We
            # pass a 1-D tensor broadcastable to any later shape handling.
            sigma_in = sigma_t.repeat(sample.shape[0])

            # x0 prediction with conditional and unconditional branches
            cond_x0 = self.denoise(sample, sigma_in, condition, use_cuda_graphs=use_cuda_graphs).x0
            uncond_x0 = self.denoise(sample, sigma_in, uncondition, use_cuda_graphs=use_cuda_graphs).x0
            x0_pred = cond_x0 + guidance * (cond_x0 - uncond_x0)

            # Scheduler step (handles float64 internally, returns original dtype)
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=step,
                sample=sample,
                x0_prev=x0_prev,
            )

        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])

        # Final clean pass.
        cond_x0 = self.denoise(sample, sigma_in, condition, use_cuda_graphs=use_cuda_graphs).x0
        uncond_x0 = self.denoise(sample, sigma_in, uncondition, use_cuda_graphs=use_cuda_graphs).x0
        samples = cond_x0 + guidance * (cond_x0 - uncond_x0)

        # decode
        image = self.decode(samples)

        log.success("Image generation completed successfully")
        return image

class Video2WorldSDEditPipeline(Video2WorldPipeline):
    @staticmethod
    def from_config(
        config: Video2WorldPipelineConfig,
        dit_path: str = "",
        text_encoder_path: str = "",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_ema_to_reg: bool = False,
        load_prompt_refiner: bool = False,
    ) -> Any:
        # Create a pipe
        pipe = Video2WorldSDEditPipeline(device=device, torch_dtype=torch_dtype)
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

        # 2. setup up diffusion processing and scaling~(pre-condition)
        pipe.scheduler = RectifiedFlowAB2Scheduler(
            sigma_min=config.timestamps.t_min,
            sigma_max=config.timestamps.t_max,
            order=config.timestamps.order,
            t_scaling_factor=config.rectified_flow_t_scaling_factor,
        )

        pipe.scaling = RectifiedFlowScaling(
            pipe.sigma_data, config.rectified_flow_t_scaling_factor, config.rectified_flow_loss_weight_uniform
        )

        # 3. Set up tokenizer
        pipe.tokenizer = instantiate(config.tokenizer)
        assert (
            pipe.tokenizer.latent_ch == pipe.config.state_ch
        ), f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {pipe.config.state_ch}"

        # 4. Load text encoder
        if text_encoder_path:
            # inference
            pipe.text_encoder = CosmosT5TextEncoder(device=device, cache_dir=text_encoder_path)
            pipe.text_encoder.to(device)
        else:
            # training
            pipe.text_encoder = None

        # 5. Initialize conditioner
        pipe.conditioner = instantiate(config.conditioner)
        assert (
            sum(p.numel() for p in pipe.conditioner.parameters() if p.requires_grad) == 0
        ), "conditioner should not have learnable parameters"

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

    @torch.no_grad()
    def __call__(
        self,
        input_path: str,
        prompt: str,
        input_video_path: str, 
        edit_strength: float = 0.5,
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        num_conditional_frames: int = 1,
        guidance: float = 7.0,
        num_sampling_step: int = 35,
        seed: int = 0,
        use_cuda_graphs: bool = False,
        return_prompt: bool = False,
    ) -> torch.Tensor | None:
        # Parameter check
        width, height = VIDEO_RES_SIZE_INFO[self.config.resolution][aspect_ratio]
        height, width = self.check_resize_height_width(height, width)
        assert num_conditional_frames in [1, 5], "num_conditional_frames must be 1 or 5"
        num_latent_conditional_frames = self.tokenizer.get_latent_num_frames(num_conditional_frames)

        # Run text guardrail on the prompt
        if self.text_guardrail_runner is not None:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            log.info("Running guardrail check on prompt...")
            if not guardrail_presets.run_text_guardrail(prompt, self.text_guardrail_runner):
                if return_prompt:
                    return None, prompt
                else:
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
                    if return_prompt:
                        return None, prompt
                    else:
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

        num_video_frames = self.tokenizer.get_pixel_num_frames(self.config.state_t)

        # Detect file extension to determine appropriate reading function
        ext = os.path.splitext(input_path)[1].lower()
        if ext in _VIDEO_EXTENSIONS:
            # Always use video reading for video files, regardless of num_latent_conditional_frames
            vid_input = read_and_process_video(
                input_path, [height, width], num_video_frames, num_latent_conditional_frames, resize=True
            )
        elif ext in _IMAGE_EXTENSIONS:
            if num_latent_conditional_frames == 1:
                # Use image reading for single frame conditioning with image files
                vid_input = read_and_process_image(input_path, [height, width], num_video_frames, resize=True)
            else:
                raise ValueError(
                    f"Cannot use multi-frame conditioning (num_conditional_frames={num_conditional_frames}) with image input. Please provide a video file."
                )
        else:
            raise ValueError(
                f"Unsupported file extension: {ext}. Supported extensions are {_IMAGE_EXTENSIONS + _VIDEO_EXTENSIONS}"
            )

        # read input video
        input_video = read_and_process_video_first_frames(input_video_path, [height, width], num_video_frames, resize=True)
        
        # Prepare the data batch with text embeddings
        data_batch = self._get_data_batch_input(
            vid_input, prompt, negative_prompt, num_latent_conditional_frames=num_latent_conditional_frames
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
            self.tokenizer.get_latent_num_frames(_T),
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]

        x0_fn = self.get_x0_fn_from_batch(
            data_batch, guidance, is_negative_prompt=True, use_cuda_graphs=use_cuda_graphs
        )
        
        log.info("Encoding input video...")
        input_video = input_video.cuda().to(dtype=torch.bfloat16)
        input_video = input_video.to(**self.tensor_kwargs) / 127.5 - 1.0
        input_video_latent = self.encode(input_video).contiguous().float()

        log.info("Starting video generation...")

        x_sigma_max = (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            # * self.scheduler.config.sigma_max
        )

        # Split the input data and condition for model parallelism, if context parallelism is enabled.
        if self.dit.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.get_context_parallel_group())

        # ------------------------------------------------------------------ #
        # Sampling loop driven by `RectifiedFlowAB2Scheduler`
        # ------------------------------------------------------------------ #
        scheduler = self.scheduler

        # Construct sigma schedule (L + 1 entries including simga_min) and timesteps
        scheduler.set_timesteps(num_sampling_step, device=x_sigma_max.device)

        # Bring the initial latent into the precision expected by the scheduler
        edit_step = int(num_sampling_step * (1 - edit_strength))
        sample = self.scheduler.add_noise(input_video_latent, x_sigma_max, self.scheduler.timesteps[edit_step:edit_step+1])
        sample = sample.to(dtype=torch.float32)
        timesteps = self.scheduler.timesteps[edit_step:]

        x0_prev: torch.Tensor | None = None

        for i, timestep in enumerate(tqdm(timesteps, desc="Generating video")):
            step = int(timestep)
            # Current noise level (sigma_t).
            sigma_t = scheduler.sigmas[step].to(sample.device, dtype=torch.float32)

            # `x0_fn` expects `sigma` as a tensor of shape [B] or [B, T]. We
            # pass a 1-D tensor broadcastable to any later shape handling.
            sigma_in = sigma_t.repeat(sample.shape[0])

            # x0 prediction with conditional and unconditional branches
            x0_pred = x0_fn(sample, sigma_in)

            # Scheduler step updates the noisy sample and returns the cached x0.
            sample, x0_prev = scheduler.step(
                x0_pred=x0_pred,
                i=step,
                sample=sample,
                x0_prev=x0_prev,
            )

        # Final clean pass at sigma_min.
        sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        sigma_in = sigma_min.repeat(sample.shape[0])
        samples = x0_fn(sample, sigma_in)

        # Merge context-parallel chunks back together if needed.
        if self.dit.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())

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
                if return_prompt:
                    return None, None, prompt
                else:
                    return None, None
            else:
                log.success("Passed guardrail on generated video")

            # Convert processed frames back to tensor format
            processed_video = torch.from_numpy(processed_frames).float().permute(3, 0, 1, 2) / 255.0
            processed_video = processed_video * 2 - 1  # back to [-1, 1]
            processed_video = processed_video.unsqueeze(0)

            video = processed_video.to(video.device, dtype=video.dtype)

        log.success("Video generation completed successfully")
        if return_prompt:
            return video, input_video.float(), prompt
        else:
            return video, input_video.float()
