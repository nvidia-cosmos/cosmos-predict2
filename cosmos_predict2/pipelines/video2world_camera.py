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
from cosmos_predict2.conditioner import DataType
from cosmos_predict2.configs.base.config_video2world import Video2WorldPipelineConfig
from cosmos_predict2.models.utils import load_state_dict
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from megatron.core import parallel_state

from cosmos_predict2.auxiliary.cosmos_reason1 import CosmosReason1
from cosmos_predict2.utils.context_parallel import cat_outputs_cp, split_inputs_cp
from cosmos_predict2.module.res_sampler import COMMON_SOLVER_OPTIONS, Sampler
from cosmos_predict2.auxiliary.qwen_text_encoder import EmbeddingConcatStrategy, CosmosQwenTextEncoder
from cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling
from cosmos_predict2.schedulers.rectified_flow_scheduler import RectifiedFlowAB2Scheduler
from imaginaire.utils import log, misc
from imaginaire.lazy_config import instantiate as lazy_instantiate

class Video2WorldCameraConditionedPipeline(Video2WorldPipeline):
    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)

    @staticmethod
    def from_config(
        config: Video2WorldPipelineConfig,
        model_path: str,
        load_ema_to_reg: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        num_gpus: int = 1,
        cache_dir: str | None = None,
    ) -> Any:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = Video2WorldCameraConditionedPipeline(device=device, torch_dtype=torch_dtype)
        pipe.config = config

        pipe.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]

        pipe.tensor_kwargs = {"device": "cuda", "dtype": pipe.precision}

        # 1. set data keys and data information
        pipe.sigma_data = config.sigma_data
        pipe.setup_data_key()

        # 2. setup up diffusion processing and scaling~(pre-condition)
        # TODO: Once reflow post-trained model, replace sampler code with reflow scheduler

        # pipe.scheduler = RectifiedFlowAB2Scheduler(
        #     sigma_min=config.timestamps.t_min,
        #     sigma_max=config.timestamps.t_max,
        #     order=config.timestamps.order,
        #     t_scaling_factor=config.rectified_flow_t_scaling_factor,
        # )

        pipe.sampler = Sampler()

        pipe.scaling = RectifiedFlowScaling(
            pipe.sigma_data, 
            config.rectified_flow_t_scaling_factor, 
            config.rectified_flow_loss_weight_uniform
        )

        # 3. tokenizer
        pipe.tokenizer: TokenizerInterface = lazy_instantiate(config.tokenizer, vae_pth=os.path.join(model_path, "tokenizer.pth"))
        assert (
            pipe.tokenizer.latent_ch == config.state_ch
        ), f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {config.state_ch}"

        # 4. Set up loss options, including loss masking, loss reduce and loss scaling
        pipe.loss_reduce = getattr(config, "loss_reduce", "mean")
        assert pipe.loss_reduce in ["mean", "sum"]
        pipe.loss_scale = getattr(config, "loss_scale", 1.0)
        log.critical(f"Using {pipe.loss_reduce} loss reduce with loss scale {pipe.loss_scale}")
        if config.adjust_video_noise:
            pipe.video_noise_multiplier = math.sqrt(config.state_t)
        else:
            pipe.video_noise_multiplier = 1.0

        # 6. Initialize conditioner
        pipe.conditioner = lazy_instantiate(config.conditioner)
        assert (
            sum(p.numel() for p in pipe.conditioner.parameters() if p.requires_grad) == 0
        ), "conditioner should not have learnable parameters"
        pipe.conditioner = pipe.conditioner.to(**pipe.tensor_kwargs)

        # 7. Set up prompt refiner
        if config.prompt_refiner_config.enabled:
            pipe.prompt_refiner = CosmosReason1(
                checkpoint_dir=config.prompt_refiner_config.checkpoint_dir,
                offload_model_to_cpu=config.prompt_refiner_config.offload_model_to_cpu,
                enabled=config.prompt_refiner_config.enabled,
            )

        # 8. Set up guardrail
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

        # 9. Set up DiT
        log.info(f"Loading DiT from {model_path}")
        dit_config = config.net
        pipe.dit = lazy_instantiate(dit_config).eval()  # inference

        state_dict = load_state_dict(os.path.join(model_path, "model_ema_reg.pt"))
        prefix_to_load = "net_ema." if load_ema_to_reg else "net."

        log.info(f"Loading {'[ema]/regular' if load_ema_to_reg else 'ema/[regular]'} weights from {model_path}/model_ema_reg.pt")
        # drop net./net_ema. prefix if it exists, depending on the load_ema_to_reg flag
        state_dict_dit_compatible = dict()
        for k, v in state_dict.items():
            if k.startswith(prefix_to_load):
                state_dict_dit_compatible[k[len(prefix_to_load):]] = v
            else:
                state_dict_dit_compatible[k] = v
        pipe.dit.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
        del state_dict, state_dict_dit_compatible
        log.success(f"Successfully loaded DiT from {model_path}/model_ema_reg.pt")

        pipe.dit = pipe.dit.to(device=device, dtype=torch_dtype)
        torch.cuda.empty_cache()

        # 10. Set up text encoder
        pipe.text_encoder = None
        # Camera model uses QwenVL, check if there's text encoder config

        pipe.text_encoder = CosmosQwenTextEncoder(**{
            "device": device,
            "torch_dtype": torch_dtype,
            "embedding_concat_strategy": EmbeddingConcatStrategy.FULL_CONCAT,  # Full concatenation for 100352-dim embeddings
            "n_layers_per_group": 5,  # For pooling strategies
            "offload_model_to_cpu": False,  # Keep model on GPU for inference
            "cache_dir": cache_dir,
        })

        log.critical(f"Successfully loaded Cosmos QwenVL text encoder")
        
        return pipe

    def get_x0_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        use_negative_prompt: bool = False,
        num_conditional_frames: int = 1,
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

        cond, out1, out2 = torch.chunk(data_batch["camera"], 3, dim=1)
        data_batch["camera"] = torch.cat((out1, cond, out2), dim=1)

        if use_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        if self.is_image_batch(data_batch):
            raise ValueError("Image input is not supported for camera-conditioned video generation")

        condition = condition.edit_data_type(DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.VIDEO)

        x0_cond = self.encode(data_batch[self.input_video_key]).contiguous().float()
        x0 = torch.cat([torch.zeros_like(x0_cond), x0_cond, torch.zeros_like(x0_cond)], dim=2)

        condition = condition.set_camera_conditioned_video_condition(
            gt_frames=x0,
            num_conditional_frames=num_conditional_frames,
        )
        uncondition = uncondition.set_camera_conditioned_video_condition(
            gt_frames=x0,
            num_conditional_frames=num_conditional_frames,
        )

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert (
                not self.dit.is_context_parallel_enabled
            ), "parallel_state is not initialized, context parallel should be turned off."

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

        return x0_fn, x0_cond

    @torch.no_grad()
    def __call__(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        use_negative_prompt: bool = False,
        num_steps: int = 35,
        solver_option: COMMON_SOLVER_OPTIONS = "2ab",
        num_conditional_frames: int = 1,
    ) -> torch.Tensor:
        """
        Generate video samples from the batch.
        Args:
            data_batch (dict): raw data batch draw from the training data loader.
            iteration (int): Current iteration number.
            guidance (float): guidance weights
            seed (int): random seed
            state_shape (tuple): shape of the state, default to data batch if not provided
            n_sample (int): number of samples to generate
            is_negative_prompt (bool): use negative prompt t5 in uncondition if true
            num_steps (int): number of steps for the diffusion process
            solver_option (str): differential equation solver option, default to "2ab"~(mulitstep solver)
        """

        if self.is_image_batch(data_batch):
            raise ValueError("Image input is not supported for camera-conditioned video generation")
        if n_sample is None:
            n_sample = data_batch[self.input_video_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[self.input_video_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        x0_fn, x0_cond = self.get_x0_fn_from_batch(data_batch, guidance, use_negative_prompt=True, num_conditional_frames=num_conditional_frames)

        sigma_max = self.config.timestamps.t_max
        sigma_min = self.config.timestamps.t_min
        
        create_x_sigma_max = lambda: (
            misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            * sigma_max
        )

        x_sigma_max = torch.cat([create_x_sigma_max(), x0_cond, create_x_sigma_max()], dim=2)

        if self.dit.is_context_parallel_enabled:
            x_sigma_max = split_inputs_cp(x=x_sigma_max, seq_dim=2, cp_group=self.get_context_parallel_group())

        # TODO: Once reflow post-trained model, replace sampler code with reflow scheduler

        # scheduler = self.scheduler

        # # Construct sigma schedule (L + 1 entries including simga_min) and timesteps
        # scheduler.set_timesteps(num_steps, device=x_sigma_max.device)

        # # Bring the initial latent into the precision expected by the scheduler
        # sample = x_sigma_max.to(dtype=torch.float32)

        # x0_prev: torch.Tensor | None = None

        # for i, _ in enumerate(scheduler.timesteps):
        #     # Current noise level (sigma_t).
        #     sigma_t = scheduler.sigmas[i].to(sample.device, dtype=torch.float32)

        #     # `x0_fn` expects `sigma` as a tensor of shape [B] or [B, T]. We
        #     # pass a 1-D tensor broadcastable to any later shape handling.
        #     sigma_in = sigma_t.repeat(sample.shape[0])

        #     # x0 prediction with conditional and unconditional branches
        #     x0_pred = x0_fn(sample, sigma_in)

        #     # Scheduler step updates the noisy sample and returns the cached x0.
        #     sample, x0_prev = scheduler.step(
        #         x0_pred=x0_pred,
        #         i=i,
        #         sample=sample,
        #         x0_prev=x0_prev,
        #     )

        # # Final clean pass at sigma_min.
        # sigma_min = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        # sigma_in = sigma_min.repeat(sample.shape[0])
        # samples = x0_fn(sample, sigma_in)

        samples = self.sampler(
            x0_fn,
            x_sigma_max,
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            solver_option=solver_option,
        )

        if self.dit.is_context_parallel_enabled:
            samples = cat_outputs_cp(samples, seq_dim=2, cp_group=self.get_context_parallel_group())

        out1, _ , out2 = torch.chunk(samples, 3, dim=2)
        video = torch.cat([self.decode(out1), self.decode(out2)], dim=3)

        return video