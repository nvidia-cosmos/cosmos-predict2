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

import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import distributed
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.nn import functional as F
from transformers.models.auto.processing_auto import AutoProcessor

from imaginaire.configs.reason1.model_config import FSDP2ModelConfig
from imaginaire.constants import COSMOS_REASON1_PRIVATE_TOKENIZER
from imaginaire.models.parallelisms.optimizer import build_lr_schedulers, build_optimizers
from imaginaire.models.parallelisms.parallel_dims import ParallelDims
from imaginaire.models.parallelisms.parallelize_qwen import parallelize_qwen
from imaginaire.networks.qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel, Qwen2_5_VLModel
from imaginaire.networks.qwen2_5_vl import get_rope_index as get_rope_index_v2
from imaginaire.networks.qwen2_5_vl import get_rope_index as get_rope_index_v2_5
from imaginaire.networks.qwen2_vl import Qwen2VisionTransformerPretrainedModel, Qwen2VLModel
from imaginaire.utils import log
from imaginaire.utils.checkpointer import _IncompatibleKeys
from imaginaire.utils.parallelism import broadcast_to_cp_or_tp_ranks
from imaginaire.utils.qwen_vl_utils import extract_vision_info, process_vision_info
from imaginaire.utils.torchtitan_utils import device_module, device_type

_LOCK_TIMEOUT_SECONDS = 60


class Processor:
    # This is a wrapper around the AutoProcessor class to add some helper functions
    def __init__(self, name="Qwen/Qwen2.5-VL-3B-Instruct", cache_dir=COSMOS_REASON1_PRIVATE_TOKENIZER):
        self.name = name
        self.processor = AutoProcessor.from_pretrained(cache_dir)
        log.info("Successfully loaded processor from local cache")

        if hasattr(self.processor, "image_token"):
            self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        else:
            self.image_token_id = None
        if hasattr(self.processor, "video_token"):
            self.video_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)
        else:
            self.video_token_id = None
        self.eos_id = self.processor.tokenizer.eos_token_id
        self.pad_id = self.processor.tokenizer.pad_token_id

    def apply_chat_template(
        self, messages, add_generation_prompt=False, return_tensors="pt", tokenize=True, add_vision_id=False
    ):
        assert tokenize, "tokenize must be True"
        if self.name.startswith("Qwen/Qwen2"):
            # Use manual workaround for add_vision_id bug
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                add_vision_id=add_vision_id,
            )
            image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)

            # add fps ourselves, as videos have been presampled according to the specified token length
            vision_infos = extract_vision_info(messages)
            fps_list = []
            for vision_info in vision_infos:
                if "video" in vision_info:
                    fps_list.append(vision_info["fps"])

            # process inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors=return_tensors,
                fps=fps_list,
                # padding="max_length",
                # max_length=5000,  # Set the fixed length (adjust based on model requirements)
                # truncation=True,  # Ensures text longer than `max_length` is truncated
            )

            # save raw text
            inputs["text"] = text

            # Convert batch features into single features
            # By default, the processor returns a batch of features, but we use processor in dataloader, so we need to convert it to single features
            inputs["input_ids"] = inputs["input_ids"][0]  # N_dialogue, N_token -> N_token
            inputs["attention_mask"] = inputs["attention_mask"][0]  # N_dialogue, N_token -> N_token
            # inputs["image_grid_thw"]: N_img, 3
            # inputs["video_grid_thw"]: N_video, 3
        else:
            raise ValueError(f"apply_chat_template is not implemented for tokenizer_type {self.name}")

        return inputs

    def add_assistant_tokens_mask(self, tokens):
        """
        Add a mask to the assistant tokens.
        This is used to mask out tokens that are not generated by the assistant (e.g.,  system prompts, user prompts, chat templates), such that in the loss computation, only the tokens generated by the assistant are used.
        If there are multiple turns in the conversation, the mask will mask all the assistant tokens in each turn.

        Args:
            tokens (Union[List[int], torch.Tensor]): The tokens to add the mask to.
        Returns:
            Union[List[bool], torch.Tensor]: The mask. True for tokens generated by the assistant (i.e. should apply loss on), False for tokens not generated by the assistant.
        """
        if isinstance(tokens, torch.Tensor) and tokens.ndim == 2:
            mask = torch.stack([self.add_assistant_tokens_mask(tokens[i]) for i in range(tokens.shape[0])])
            assert mask.shape == tokens.shape
            return mask
        np_tokens = tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else np.array(tokens)
        assert np_tokens.ndim == 1

        if self.name.startswith("Qwen/Qwen2"):
            # Constants defining bos, eos and fixed offsets.
            BOS_TOKEN = "<|im_start|>"
            EOS_TOKEN = "<|im_end|>"
            ROLE = "assistant"
            # Offsets: skip the bos + "assistant\n" (always 3 tokens) and include the eos (+1) for supervision
            START_OFFSET = 3
            END_OFFSET = 1

            # Retrieve token IDs for the markers and the role.
            bos_token_id = self.processor.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
            eos_token_id = self.processor.tokenizer.convert_tokens_to_ids(EOS_TOKEN)
            role_id = self.processor.tokenizer.convert_tokens_to_ids(ROLE)

            # Locate all positions where the start and end markers appear.
            start_indices = np.where(np_tokens == bos_token_id)[0]
            end_indices = np.where(np_tokens == eos_token_id)[0]

            # Initialize the mask with False values.
            masks = np.zeros_like(np_tokens, dtype=bool)
            assert len(start_indices) == len(end_indices)
            # For each pair of bos/eos, check if the role is 'assistant'
            # and apply the mask accordingly.
            for start, end in zip(start_indices, end_indices, strict=False):
                if np_tokens[start + 1] == role_id:
                    # Mask tokens from after the assistant header (start+3) to include the end marker (end+1)
                    masks[start + START_OFFSET : end + END_OFFSET] = True
        else:
            raise ValueError(f"add_assistant_tokens_mask is not implemented for tokenizer_type {self.name}")

        assert masks.shape == np_tokens.shape
        if isinstance(tokens, torch.Tensor):
            return torch.from_numpy(masks)
        else:
            return masks.tolist()

    def encode(self, *args, **kwargs):
        return self.processor.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.processor.decode(*args, **kwargs)


class VLMBaseModel(torch.nn.Module):
    """
    A class for base VLM model, has the shared methods for all VLM models

    Methods:
        build_model: build the model, should be implemented by each VLM model
        maybe_freeze_pretrained_modules: freeze the pretrained modules
        init_optimizer_scheduler: initialize the optimizer and scheduler
        get_num_params: get the number of parameters in the model
        load_state_dict: load the state dict
        validation_step: validation step
        forward: forward pass, should be implemented by each VLM model
        training_step: training step
        init_weights: initialize the weights, should be implemented by each VLM model
    """

    def __init__(
        self,
        model_config: FSDP2ModelConfig,
        tokenizer: Processor,
    ):
        super().__init__()
        """
        Build a AutoRegressiveModel instance by initializing and loading a model checkpoint.

        Args:
            model_config (FSDP2ModelConfig): The model configuration for the AutoRegressiveModel instance.
            tokenizer (Tokenizer): The tokenizer for the AutoRegressiveModel instance.
            download_rank_sync (bool, optional): Whether to download the checkpoint in a rank-synchronized manner. Defaults to True.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory.

        Note:
            This method sets the device to CUDA and loads the pre-trained model and tokenizer.
        """
        orig_precision = torch.get_default_dtype()
        precision = getattr(torch, model_config.precision)
        torch.set_default_dtype(precision)
        log.info(f"Setting torch default dtype from {orig_precision} to {precision}")
        self.tokenizer = tokenizer
        self.config = model_config
        self.precision = getattr(torch, model_config.precision)

        self.build_model(model_config)
        torch.set_default_dtype(orig_precision)  # Reset the default dtype to the original value
        log.info(f"Reset torch default dtype to {orig_precision}")

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        """The model preparation before the training is launched

        Args:
            memory_format (torch.memory_format): Memory format of the model.
        """
        pass

    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        """Hook before zero_grad() is called.

        Args:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            iteration (int): Current iteration number.
        """
        pass

    def on_after_backward(self, iteration: int = 0) -> None:
        """Hook after loss.backward() is called.

        This method is called immediately after the backward pass, allowing for custom operations
        or modifications to be performed on the gradients before the optimizer step.

        Args:
            iteration (int): Current iteration number.
        """
        pass

    def maybe_freeze_pretrained_modules(self):
        if self.config.freeze_vision_encoder:
            log.info("Freezing vision_encoder")
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        if self.config.freeze_mm_projector:
            log.info("Freezing mm_projector")
            for param in self.mm_projector.parameters():
                param.requires_grad = False
        if self.config.freeze_llm:
            log.info("Freezing llm")
            for param in self.model.parameters():
                param.requires_grad = False
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(
            f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
        )

    def init_optimizer_scheduler(
        self, optimizer_config, scheduler_config
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:


        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """

        model_parts = []
        lr_multiplier = []
        if not self.config.freeze_vision_encoder and self.vision_encoder is not None:
            log.info(
                f"adding vision_encoder to optimizer, lr_multiplier: {self.config.optimizer.lr_multiplier_vision_encoder}"
            )
            model_parts.append(self.vision_encoder)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_vision_encoder)
        if not self.config.freeze_mm_projector and self.mm_projector is not None:
            log.info(
                f"adding mm_projector to optimizer, lr_multiplier: {self.config.optimizer.lr_multiplier_mm_projector}"
            )
            model_parts.append(self.mm_projector)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_mm_projector)
        if not self.config.freeze_llm:
            log.info(f"adding llm to optimizer, lr_multiplier: {self.config.optimizer.lr_multiplier_llm}")
            model_parts.append(self.model)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_llm)
        optimizers = build_optimizers(model_parts, self.config, lr_multiplier)
        lr_schedulers = build_lr_schedulers(optimizers, self.config)
        return optimizers, lr_schedulers

    def get_num_params(
        self,
    ) -> int:
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False):
        """
        Ignore the missing keys with substrings matching `substring_to_ignore` (e.g., "_extra_state" keys imposed by
        TransformerEngine for FP8).
        """
        actual_missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=False, assign=assign)
        if strict:
            if len(actual_missing_keys) > 0 or len(unexpected_keys) > 0:
                raise ValueError(f"Missing keys: {actual_missing_keys}\n\nUnexpected keys: {unexpected_keys}")
        return _IncompatibleKeys(actual_missing_keys, unexpected_keys, incorrect_shapes=None)

    @torch.no_grad()
    def validation_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Perform a validation step for the model, which is the same as the training step (but without backpropagation).
        """
        return self.training_step(data_batch, iteration)

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        self.model.init_weights(buffer_device)
        if self.vision_encoder is not None:
            if self.config.vision_encoder.startswith("siglip"):
                pass
            elif self.config.vision_encoder in [
                "internvit-300m-448px-v2.5",
                "internvit-6b-448px-v2.5",
            ]:
                self.vision_encoder.init_weights()
            else:
                self.vision_encoder.init_weights(buffer_device)
        if self.mm_projector is not None:
            self.mm_projector.init_weights()

    @property
    def cp_mesh(self):
        return None

    @property
    def tp_mesh(self):
        return None

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        output_batch = {}
        if iteration < 20:
            summary_str = f"data_batch: {data_batch.keys()}"
            for key in data_batch.keys():
                if isinstance(data_batch[key], torch.Tensor):
                    summary_str += f" | {key} shape: {data_batch[key].shape}, dtype: {data_batch[key].dtype}"
            for key in ["__url__", "__key__", "image_grid_thw", "video_grid_thw"]:
                if key in data_batch:
                    summary_str += f" | {key}: {data_batch[key]}"
            log.info(summary_str, rank0_only=False)

        # first, broadcast if needed
        if self.cp_mesh is not None:
            broadcast_to_cp_or_tp_ranks(data_batch, self.cp_mesh)
        elif self.tp_mesh is not None:
            broadcast_to_cp_or_tp_ranks(data_batch, self.tp_mesh)

        # continue training
        tokens = data_batch["tokens"]
        tokens = tokens.to(device="cuda")

        # Token Mask (Note: this is not attention mask)
        token_mask = data_batch.get("token_mask", None)
        apply_token_mask = token_mask is not None

        if token_mask is None:
            token_mask = torch.ones_like(tokens, dtype=torch.bool)
        token_mask = token_mask.to(device="cuda")

        if self.config.aux_loss_coeff > 0:
            logits, aux_loss_list = self(tokens, data_batch, return_aux_loss=True)
            if len(aux_loss_list) > 0:
                assert aux_loss_list[0] is not None
                aux_loss = sum(aux_loss_list)
                output_batch["aux_loss_sum"] = aux_loss
                for i, aux_loss in enumerate(aux_loss_list):
                    output_batch[f"aux_loss_{i}"] = aux_loss
            else:
                aux_loss = None
        else:
            logits = self(tokens, data_batch)
        # For auto-regressive models, the labels are the same as the
        # input tokens shifted by one position
        logits = logits[:, :-1]
        token_mask = token_mask[:, 1:]
        labels = tokens[:, 1:].clone()

        # The PyTorch default ignore_index for the cross-entropy loss is -100.
        ignore_index = -100
        if apply_token_mask:
            labels[~token_mask] = ignore_index
        num_assistant_tokens = token_mask.float().sum()
        current_num_assistant_tokens = token_mask.float().sum()
        batch_size_local = tokens.shape[0]
        batch_size_global = torch.tensor(tokens.shape[0], device=tokens.device)

        dist.all_reduce(num_assistant_tokens, op=dist.ReduceOp.SUM)  # Sum of all num tokens with loss
        dist.all_reduce(batch_size_global, op=dist.ReduceOp.SUM)  # Sum of num of sequences
        avg_num_assistant_tokens = num_assistant_tokens / batch_size_global
        if "padding_mask" in data_batch:
            padding_mask = data_batch["padding_mask"]
            num_real_tokens = (~padding_mask).float().sum()
            dist.all_reduce(num_real_tokens, op=dist.ReduceOp.SUM)  # Sum of all tokens excluding padding
            avg_num_real_tokens = num_real_tokens / batch_size_global
            max_num_real_tokens = (~padding_mask).float().sum(dim=-1).max()
            dist.all_reduce(max_num_real_tokens, op=dist.ReduceOp.MAX)
            min_num_real_tokens = (~padding_mask).float().sum(dim=-1).min()
            dist.all_reduce(min_num_real_tokens, op=dist.ReduceOp.MIN)
        else:
            # No padding mask means all tokens are real tokens
            num_real_tokens = torch.tensor(float(tokens.numel()), device=tokens.device)
            dist.all_reduce(num_real_tokens, op=dist.ReduceOp.SUM)  # Sum of all tokens (no padding)
            avg_num_real_tokens = num_real_tokens / batch_size_global
            max_num_real_tokens = torch.tensor(float(tokens.shape[1]), device=tokens.device)
            dist.all_reduce(max_num_real_tokens, op=dist.ReduceOp.MAX)
            min_num_real_tokens = torch.tensor(float(tokens.shape[1]), device=tokens.device)
            dist.all_reduce(min_num_real_tokens, op=dist.ReduceOp.MIN)

        output_batch.update(
            {
                "encode_tokens": tokens,
                "logits": logits.detach(),
                "labels": labels.detach(),
                "ignore_index": ignore_index,
                "avg_num_assistant_tokens": avg_num_assistant_tokens.detach().item(),
                "avg_num_real_tokens": avg_num_real_tokens.detach().item(),
                "max_num_real_tokens": max_num_real_tokens.detach().item(),
                "min_num_real_tokens": min_num_real_tokens.detach().item(),
                "current_num_assistant_tokens": token_mask.float().sum().detach().item(),
                "batch_size_local": batch_size_local,
            }
        )
        logits = logits.flatten(0, 1)
        labels = labels.flatten(0, 1)

        # Main cross entropy loss
        if self.config.loss_per_token:
            ce_loss = F.cross_entropy(
                input=logits,
                target=labels,
                ignore_index=ignore_index,  # ignore prompt (turn prompt tokens into pad_id here)
                reduction="sum",
            )

            ce_loss = ce_loss / (batch_size_local * avg_num_assistant_tokens).detach()
        else:
            ce_loss = F.cross_entropy(
                input=logits,
                target=labels,
                ignore_index=ignore_index,  # ignore prompt (turn prompt tokens into pad_id here)
            )

        # Z-loss
        if self.config.z_loss_coeff > 0:
            if isinstance(logits, DTensor):
                local_logits = logits.to_local()  # Convert to a local tensor
            else:
                local_logits = logits
            log_z_local = torch.logsumexp(local_logits, dim=-1)

            z_loss_local = self.config.z_loss_coeff * (log_z_local**2).mean()
            if isinstance(ce_loss, DTensor):
                z_loss_dtensor = DTensor.from_local(
                    z_loss_local,
                    device_mesh=ce_loss.device_mesh,  # use the same device mesh as ce_loss
                    placements=ce_loss.placements,  # use the same sharding/placement strategy
                )
            else:
                z_loss_dtensor = z_loss_local
            # Combined loss
            total_loss = ce_loss + z_loss_dtensor
        else:
            total_loss = ce_loss

        output_batch["ce_loss"] = ce_loss
        if self.config.aux_loss_coeff > 0 and aux_loss is not None:
            total_loss += aux_loss * self.config.aux_loss_coeff
        return output_batch, total_loss  # skip returning output logits

    # These methods should be implemented by each VLM model
    def build_model(self, model_config):
        raise NotImplementedError

    def forward(self, tokens, data_batch={}, start_pos: int = 0) -> torch.Tensor:  # noqa: B006
        """
        The forward pass of the model.
        Returns:
            logits (torch.Tensor): The logits of the model.
        """
        raise NotImplementedError


class QwenModel(VLMBaseModel):
    """
    A class to build and use a AutoRegressiveModel model for text generation.
    This class is mimicing Qwen2_5_VLForConditionalGenerationSimple

    Methods:
        generate: Generate text sequences based on provided prompts using the language generation model.
    """

    def __init__(
        self,
        model_config: FSDP2ModelConfig,
        tokenizer: Processor,
    ) -> "QwenModel":
        super().__init__(model_config, tokenizer)
        self.forward_time = []

    def build_model(self, model_config):
        if model_config.model_type == "qwen2_5_vl":
            self.visual = Qwen2_5_VisionTransformerPretrainedModel(model_config.vision_config)
            self.model = Qwen2_5_VLModel(model_config)
        elif model_config.model_type == "qwen2_vl":
            self.visual = Qwen2VisionTransformerPretrainedModel(model_config.vision_config)
            self.model = Qwen2VLModel(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")
        self.vocab_size = model_config.vocab_size
        self.lm_head = nn.Linear(model_config.hidden_size, model_config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here]

        if torch.distributed.is_initialized():
            # TODO: apply the parallelisms
            self.world_mesh, self.parallel_dims = init_mesh(model_config)
            parallelize_qwen(self, self.world_mesh, self.parallel_dims, model_config)
            self.model.set_cp_mesh(self.cp_mesh)

    @property
    def vision_encoder(self):
        # This is to be compatible with VLMBaseModel
        return self.visual

    @property
    def mm_projector(self):
        # This is to be compatible with VLMBaseModel
        return self.visual.merger

    def init_optimizer_scheduler(
        self, optimizer_config, scheduler_config
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Creates the optimizer and scheduler for the model.

        Args:


        Returns:
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
        """

        model_parts = []
        model_part_names = []
        lr_multiplier = []
        if not self.config.freeze_vision_encoder and self.vision_encoder is not None:
            log.info(
                f"adding vision_encoder to optimizer, lr_multiplier: {self.config.optimizer.lr_multiplier_vision_encoder}"
            )
            model_parts.append(self.visual.patch_embed)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_vision_encoder)
            model_part_names.append("visual.patch_embed")
            model_parts.append(self.visual.blocks)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_vision_encoder)
            model_part_names.append("visual.blocks")
        if not self.config.freeze_mm_projector and self.mm_projector is not None:
            log.info(
                f"adding mm_projector to optimizer, lr_multiplier: {self.config.optimizer.lr_multiplier_mm_projector}"
            )
            model_parts.append(self.visual.merger)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_mm_projector)
            model_part_names.append("visual.merger")
        if not self.config.freeze_llm:
            log.info(f"adding llm to optimizer, lr_multiplier: {self.config.optimizer.lr_multiplier_llm}")
            model_parts.append(self.model)
            lr_multiplier.append(self.config.optimizer.lr_multiplier_llm)
            model_part_names.append("llm")
        optimizers = build_optimizers(model_parts, self.config, lr_multiplier, model_part_names)
        lr_schedulers = build_lr_schedulers(optimizers, self.config)
        return optimizers, lr_schedulers

    def maybe_freeze_pretrained_modules(self):
        if self.config.freeze_vision_encoder:
            log.info("Freezing vision_encoder")
            for param in self.visual.patch_embed.parameters():
                param.requires_grad = False
            for param in self.visual.blocks.parameters():
                param.requires_grad = False
        if self.config.freeze_mm_projector:
            log.info("Freezing mm_projector")
            for param in self.visual.merger.parameters():
                param.requires_grad = False
        if self.config.freeze_llm:
            log.info("Freezing llm")
            for param in self.model.parameters():
                param.requires_grad = False
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(
            f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
        )

    @property
    def cp_mesh(self):
        if not torch.distributed.is_initialized():
            return None
        # when none of the parallelisms are enabled, the world_mesh.mesh_dim_names is None
        if self.world_mesh.mesh_dim_names is not None and "cp" in self.world_mesh.mesh_dim_names:
            return self.world_mesh["cp"]
        else:
            return None

    @property
    def tp_mesh(self):
        if not torch.distributed.is_initialized():
            return None
        # when none of the parallelisms are enabled, the world_mesh.mesh_dim_names is None
        if self.world_mesh.mesh_dim_names is not None and "tp" in self.world_mesh.mesh_dim_names:
            return self.world_mesh["tp"]
        else:
            return None

    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            # This is a trick to handle TP for LLM but no TP for vision encoder, we need to convert DTensor to regular tensor later
            is_inputs_embeds_dtensor = isinstance(inputs_embeds, DTensor)  # This is True for TP>1, False for TP=1
            if is_inputs_embeds_dtensor:
                target_device_mesh = inputs_embeds.device_mesh
                target_placements = inputs_embeds.placements
                inputs_embeds = inputs_embeds.full_tensor()

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if is_inputs_embeds_dtensor:
                inputs_embeds = (
                    DTensor.from_local(inputs_embeds, device_mesh=target_device_mesh)
                    .redistribute(placements=target_placements)
                    .to_local()
                )
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                if self.config.model_type == "qwen2_5_vl":
                    position_ids, rope_deltas = get_rope_index_v2_5(
                        self.config,
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts,
                        attention_mask,
                    )
                elif self.config.model_type == "qwen2_vl":
                    position_ids, rope_deltas = get_rope_index_v2(
                        self.config,
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        attention_mask,
                    )
                else:
                    raise ValueError(f"Unsupported model type: {self.config.model_type}")
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(  # Qwen2_5_VLModel
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if self.cp_mesh is not None:
            logits = DTensor.from_local(logits, device_mesh=self.cp_mesh, placements=[Shard(1)]).full_tensor()
        return logits

    def forward(self, tokens, data_batch={}, start_pos: int = 0) -> torch.Tensor:  # noqa: B006
        """
        The training step of the model, including the loss computation.
        """
        assert "pixel_values" not in data_batch, "pixel_values should not be in data_batch, use images instead"
        pixel_values = data_batch.get("images", None)
        image_grid_thw = data_batch.get("image_grid_thw", None)
        pixel_values_videos = data_batch.get("videos", None)
        video_grid_thw = data_batch.get("video_grid_thw", None)
        if image_grid_thw is not None:
            assert len(image_grid_thw) == 1, "Only batch=1 is supported for now, due to `get_rope_index`"
            image_grid_thw = image_grid_thw[0]  # 1, N_img, 3 -> N_img, 3
            second_per_grid_ts = None
        if video_grid_thw is not None:
            assert len(video_grid_thw) == 1, "Only batch=1 is supported for now, due to `get_rope_index`"
            video_grid_thw = video_grid_thw[0]  # 1, N_video, 3 -> N_video, 3
            if "second_per_grid_ts" in data_batch:  # only 2.5VL has fps
                second_per_grid_ts = data_batch["second_per_grid_ts"][0]  # 1, N_video -> N_video
            else:
                second_per_grid_ts = None
        logits = self._forward(
            input_ids=tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )
        return logits

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if iteration < 20:
            if "raw_video" in data_batch:
                log.info(f"Raw video shape: {data_batch['raw_video'].shape}")
            if "videos" in data_batch:
                log.info(f"Processed video tokens shape: {data_batch['videos'].shape}")
                if "second_per_grid_ts" in data_batch:  # only 2.5VL has fps
                    log.info(f"second_per_grid_ts: {data_batch['second_per_grid_ts']}")
            if "images" in data_batch:
                log.info(f"images shape: {data_batch['images'].shape}")
        return super().training_step(data_batch, iteration)


def broadcast_object(local_str: list[str], cp_or_tp_mesh: DeviceMesh):
    """
    Broadcast a string to all ranks.
    """
    group = cp_or_tp_mesh.get_group()
    gathered_list = [None for _ in range(dist.get_world_size(group=group))]
    dist.all_gather_object(gathered_list, local_str, group=group)
    output_str = gathered_list[0]
    return output_str


def init_mesh(model_config):
    world_size = distributed.get_world_size()
    parallel_dims = ParallelDims(
        dp_shard=model_config.training.data_parallel_shard_degree,
        dp_replicate=model_config.training.data_parallel_replicate_degree,
        cp=model_config.training.context_parallel_degree,
        tp=model_config.training.tensor_parallel_degree,
        pp=model_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not model_config.training.disable_loss_parallel,
    )
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    return world_mesh, parallel_dims


def build_tokenizer(
    tokenizer_type: str,
    cache_dir: str = COSMOS_REASON1_PRIVATE_TOKENIZER,
):
    return Processor(tokenizer_type, cache_dir)
