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

import functools
from enum import Enum
from typing import Any, TypeAlias

import attrs
import torch
import transformers
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from transformers import T5EncoderModel, T5TokenizerFast
from typing_extensions import Self

from imaginaire.configs.reason1.model_config_qwen import QwenModelConfig, QwenVisionConfig
from imaginaire.constants import COSMOS_REASON1_PRIVATE_CHECKPOINT, T5_MODEL_DIR, TEXT_ENCODER_CLASS, TextEncoderClass
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import instantiate as lazy_instantiate
from imaginaire.models.vlm_qwen import build_tokenizer
from imaginaire.models.vlm_qwen_omni import QwenVLBaseModel
from imaginaire.utils import log

transformers.utils.logging.set_verbosity_error()

NUM_EMBEDDING_PADDING_TOKENS = 512


class ModelWrapper(Stateful):
    """Wrapper for model state dict handling"""

    def __init__(self, model: nn.Module | list[nn.Module]):
        self.model = [model] if isinstance(model, nn.Module) else model

    def state_dict(self) -> dict[str, Any]:
        return {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_model_state_dict,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )
        list(map(func, self.model))


class EmbeddingConcatStrategy(str, Enum):
    FULL_CONCAT = "full_concat"  # Concatenate embeddings all layers
    MEAN_POOLING = "mean_pooling"  # Average pool embeddings all layers
    POOL_EVERY_N_LAYERS_AND_CONCAT = "pool_every_n_layers_and_concat"  # Pool every n layers and concatenatenate

    def __str__(self) -> str:
        return self.value


@attrs.define(slots=False)
class CosmosReason1TextEncoderConfig:
    """
    Config for the text encoder model
    """

    compute_online: bool = True
    embedding_concat_strategy: str = str(EmbeddingConcatStrategy.FULL_CONCAT)
    n_layers_per_group: int = 5
    ckpt_path: str = COSMOS_REASON1_PRIVATE_CHECKPOINT
    model_config: QwenVLBaseModel = L(QwenVLBaseModel)(  # noqa: RUF009
        model_config=L(QwenModelConfig)(
            tokenizer_type="Qwen/Qwen2.5-VL-7B-Instruct",
            name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
            hidden_size=3584,
            intermediate_size=18944,
            max_window_layers=28,
            num_attention_heads=28,
            num_hidden_layers=28,
            num_key_value_heads=4,
            tie_word_embeddings=False,
            vocab_size=152064,
            vision_config=L(QwenVisionConfig)(out_hidden_size=3584),
            output_hidden_states=True,
        ),
        tokenizer=L(build_tokenizer)(
            tokenizer_type="Qwen/Qwen2.5-VL-7B-Instruct",
        ),
    )


class CosmosReason1TextEncoder(torch.nn.Module):
    def __init__(
        self, config: CosmosReason1TextEncoderConfig, device: str = "cuda", torch_dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.config = config

        log.info("Instantiating text encoder model...")
        with torch.device("meta"):
            self.model: QwenVLBaseModel = lazy_instantiate(self.config.model_config)
        self.model.to_empty(device=device)
        with torch.no_grad():
            self.model.init_weights()
        self.load_checkpoint(self.model, self.config.ckpt_path)
        self.model.eval()
        torch.cuda.empty_cache()
        log.info("Text encoder model instantiated")

    @staticmethod
    def load_checkpoint(
        model_parts: list[nn.Module],
        ckpt_path: str,
        model_ckpt_key_map: dict[str, str] = {},  # noqa: B006
    ):
        log.info(f"Loading checkpoint from {ckpt_path}.")

        _model_wrapper = ModelWrapper(model_parts)
        state_dict = _model_wrapper.state_dict()
        # remove _extra_state
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith("._extra_state")}

        # remap keys if needed
        if model_ckpt_key_map:
            for model_key, checkpoint_key in model_ckpt_key_map.items():
                state_dict[checkpoint_key] = state_dict.pop(model_key)
                log.info(f"Re-mapping {model_key} to {checkpoint_key}")

        state_dict = torch.load(ckpt_path)

        # inverse the remapping if needed
        if model_ckpt_key_map:
            for model_key, checkpoint_key in model_ckpt_key_map.items():
                state_dict[model_key] = state_dict.pop(checkpoint_key)
                log.info(f"Inverse re-mapping {checkpoint_key} to {model_key}")

        _model_wrapper.load_state_dict(state_dict)

        log.info(f"Finished loading checkpoint from {ckpt_path}.")

    @staticmethod
    def mean_normalize(tensor: torch.Tensor) -> torch.Tensor:
        """
        Mean normalize a tensor by subtracting the mean and dividing by the standard deviation.

        Args:
        tensor (torch.tensor): The tensor to normalize

        Returns:
        torch.tensor: The normalized tensor
        """
        return (tensor - tensor.mean(dim=-1, keepdim=True)) / (tensor.std(dim=-1, keepdim=True) + 1e-8)

    def compute_text_embeddings_online(self, prompts: list[str]) -> torch.Tensor:
        """
        Compute text embeddings for the given prompts.
        """
        assert self.model is not None, "Text encoder is not initialized"

        # Tokenize prompts
        input_ids_batch = []

        for sample_idx in range(len(prompts)):
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompts[sample_idx],
                        }
                    ],
                },
            ]
            tokenizer_output = self.model.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
            )
            input_ids = tokenizer_output["input_ids"]
            pad_id = self.model.tokenizer.pad_id

            # Do padding or truncation
            if NUM_EMBEDDING_PADDING_TOKENS > len(input_ids):
                # Do padding:
                pad_len = NUM_EMBEDDING_PADDING_TOKENS - len(input_ids)
                input_ids = input_ids.tolist() + [pad_id] * pad_len
            else:
                # Do truncation:
                input_ids = input_ids.tolist()[:NUM_EMBEDDING_PADDING_TOKENS]
            input_ids = torch.LongTensor(input_ids).to(device="cuda")
            input_ids_batch.append(input_ids)

        input_ids_batch = torch.stack(input_ids_batch, dim=0)

        # Compute text embeddings
        with torch.no_grad():
            _, outputs_batch = self.model(input_ids_batch, {})
        hidden_states = outputs_batch["hidden_states"]

        # # Skip the embeddings of the system prompt
        # hidden_states = hidden_states[:, num_system_prompt_tokens:]

        # Now compute the normalized embeddings
        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            normalized_state = self.mean_normalize(hidden_states[layer_idx])
            normalized_hidden_states.append(normalized_state)

        text_embeddings = None
        if self.config.embedding_concat_strategy == str(EmbeddingConcatStrategy.FULL_CONCAT):
            text_embeddings = torch.cat(normalized_hidden_states, dim=-1)
        elif self.config.embedding_concat_strategy == str(EmbeddingConcatStrategy.MEAN_POOLING):
            # Stack the normalized hidden states and calculate the mean
            text_embeddings = torch.stack(normalized_hidden_states)
            text_embeddings = text_embeddings.mean(dim=0)
        elif self.config.embedding_concat_strategy == str(EmbeddingConcatStrategy.POOL_EVERY_N_LAYERS_AND_CONCAT):
            # Split the l
            n_layers_per_group = self.config.n_layers_per_group
            text_embeddings = []
            for i in range(0, len(normalized_hidden_states), n_layers_per_group):
                group_embeddings = normalized_hidden_states[i : i + n_layers_per_group]
                group_embedding = torch.stack(group_embeddings)
                group_embedding = group_embedding.mean(dim=0)
                text_embeddings.append(group_embedding)
            text_embeddings = torch.cat(text_embeddings, dim=-1)
        else:
            raise ValueError(f"Invalid embedding_concat_strategy: {self.config.embedding_concat_strategy}")

        return text_embeddings

    def encode_prompts(self, prompts: str | list[str], max_length: int = 512) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        return self.compute_text_embeddings_online(prompts)


@attrs.define(slots=False)
class CosmosT5TextEncoderConfig:
    """
    Config for the T5 text encoder model
    """

    ckpt_path: str = T5_MODEL_DIR


class CosmosT5TextEncoder(torch.nn.Module):
    """Handles T5 text encoding operations."""

    def __init__(
        self,
        config: CosmosT5TextEncoderConfig,
        device: str = "cuda",
        torch_dtype: torch.dtype | None = None,
    ):
        """Initializes the T5 tokenizer and encoder.

        Args:
            model_name: The name of the T5 model to use.
            device: The device to use for computations.
        """
        super().__init__()
        self.config = config
        self.device = device
        self.tokenizer = T5TokenizerFast.from_pretrained(self.config.ckpt_path, torch_dtype=torch_dtype)
        self.text_encoder = T5EncoderModel.from_pretrained(self.config.ckpt_path, torch_dtype=torch_dtype).to(device)
        self.text_encoder.eval()

        log.info("T5 Text encoder model instantiated")

    @property
    def model(self) -> Self:
        return self

    @torch.inference_mode()
    def encode_prompts(self, prompts: str | list[str], max_length: int = 512) -> torch.Tensor:
        """Encodes text prompts into hidden state representations using a T5 encoder.

        This function tokenizes the input prompts, processes them through a T5 text encoder,
        and returns the last hidden states. The encoded outputs beyond the actual sequence
        length are zero-padded. All prompts in a batch are padded to max_length.

        Args:
            prompts: Input text to encode. Can be a single string or a list of strings.
            max_length: Maximum sequence length for tokenization and padding. Longer
                sequences will be truncated. Defaults to 512.

        Returns:
            torch.Tensor: Encoded text embeddings of shape (batch_size, max_length, hidden_size).

        Raises:
            ValueError: If the input prompts list is empty.

        Example:
            >>> encoder = CosmosT5TextEncoder()
            >>> prompts = ["Hello world", "Another example"]
            >>> embeddings = encoder.encode_prompts(prompts, max_length=128)
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if not prompts:
            raise ValueError("The input prompt list is empty.")

        batch_encoding = self.tokenizer.batch_encode_plus(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_length=True,
            return_offsets_mapping=False,
        )

        input_ids = batch_encoding.input_ids.to(self.device)
        attn_mask = batch_encoding.attention_mask.to(self.device)

        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)

        encoded_text = outputs.last_hidden_state
        lengths = attn_mask.sum(dim=1).cpu()

        for batch_id in range(encoded_text.shape[0]):
            encoded_text[batch_id][lengths[batch_id] :] = 0

        return encoded_text


CosmosTextEncoderConfig: TypeAlias = CosmosReason1TextEncoderConfig | CosmosT5TextEncoderConfig
CosmosTextEncoder: TypeAlias = CosmosReason1TextEncoder | CosmosT5TextEncoder


def get_text_encoder_config(text_encoder_class: TextEncoderClass = TEXT_ENCODER_CLASS) -> CosmosTextEncoderConfig:
    """
    Get the text encoder config for the given text encoder class.

    Args:
        text_encoder_class: The text encoder class.

    Returns:
        The text encoder config.
    """
    if text_encoder_class == TextEncoderClass.COSMOS_REASON1:
        return CosmosReason1TextEncoderConfig()
    elif text_encoder_class == TextEncoderClass.T5:
        return CosmosT5TextEncoderConfig()
    else:
        raise ValueError(f"Invalid text encoder class: {text_encoder_class}")


def get_text_encoder(
    config: CosmosTextEncoderConfig, device: str = "cuda", torch_dtype: torch.dtype | None = None
) -> CosmosTextEncoder | None:
    """Create a text encoder from a config.

    Args:
        config: The config for the text encoder.
        device: The device to use for computations.

    Returns:
        A text encoder instance.
    """

    if not config.ckpt_path:
        return None
    if isinstance(config, CosmosReason1TextEncoderConfig):
        return CosmosReason1TextEncoder(config=config, device=device)
    elif isinstance(config, CosmosT5TextEncoderConfig):
        return CosmosT5TextEncoder(config=config, device=device, torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Invalid text encoder config type: {type(config)}")
