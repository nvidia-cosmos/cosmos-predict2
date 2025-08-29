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

import abc
import functools
from enum import Enum
from typing import Any, ClassVar, Literal, TypeAlias, overload

import attrs
import torch
import transformers
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from transformers import T5EncoderModel, T5TokenizerFast
from typing_extensions import Self, assert_never, override

from imaginaire.configs.reason1.model_config_qwen import QwenModelConfig, QwenVisionConfig
from imaginaire.constants import COSMOS_REASON1_PRIVATE_CHECKPOINT, T5_MODEL_DIR, TEXT_ENCODER_CLASS, TextEncoderClass
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import instantiate as lazy_instantiate
from imaginaire.models.utils import load_state_dict
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


class CosmosTextEncoderBase(torch.nn.Module, abc.ABC):
    @overload
    def encode_prompts(self, prompts: str, max_length: int, return_mask: Literal[False]) -> torch.Tensor: ...
    @overload
    def encode_prompts(
        self, prompts: list[str], max_length: int, return_mask: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @abc.abstractmethod
    def encode_prompts(
        self, prompts: str | list[str], max_length: int | None = None, return_mask: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encodes text prompts into hidden state representations.

        This function tokenizes the input prompts, processes them through a T5 text encoder,
        and returns the last hidden states. The encoded outputs beyond the actual sequence
        length are zero-padded. All prompts in a batch are padded to max_length.

        Args:
            prompts: Input text to encode. Can be a single string or a list of strings.
            max_length: Maximum sequence length for tokenization and padding. Longer
                sequences will be truncated. Defaults to num_tokens.
            return_mask: If True, returns the attention mask along with encoded text.
                Defaults to False.

        Returns:
            If return_mask is False:
                torch.Tensor: Encoded text embeddings of shape (batch_size, max_length, hidden_size).
            If return_mask is True:
                tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                    - Encoded text embeddings of shape (batch_size, max_length, hidden_size)
                    - Attention mask of shape (batch_size, max_length) as boolean tensor

        Raises:
            ValueError: If the input prompts list is empty.
        """


@attrs.define(slots=False)
class CosmosReason1TextEncoderConfig:
    """
    Config for the text encoder model
    """

    CKPT_PATH: ClassVar[str] = COSMOS_REASON1_PRIVATE_CHECKPOINT
    NUM_TOKENS: ClassVar[int] = 512
    EMBED_DIM: ClassVar[int] = 100352

    compute_online: bool = True
    embedding_concat_strategy: str = str(EmbeddingConcatStrategy.FULL_CONCAT)
    n_layers_per_group: int = 5
    ckpt_path: str = CKPT_PATH
    num_tokens: int = NUM_TOKENS
    embed_dim: int = EMBED_DIM
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


class CosmosReason1TextEncoder(CosmosTextEncoderBase):
    def __init__(
        self, config: CosmosReason1TextEncoderConfig, device: str = "cuda", torch_dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.config = config
        self.device = device

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
        model: nn.Module,
        ckpt_path: str,
    ):
        log.info(f"Loading checkpoint from {ckpt_path}.")
        is_fsdp = False
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            is_fsdp = torch.distributed.get_world_size() > 1
        state_dict = load_state_dict(ckpt_path)
        # remove _extra_state
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith("._extra_state")}

        # Load Regular weights.
        if is_fsdp:
            set_model_state_dict(
                model,
                state_dict,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    strict=False,
                ),
            )
        else:
            model.load_state_dict(state_dict, strict=False)

        del state_dict
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

        self.model = self.model.to(self.device)
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

    @override
    def encode_prompts(self, prompts: str | list[str], max_length: int | None = None, return_mask: bool = False):
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("The input prompt list is empty.")
        if return_mask:
            raise NotImplementedError("return_mask is not supported for CosmosReason1TextEncoder")
        return self.compute_text_embeddings_online(prompts)


@attrs.define(slots=False)
class CosmosT5TextEncoderConfig:
    """
    Config for the T5 text encoder model
    """

    CKPT_PATH: ClassVar[str] = T5_MODEL_DIR
    NUM_TOKENS: ClassVar[int] = 512
    EMBED_DIM: ClassVar[int] = 1024

    ckpt_path: str = CKPT_PATH
    num_tokens: int = NUM_TOKENS
    embed_dim: int = EMBED_DIM


class CosmosT5TextEncoder(CosmosTextEncoderBase):
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

    @override
    @torch.inference_mode()
    def encode_prompts(self, prompts: str | list[str], max_length: int | None = None, return_mask: bool = False):
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            raise ValueError("The input prompt list is empty.")
        if max_length is None:
            max_length = self.config.num_tokens

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

        if return_mask:
            return encoded_text, attn_mask.bool()
        return encoded_text


if TEXT_ENCODER_CLASS == TextEncoderClass.COSMOS_REASON1:
    _TEXT_ENCODER_CONFIG = CosmosReason1TextEncoderConfig
elif TEXT_ENCODER_CLASS == TextEncoderClass.T5:
    _TEXT_ENCODER_CONFIG = CosmosT5TextEncoderConfig
else:
    assert_never(TEXT_ENCODER_CLASS)


@attrs.define(slots=False)
class CosmosTextEncoderConfig:
    NUM_TOKENS: ClassVar[int] = _TEXT_ENCODER_CONFIG.NUM_TOKENS
    EMBED_DIM: ClassVar[int] = _TEXT_ENCODER_CONFIG.EMBED_DIM

    cls: TextEncoderClass = TEXT_ENCODER_CLASS
    cosmos_reason1: CosmosReason1TextEncoderConfig = attrs.field(factory=CosmosReason1TextEncoderConfig)
    t5: CosmosT5TextEncoderConfig = attrs.field(factory=CosmosT5TextEncoderConfig)


CosmosTextEncoder: TypeAlias = CosmosReason1TextEncoder | CosmosT5TextEncoder


def get_cosmos_text_encoder(
    config: CosmosTextEncoderConfig, device: str = "cuda", torch_dtype: torch.dtype | None = None
) -> CosmosTextEncoder | None:
    """Create a text encoder from a config.

    Args:
        config: The config for the text encoder.
        device: The device to use for computations.
        torch_dtype: The torch dtype to use for computations.

    Returns:
        A text encoder instance.
    """

    if config.cls == TextEncoderClass.COSMOS_REASON1:
        if not config.cosmos_reason1.ckpt_path:
            return None
        return CosmosReason1TextEncoder(config=config.cosmos_reason1, device=device)
    elif config.cls == TextEncoderClass.T5:
        if not config.t5.ckpt_path:
            return None
        return CosmosT5TextEncoder(config=config.t5, device=device, torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Invalid text encoder config type: {config.cls}")
