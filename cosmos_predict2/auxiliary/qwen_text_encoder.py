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

from enum import Enum
from typing import List, Union

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from imaginaire.utils import log

NUM_EMBEDDING_PADDING_TOKENS = 512


class EmbeddingConcatStrategy(str, Enum):
    FULL_CONCAT = "full_concat"  # Concatenate embeddings all layers
    MEAN_POOLING = "mean_pooling"  # Average pool embeddings all layers
    POOL_EVERY_N_LAYERS_AND_CONCAT = "pool_every_n_layers_and_concat"  # Pool every n layers and concatenatenate

    def __str__(self) -> str:
        return self.value


class CosmosQwenTextEncoder(torch.nn.Module):
    """Handles Qwen text encoding operations."""

    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason1-7B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        embedding_concat_strategy: str = str(EmbeddingConcatStrategy.FULL_CONCAT),
        n_layers_per_group: int = 5,
        offload_model_to_cpu: bool = False,
        cache_dir: str | None = None,
    ):
        """Initializes the Qwen tokenizer and encoder.

        Args:
            model_name: The name of the Qwen model to use.
            device: The device to use for computations.
        """
        super().__init__()
        
        self.device = device
        self.torch_dtype = torch_dtype
        self.embedding_concat_strategy = embedding_concat_strategy
        self.n_layers_per_group = n_layers_per_group
        self.offload_model_to_cpu = offload_model_to_cpu
        
        log.info("Instantiating text encoder model...")
        
        # Build processor kwargs
        processor_kwargs = {
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "use_fast": True,
        }   

        # Build model kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": "flash_attention_2",
            "device_map": "cpu" if offload_model_to_cpu else device,
        }

        if cache_dir is not None:
            processor_kwargs["cache_dir"] = cache_dir
            model_kwargs["cache_dir"] = cache_dir

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            **processor_kwargs
        )
        # Load model - Use Qwen2_5_VLForConditionalGeneration for vision-language model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Configure for embedding extraction - critical for getting hidden states
        self.model.config.output_hidden_states = True
        
        if not offload_model_to_cpu:
            self.model = self.model.to(device)
            
        self.model.eval()
        torch.cuda.empty_cache()
        log.info("Text encoder model instantiated")

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

    def compute_text_embeddings_online(
        self, data_batch: dict[str, Union[List[str], torch.Tensor]], input_caption_key: str
    ) -> torch.Tensor:
        """
        Compute text embeddings for the given prompts.
        
        Args:
            data_batch: Dictionary containing prompts
            input_caption_key: Key to extract prompts from data_batch
            
        Returns:
            Text embeddings tensor
        """
        assert self.model is not None, "Text encoder is not initialized"
        
        # Move model to GPU if offloaded
        if self.offload_model_to_cpu:
            self.model = self.model.to(self.device)
            log.debug("Moved QwenVL model to GPU")
        
        # Tokenize prompts
        input_ids_batch = []
        
        prompts = data_batch[input_caption_key]
        if isinstance(prompts, str):
            prompts = [prompts]
        
        for prompt in prompts:
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
                            "text": prompt,
                        }
                    ],
                },
            ]
            
            # Apply chat template - this is Qwen-specific tokenization
            text = self.processor.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False,
            )
            
            # Tokenize the text
            tokenizer_output = self.processor.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=NUM_EMBEDDING_PADDING_TOKENS,
                padding="max_length",
            )
            
            input_ids = tokenizer_output["input_ids"][0].to(device=self.device)
            input_ids_batch.append(input_ids)
        
        input_ids_batch = torch.stack(input_ids_batch, dim=0)
        
        # Compute text embeddings
        with torch.no_grad():
            outputs = self.model(input_ids_batch, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        
        # Now compute the normalized embeddings
        # Skip layer 0 (embeddings layer) and normalize the rest
        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            normalized_state = self.mean_normalize(hidden_states[layer_idx])
            normalized_hidden_states.append(normalized_state)
        
        text_embeddings = None
        if self.embedding_concat_strategy == str(EmbeddingConcatStrategy.FULL_CONCAT):
            # Concatenate all layer embeddings - this gives 100352-dim for 7B model
            text_embeddings = torch.cat(normalized_hidden_states, dim=-1)
        elif self.embedding_concat_strategy == str(EmbeddingConcatStrategy.MEAN_POOLING):
            # Stack the normalized hidden states and calculate the mean
            text_embeddings = torch.stack(normalized_hidden_states)
            text_embeddings = text_embeddings.mean(dim=0)
        elif self.embedding_concat_strategy == str(EmbeddingConcatStrategy.POOL_EVERY_N_LAYERS_AND_CONCAT):
            # Pool every n layers and concatenate
            n_layers_per_group = self.n_layers_per_group
            text_embeddings = []
            for i in range(0, len(normalized_hidden_states), n_layers_per_group):
                group_embeddings = normalized_hidden_states[i : i + n_layers_per_group]
                group_embedding = torch.stack(group_embeddings)
                group_embedding = group_embedding.mean(dim=0)
                text_embeddings.append(group_embedding)
            text_embeddings = torch.cat(text_embeddings, dim=-1)
        else:
            raise ValueError(f"Invalid embedding_concat_strategy: {self.embedding_concat_strategy}")
        
        # Offload model if needed
        if self.offload_model_to_cpu:
            self.model = self.model.to("cpu")
            log.debug("Offloaded QwenVL model to CPU")
        
        return text_embeddings

    @torch.inference_mode()
    def encode_prompts(
        self, prompts: Union[str, List[str]], max_length: int = 512, return_mask: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Convenience method to encode prompts in the same interface as T5 encoder.
        This wraps compute_text_embeddings_online for compatibility.
        
        Args:
            prompts: Single prompt or list of prompts to encode
            max_length: Maximum sequence length (ignored - uses NUM_EMBEDDING_PADDING_TOKENS=512 internally)
            return_mask: Whether to return attention mask along with embeddings
            
        Returns:
            Text embeddings tensor of shape [B, 512, embed_dim]
            If return_mask is True, also returns attention mask
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Create data batch in the format expected by compute_text_embeddings_online
        data_batch = {"prompts": prompts}
        embeddings = self.compute_text_embeddings_online(data_batch, "prompts")
        
        if return_mask:
            # Create a simple mask of all ones since all tokens are valid after padding
            batch_size = embeddings.shape[0]
            mask = torch.ones(batch_size, NUM_EMBEDDING_PADDING_TOKENS, dtype=torch.bool, device=embeddings.device)
            return embeddings, mask
        
        return embeddings

    def to(self, device):
        """Move the encoder to specified device."""
        self.device = device
        if not self.offload_model_to_cpu:
            self.model = self.model.to(device)
        return self