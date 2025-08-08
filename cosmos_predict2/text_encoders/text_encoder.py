from enum import Enum

import attrs
import torch

from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import instantiate as lazy_instantiate
from imaginaire.utils import log
from projects.cosmos.omni1.utils.dcp_checkpointer import load_checkpoint
from projects.cosmos.predict2.text_encoders.reason1 import QwenVLBaseModel
from projects.cosmos.reason1.configs.default.model_config_qwen import QwenModelConfig, QwenVisionConfig
from projects.cosmos.reason1.tokenizer.processor import build_tokenizer

NUM_EMBEDDING_PADDING_TOKENS = 512


class EmbeddingConcatStrategy(str, Enum):
    FULL_CONCAT = "full_concat"  # Concatenate embeddings all layers
    MEAN_POOLING = "mean_pooling"  # Average pool embeddings all layers
    POOL_EVERY_N_LAYERS_AND_CONCAT = "pool_every_n_layers_and_concat"  # Pool every n layers and concatenatenate

    def __str__(self) -> str:
        return self.value


@attrs.define(slots=False)
class TextEncoderConfig:
    """
    Config for the text encoder model
    """

    compute_online: bool = False
    embedding_concat_strategy: str = str(EmbeddingConcatStrategy.MEAN_POOLING)
    n_layers_per_group: int = 5
    ckpt_path: str = (
        "s3://checkpoints-us-east-1/cosmos_reasoning1/sft_exp500/sft_exp510_qwen7b_w_critique_n32/checkpoints/iter_000008000/model/"
    )
    s3_credential_path: str = "credentials/s3_checkpoint.secret"
    model_config: QwenVLBaseModel = L(QwenVLBaseModel)(
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


class TextEncoder:
    def __init__(self, config: TextEncoderConfig):
        self.config = config

        log.info("Instantiating text encoder model...")
        with torch.device("meta"):
            self.model = lazy_instantiate(self.config.model_config)
        self.model.to_empty(device="cuda")
        with torch.no_grad():
            self.model.init_weights()
        load_checkpoint(
            self.model,
            ckpt_dir=self.config.ckpt_path,
            checkpoint_cred=self.config.s3_credential_path,
        )
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
        self, data_batch: dict[str, torch.Tensor], input_caption_key: str
    ) -> torch.Tensor:
        """
        Compute text embeddings for the given prompts.
        """
        assert self.model is not None, "Text encoder is not initialized"

        # Tokenize prompts
        input_ids_batch = []

        for sample_idx in range(len(data_batch[input_caption_key])):
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
                            "text": data_batch[input_caption_key][sample_idx],
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


def get_reason1_embeddings(text: str):
    """
    Get reason1 embeddings for a given text.
    Output (1, seq len, d) embeddings
    """
    config = TextEncoderConfig(
        embedding_concat_strategy="full_concat",
    )
    text_encoder = TextEncoder(config)
    text_embeddings = text_encoder.compute_text_embeddings_online(
        {
            "text": [text],
        },
        "text",
    )
    return text_embeddings
