# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""
Unit tests for the TextEncoder class.

Usage:
    pytest -s projects/cosmos/predict2/text_encoders/text_encoder_test.py --L0
    pytest -s projects/cosmos/predict2/text_encoders/text_encoder_test.py --L1
"""

import unittest

import pytest
import torch

from imaginaire.utils import log
from projects.cosmos.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy, TextEncoder, TextEncoderConfig


class TestTextEncoder(unittest.TestCase):
    """Test the TextEncoder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TextEncoderConfig(
            compute_online=True,
            embedding_concat_strategy=str(EmbeddingConcatStrategy.MEAN_POOLING),
            n_layers_per_group=2,
        )

    @pytest.mark.L0
    def test_mean_normalize(self):
        """Test the mean_normalize static method."""
        # Create a test tensor
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Apply mean normalization
        normalized = TextEncoder.mean_normalize(tensor)

        # Check that the result has the same shape
        assert normalized.shape == tensor.shape

        # Check that each row has mean close to 0 and std close to 1
        for i in range(tensor.shape[0]):
            assert abs(normalized[i].mean().item()) < 1e-6
            assert abs(normalized[i].std().item() - 1.0) < 1e-6

    @pytest.mark.L1
    def test_compute_text_embeddings_online_full_concat(self):
        """Test text embedding computation with FULL_CONCAT strategy."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create encoder with FULL_CONCAT strategy
        config = TextEncoderConfig(
            compute_online=True,
            embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
            n_layers_per_group=2,
        )
        encoder = TextEncoder(config)

        # Test data
        data_batch = {"input_caption": ["A beautiful sunset", "A cat playing"]}

        # Compute embeddings
        embeddings = encoder.compute_text_embeddings_online(data_batch, "input_caption")

        log.info(f"Embeddings shape: {embeddings.shape}")

        # Verify the output is a tensor
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.dim() == 3  # [batch_size, seq_len, hidden_dim]
        assert embeddings.shape[0] == 2  # batch_size
        assert embeddings.shape[1] == 512  # sequence length
        assert embeddings.shape[2] == 3584 * 28  # hidden dimension (num_layers * hidden_dim)

    @pytest.mark.L0
    def test_config_defaults(self):
        """Test TextEncoderConfig default values."""
        config = TextEncoderConfig()

        assert config.compute_online is False
        assert config.embedding_concat_strategy == str(EmbeddingConcatStrategy.MEAN_POOLING)
        assert config.n_layers_per_group == 5
        assert "s3://checkpoints-us-east-1/cosmos_reasoning1" in config.ckpt_path
        assert config.model_config is not None


if __name__ == "__main__":
    # Set up test environment
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Run tests
    unittest.main(verbosity=2)
