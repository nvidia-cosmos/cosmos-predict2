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

from cosmos_predict2.conditioner import CameraConditioner, BooleanFlag, ReMapkey, TextAttr
from cosmos_predict2.configs.base.config_video2world import (
    ConditioningStrategy,
    CosmosGuardrailConfig,
    CosmosReason1Config,
    Video2WorldPipelineConfig,
)
from cosmos_predict2.configs.base.config_text2image import SolverTimestampConfig
from cosmos_predict2.configs.base.defaults.ema import EMAConfig
from cosmos_predict2.models.text2image_dit import SACConfig
from cosmos_predict2.models.video2world_camera_dit import CameraConditionedMinimalV1LVGDiT
from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface
from imaginaire.lazy_config import LazyCall as L

# Cosmos Predict2 Video2World 2B Camera Conditioned
_PREDICT2_VIDEO2WORLD_NET_2B_CAMERA_CONDITIONED = L(CameraConditionedMinimalV1LVGDiT)(
    # Input/output dimensions
    max_img_h=240,  # Maximum image height in latent space (480p / 2 due to patch)
    max_img_w=240,  # Maximum image width in latent space (480p / 2 due to patch)
    max_frames=128,  # Maximum number of frames the model can handle
    in_channels=16,  # Input latent channels from VAE
    out_channels=16,  # Output latent channels to VAE
    
    # Patch settings for space-time tokenization
    patch_spatial=2,  # Spatial patch size (2x2)
    patch_temporal=1,  # Temporal patch size (1 frame)
    
    # Camera conditioning specific
    camera_dim=1536,  # Dimension for camera embeddings
    
    # Padding mask handling
    concat_padding_mask=True,  # Concatenate padding mask to input
    
    # Core transformer architecture (2B model)
    model_channels=2048,  # Hidden dimension of transformer
    num_blocks=28,  # Number of transformer blocks (28 for 2B model)
    num_heads=16,  # Number of attention heads
    atten_backend="minimal_a2a",  # Attention backend implementation
    
    # Cross-attention configuration for QwenVL text encoder
    crossattn_emb_channels=1024,  # Output dimension for cross-attention
    use_crossattn_projection=True,  # Enable projection for QwenVL embeddings
    crossattn_proj_in_channels=100352,  # QwenVL embedding dimension (28 layers × 3584 dims)
    
    # Positional embedding configuration
    pos_emb_cls="rope3d",  # Use RoPE 3D positional embeddings
    pos_emb_learnable=True,  # Make position embeddings learnable
    pos_emb_interpolation="crop",  # Interpolation method for position embeddings
    
    # AdaLN (Adaptive Layer Normalization) with LoRA
    use_adaln_lora=True,  # Use AdaLN with LoRA modulation
    adaln_lora_dim=256,  # Dimension for AdaLN LoRA
    
    # RoPE extrapolation ratios for handling different resolutions
    rope_h_extrapolation_ratio=3.0,  # Height extrapolation (480p training)
    rope_w_extrapolation_ratio=3.0,  # Width extrapolation (480p training)
    rope_t_extrapolation_ratio=1.0,  # Temporal extrapolation
    
    # Additional position embedding settings
    extra_per_block_abs_pos_emb=False,  # No extra absolute position embeddings
    rope_enable_fps_modulation=False,  # No FPS-based RoPE modulation
    
    # SAC (Spatially Adaptive Convolution) configuration
    sac_config=L(SACConfig)(
        every_n_blocks=1,  # Apply SAC every N blocks
        mode="predict2_2b_720_aggressive",  # SAC mode for 2B model
    ),
)

_PREDICT2_VIDEO2WORLD_CONDITIONER_2B_CAMERA_CONDITIONED = L(CameraConditioner)(
    # FPS (frames per second) conditioning
    fps=L(ReMapkey)(
        dropout_rate=0.0,  # No dropout for FPS conditioning
        dtype=None,  # Use default dtype
        input_key="fps",  # Input key in data dict
        output_key="fps",  # Output key for model
    ),
    
    # Padding mask for variable-length sequences
    padding_mask=L(ReMapkey)(
        dropout_rate=0.0,  # No dropout for padding mask
        dtype=None,  # Use default dtype
        input_key="padding_mask",  # Input key in data dict
        output_key="padding_mask",  # Output key for model
    ),
    
    # Text conditioning from prompts
    text=L(TextAttr)(
        dropout_rate=0.2,
        input_key=["t5_text_embeddings"],  # Input key for text embeddings
    ),
    
    # Video conditioning flag (for img2vid vs vid2vid)
    use_video_condition=L(BooleanFlag)(
        dropout_rate=0.0,  # No dropout for video condition flag
        input_key="fps",  # Derive from FPS (if FPS exists, it's a video)
        output_key="use_video_condition",  # Output key for model
    ),
    
    # Camera conditioning (unique to camera conditioned model)
    camera=L(ReMapkey)(
        dropout_rate=0.0,  # No dropout for camera parameters
        dtype=None,  # Use default dtype
        input_key="camera",  # Input key for camera data
        output_key="camera",  # Output key for model
    ),
)

_PREDICT2_VIDEO2WORLD_TOKENIZER_2B_CAMERA_CONDITIONED = L(TokenizerInterface)(
    chunk_duration=81,  # Duration of each chunk in frames
    temporal_window=16,  # Temporal window size for processing
    load_mean_std=False,  # Don't load mean/std normalization
)

_PREDICT2_VIDEO2WORLD_EMA_2B_CAMERA_CONDITIONED = L(EMAConfig)(
    enabled=False,
    rate=0.1,  # Exponential moving average decay rate
    iteration_shift=0,  # Iteration offset for EMA updates
)

PREDICT2_VIDEO2WORLD_PIPELINE_2B_CAMERA_CONDITIONED = Video2WorldPipelineConfig(
    # Video processing settings
    adjust_video_noise=True,  # Apply noise adjustment for video generation
    
    # Conditioning configuration
    conditioner=_PREDICT2_VIDEO2WORLD_CONDITIONER_2B_CAMERA_CONDITIONED,
    conditioning_strategy=ConditioningStrategy.FRAME_REPLACE,  # Replace first frames with conditional frames
    min_num_conditional_frames=1,  # Minimum 1 conditional frame (for img2vid)
    max_num_conditional_frames=2,  # Maximum 2 conditional frames (for vid2vid)
    
    # Network architecture
    net=_PREDICT2_VIDEO2WORLD_NET_2B_CAMERA_CONDITIONED,
    
    # Numerical precision
    precision="bfloat16",  # Use bfloat16 for efficient computation
    
    # Rectified flow settings
    rectified_flow_t_scaling_factor=1.0,  # Time scaling for rectified flow
    rectified_flow_loss_weight_uniform=True,  # Uniform loss weighting across timesteps
    
    # Resolution and frame settings
    resize_online=True,  # Resize inputs during inference
    resolution="720",  # 720p resolution
    state_ch=16,  # Number of latent channels
    state_t=24,  # Temporal dimension (24 for 16fps)
    
    # EMA (Exponential Moving Average) configuration
    ema=_PREDICT2_VIDEO2WORLD_EMA_2B_CAMERA_CONDITIONED,
    
    # Noise and sigma parameters
    sigma_conditional=0.0001,  # Conditional noise level
    sigma_data=1.0,  # Data sigma for score matching
    
    # Input keys for data loading
    input_video_key="video",  # Key for video input in data dict
    input_image_key="images",  # Key for image input in data dict
    
    # Tokenizer/VAE configuration
    tokenizer=_PREDICT2_VIDEO2WORLD_TOKENIZER_2B_CAMERA_CONDITIONED,
    
    # Prompt refiner configuration (disabled for camera conditioned)
    prompt_refiner_config=CosmosReason1Config(
        checkpoint_dir="/workspace/checkpoints/nvidia/Cosmos-Reason1-7B",
        offload_model_to_cpu=True,
        enabled=False,  # Disabled for camera conditioned inference
    ),
    
    # Safety guardrail configuration (disabled for camera conditioned)
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir="/workspace/checkpoints/nvidia/Cosmos-Guardrail1",
        offload_model_to_cpu=True,
        enabled=False,  # Disabled for camera conditioned inference
    ),
    
    # ODE solver timestamps configuration
    timestamps=SolverTimestampConfig(
        nfe=35,  # Number of function evaluations for ODE solver
        t_min=0.01,  # Minimum timestep
        t_max=200,  # Maximum timestep
        order=7.0,  # Order of the ODE solver
        is_forward=False,  # Use backward (denoising) timestamps
    ),
)