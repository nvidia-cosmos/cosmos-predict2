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

"""Auto-deploy optimization utilities for Cosmos models.

This module provides optimization utilities for Cosmos models using TensorRT-LLM's
auto-deploy framework. It includes shape inference, input generation, and model
compilation functionality to accelerate inference.

Classes:
    ShapeInferer: Infers tensor shapes and dynamic constraints from model configuration
    ModelOptimizer: Handles model preparation, benchmarking, and compilation
    InputGenerator: Generates input tensors based on model configuration

Functions:
    optimize_model: Main API for optimizing a model with auto_deploy
    optimize_model_with_context_parallel: Context-parallel-aware optimization
    ad_optimize_dit: Unified entry point for DiT optimization
"""

import time
from typing import Dict, Any, Optional, Tuple, Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.export import Dim
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from imaginaire.utils import log
from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.text2image_dit import Attention as _Attn, torch_attention_op


class ShapeInferer:
    """Infers tensor shapes and dynamic constraints from model configuration.

    This class provides utilities to extract shape information from pipeline
    configurations and create dynamic shape specifications for torch.export.
    """

    @staticmethod
    def infer_from_config(pipe) -> Dict[str, Any]:
        """Infer shape constraints from pipeline configuration.

        Args:
            pipe: Pipeline object containing the model and configuration.

        Returns:
            Dict containing shape configuration parameters including:
                - max_frames: Maximum number of frames
                - max_latent_h/w: Maximum latent space dimensions
                - max_image_h/w: Maximum image space dimensions
                - state_ch: Number of state channels
                - state_t: State temporal dimension
                - spatial_compression: Compression factor between image and latent space
        """
        config = pipe.config

        # Extract key config parameters
        max_frames = getattr(config.net, "max_frames", 128)
        max_img_h = getattr(config.net, "max_img_h", 240)
        max_img_w = getattr(config.net, "max_img_w", 240)
        state_ch = getattr(config, "state_ch", 16)
        state_t = getattr(config, "state_t", 24)

        # Get tokenizer compression factor
        spatial_compression = getattr(pipe.tokenizer, "spatial_compression_factor", 8)

        log.debug(
            f"Shape inference - max_frames: {max_frames}, max_img_h: {max_img_h}, max_img_w: {max_img_w}, state_ch: {state_ch}, state_t: {state_t}, spatial_compression: {spatial_compression}"
        )

        return {
            "max_frames": max_frames,
            "max_latent_h": max_img_h,
            "max_latent_w": max_img_w,
            "max_image_h": max_img_h * spatial_compression,
            "max_image_w": max_img_w * spatial_compression,
            "state_ch": state_ch,
            "state_t": state_t,
            "spatial_compression": spatial_compression,
        }

    @staticmethod
    def create_dynamic_shapes(shape_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create dynamic shape specifications from shape info.

        Args:
            shape_info: Dictionary containing shape configuration parameters.

        Returns:
            Dict containing torch.export.Dim objects for dynamic dimensions:
                - time_dim: Dynamic time dimension
                - height_dim: Dynamic height dimension
                - width_dim: Dynamic width dimension
                - pad_height_dim: Dynamic padding height dimension
                - pad_width_dim: Dynamic padding width dimension
        """

        # Create dynamic dimensions following PyTorch's suggested constraints
        time_dim = Dim("time", min=1, max=shape_info["max_frames"])

        # For latent space tensors (follow 2*_base pattern as suggested by PyTorch)
        _height = Dim("_height", min=8, max=shape_info["max_latent_h"])
        _width = Dim("_width", min=8, max=shape_info["max_latent_w"])
        height_dim = 2 * _height  # PyTorch detected this constraint
        width_dim = 2 * _width  # PyTorch detected this constraint

        # For image space tensors (padding_mask operates in image space)
        _pad_height = Dim("_pad_height", min=16, max=shape_info["max_image_h"] // 2)
        _pad_width = Dim("_pad_width", min=32, max=shape_info["max_image_w"] // 2)
        pad_height_dim = 2 * _pad_height
        pad_width_dim = 2 * _pad_width

        return {
            "time_dim": time_dim,
            "height_dim": height_dim,
            "width_dim": width_dim,
            "pad_height_dim": pad_height_dim,
            "pad_width_dim": pad_width_dim,
        }


class ModelOptimizer:
    """Handles model preparation, benchmarking, and compilation for auto-deploy.

    This class provides utilities for preparing models for optimization,
    benchmarking performance, and compiling models using TensorRT-LLM's
    auto-deploy framework.
    """

    @staticmethod
    def prepare_model(model: nn.Module) -> None:
        """Prepare model for auto-deploy optimization.

        This method patches attention modules and unwraps checkpoint wrappers
        to make the model compatible with torch.export.

        Args:
            model: The PyTorch model to prepare for optimization.
        """
        # Force torch attention backend to avoid Flash Attention issues during export
        try:
            model.atten_backend = "torch"
        except AttributeError:
            pass

        # Patch all attention modules to use torch backend

        for name, mod in model.named_modules():
            if isinstance(mod, _Attn):
                mod.backend = "torch"
                if hasattr(mod, "attn_op"):
                    if isinstance(mod.attn_op, torch.nn.Module):
                        delattr(mod, "attn_op")
                        mod.__dict__["attn_op"] = torch_attention_op
                    else:
                        mod.attn_op = torch_attention_op
                if hasattr(mod, "atten_backend"):
                    mod.atten_backend = "torch"
                log.debug(f"Patched Attention backend on module {name}")

        # Unwrap checkpoint wrappers for export

        for i, block in enumerate(model.blocks):
            if isinstance(block, CheckpointWrapper):
                model.blocks[i] = block._checkpoint_wrapped_module

        if hasattr(model, "final_layer") and isinstance(model.final_layer, CheckpointWrapper):
            model.final_layer = model.final_layer._checkpoint_wrapped_module

        # Disable context parallelism for export
        if hasattr(model, "disable_context_parallel"):
            model.disable_context_parallel()

    @staticmethod
    def benchmark_model(model: nn.Module, inputs: Dict[str, Any], num_warmup: int = 2, num_iter: int = 10) -> float:
        """Benchmark model inference latency.

        Args:
            model: The PyTorch model to benchmark.
            inputs: Dictionary of input tensors for the model.
            num_warmup: Number of warmup iterations before timing.
            num_iter: Number of iterations to measure for latency calculation.

        Returns:
            Average latency in milliseconds across all iterations.
        """
        # Warmup
        for _ in range(num_warmup):
            _ = model.forward(**inputs)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Measure
        latencies = []
        for _ in range(num_iter):
            t0 = time.time()
            _ = model.forward(**inputs)
            torch.cuda.synchronize()
            latencies.append((time.time() - t0) * 1000)

        return np.mean(latencies)

    @staticmethod
    def compile_model(
        model: nn.Module, inputs: Dict[str, Any], dynamic_shapes: Dict[str, Any], backend: str = "torch-opt"
    ):
        """Compile model using auto_deploy."""
        from tensorrt_llm._torch.auto_deploy.compile import compile_and_capture
        from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export_to_gm

        # Export to graph module with dynamic shapes
        log.info("Exporting model to GraphModule...")
        gm = torch_export_to_gm(model, args=(), kwargs=inputs, clone=False, dynamic_shapes=dynamic_shapes)

        # Compile with specified backend
        log.info(f"Compiling with backend: {backend}")
        gm_opt = compile_and_capture(gm, backend=backend, args=(), kwargs=inputs)

        return gm_opt

    @staticmethod
    def patch_compiled_model(compiled_model, original_model):
        """Patch compiled model for pipeline compatibility and preserve model attributes."""
        # Preserve any model attributes that might be needed
        if hasattr(original_model, "context_parallel_group"):
            compiled_model.context_parallel_group = getattr(original_model, "context_parallel_group", None)

        # Add missing context parallel attributes and methods that pipelines expect
        compiled_model._is_context_parallel_enabled = getattr(original_model, "_is_context_parallel_enabled", False)
        compiled_model.is_context_parallel_enabled = getattr(original_model, "is_context_parallel_enabled", False)
        compiled_model.context_parallel_size = getattr(original_model, "context_parallel_size", 1)

        # Add context parallel methods that pipelines call
        def enable_context_parallel(cp_group_or_size=None):
            """Enable context parallelism on compiled model."""
            if cp_group_or_size is not None:
                if isinstance(cp_group_or_size, int):
                    compiled_model.context_parallel_size = cp_group_or_size
                # For ProcessGroup objects, we just enable CP
            compiled_model.is_context_parallel_enabled = True
            compiled_model._is_context_parallel_enabled = True
            log.debug(f"Context parallelism enabled on compiled model (size={compiled_model.context_parallel_size})")

        def disable_context_parallel():
            """Disable context parallelism on compiled model."""
            compiled_model.is_context_parallel_enabled = False
            compiled_model._is_context_parallel_enabled = False
            log.debug("Context parallelism disabled on compiled model")

        def set_context_parallel_group(cp_group):
            """Set context parallel group (no-op for compiled model)."""
            # The compiled model doesn't need to handle the actual group
            # The parallelism is handled at the pipeline level
            compiled_model.context_parallel_group = cp_group
            log.debug("Context parallel group set on compiled model")

        # Attach the methods to the compiled model
        compiled_model.enable_context_parallel = enable_context_parallel
        compiled_model.disable_context_parallel = disable_context_parallel
        compiled_model.set_context_parallel_group = set_context_parallel_group

        # Store the compilation input keys that the compiled model expects
        # This will be set by the optimize_model function
        compilation_input_keys = getattr(compiled_model, "_compilation_input_keys", set())

        # Create wrapper to filter inference inputs
        original_forward = compiled_model.forward
        pipeline_only_fields = {"use_video_condition", "gt_frames"}

        def filtered_forward(x_B_C_T_H_W, timesteps_B_T, **kwargs):
            # Reconstruct inputs dictionary as expected by compiled model
            inference_inputs = {
                "x_B_C_T_H_W": x_B_C_T_H_W,
                "timesteps_B_T": timesteps_B_T,
            }

            # First, add all inputs that were provided by the pipeline
            for key, value in kwargs.items():
                if key not in pipeline_only_fields:
                    inference_inputs[key] = value

            # Then, ensure all compilation keys are present with defaults if missing
            for field in compilation_input_keys:
                if field in ["x_B_C_T_H_W", "timesteps_B_T"]:
                    continue  # Already added above

                if field not in inference_inputs:
                    # Provide defaults for missing compilation inputs
                    if field == "use_cuda_graphs":
                        inference_inputs[field] = False
                    elif field == "data_type":
                        from cosmos_predict2.conditioner import DataType

                        inference_inputs[field] = DataType.VIDEO
                    elif field == "fps":
                        inference_inputs[field] = torch.tensor([16.0], device=x_B_C_T_H_W.device, dtype=torch.float32)
                    elif field == "padding_mask":
                        B, C, T, H, W = x_B_C_T_H_W.shape
                        inference_inputs[field] = torch.zeros(
                            B, 1, H * 8, W * 8, device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype
                        )
                    elif field == "crossattn_emb":
                        # This is critical and should always be provided by the pipeline
                        raise RuntimeError(
                            f"Critical field '{field}' missing during inference. This indicates a pipeline issue."
                        )
                    elif field == "condition_video_input_mask_B_C_T_H_W":
                        B, C, T, H, W = x_B_C_T_H_W.shape
                        inference_inputs[field] = torch.ones(
                            B, 1, T, H, W, device=x_B_C_T_H_W.device, dtype=x_B_C_T_H_W.dtype
                        )
                    else:
                        log.warning(f"Unknown compilation field '{field}' missing during inference, skipping")

            return original_forward(**inference_inputs)

        compiled_model.forward = filtered_forward
        return compiled_model


class InputGenerator:
    """Generates input tensors for DiT models based on pipeline configuration.

    This class provides utilities to automatically generate appropriate input
    tensors for DiT compilation by extracting dimensions and parameters from
    the pipeline configuration.
    """

    @staticmethod
    def generate_from_config(pipe, target_scale: float = 0.25) -> Dict[str, Any]:
        """Generate DiT inputs based on pipeline configuration.

        Automatically generates input tensors with appropriate shapes and dtypes
        for DiT model compilation. The input dimensions are extracted from the
        pipeline configuration and scaled according to target_scale.

        Args:
            pipe: Pipeline object containing the model and config.
            target_scale: Scale factor for tensor size reduction to improve
                compilation memory efficiency. Default: 0.25 (1/4 scale).

        Returns:
            Dictionary of input tensors including:
                - x_B_C_T_H_W: Main input tensor with latent dimensions
                - timesteps_B_T: Timestep tensor with model dtype
                - crossattn_emb: Cross-attention embedding tensor
                - fps: Frame rate tensor (always float32)
                - padding_mask: Padding mask tensor in image space
                - data_type: DataType enum (IMAGE or VIDEO)
                - use_cuda_graphs: Boolean flag for CUDA graphs
                - condition_video_input_mask_B_C_T_H_W: Condition mask for video models

        Note:
            The function automatically detects the pipeline type and generates
            appropriate inputs for Text2Image vs Video2World models.
        """
        device = next(pipe.dit.parameters()).device
        dtype = next(pipe.dit.parameters()).dtype

        # Get configuration parameters
        config = pipe.config if hasattr(pipe, "config") else None

        if config:
            # Extract from config
            max_frames = getattr(config.net, "max_frames", 128) if hasattr(config, "net") else 128
            max_img_h = getattr(config.net, "max_img_h", 240) if hasattr(config, "net") else 240
            max_img_w = getattr(config.net, "max_img_w", 240) if hasattr(config, "net") else 240
            state_ch = getattr(config, "state_ch", 16) if hasattr(config, "state_ch") else 16

            # Get the actual cross-attention embedding dimension from the DiT model
            # This is the context_dim used in the cross-attention layers
            if hasattr(pipe, "dit") and hasattr(pipe.dit, "blocks") and len(pipe.dit.blocks) > 0:
                # Get context_dim from the first block's cross-attention layer
                text_emb_dim = pipe.dit.blocks[0].cross_attn.context_dim
                log.debug(f"Using actual context_dim from model: {text_emb_dim}")
            else:
                # Fallback to config
                text_emb_dim = (
                    getattr(config.text_encoder, "width", 1024)
                    if hasattr(config, "text_encoder") and hasattr(config.text_encoder, "width")
                    else 1024
                )
                log.warning(f"Could not get context_dim from model, using fallback: {text_emb_dim}")

            text_seq_len = (
                getattr(config.text_encoder, "context_length", 512)
                if hasattr(config, "text_encoder") and hasattr(config.text_encoder, "context_length")
                else 512
            )
        else:
            # Fallback defaults - try to get from model first
            max_frames = 128
            max_img_h = 240
            max_img_w = 240
            state_ch = 16

            if hasattr(pipe, "dit") and hasattr(pipe.dit, "blocks") and len(pipe.dit.blocks) > 0:
                text_emb_dim = pipe.dit.blocks[0].cross_attn.context_dim
                log.debug(f"Using actual context_dim from model (no config): {text_emb_dim}")
            else:
                text_emb_dim = 1024  # Safe fallback
                log.warning(f"No config and no model access, using fallback context_dim: {text_emb_dim}")

            text_seq_len = 512

        # Apply target scale for compilation efficiency
        scaled_frames = max(8, int(max_frames * target_scale))
        scaled_h = max(16, int(max_img_h * target_scale))
        scaled_w = max(16, int(max_img_w * target_scale))

        log.debug(f"Generating inputs: frames={scaled_frames}, h={scaled_h}, w={scaled_w}, ch={state_ch}")

        # Determine data type based on pipeline
        if hasattr(pipe, "__class__") and "Text2Image" in pipe.__class__.__name__:
            data_type = DataType.IMAGE
        else:
            data_type = DataType.VIDEO

        # Generate input tensors
        inputs = {
            "x_B_C_T_H_W": torch.randn(1, state_ch, scaled_frames, scaled_h, scaled_w, device=device, dtype=dtype),
            "timesteps_B_T": torch.randint(0, 1000, (1, scaled_frames), device=device).to(
                dtype=dtype
            ),  # Convert to model dtype
            "crossattn_emb": torch.randn(1, text_seq_len, text_emb_dim, device=device, dtype=dtype),
            "fps": torch.tensor([16.0], device=device, dtype=torch.float32),  # fps should always be float32
            "padding_mask": torch.zeros(
                1, 1, scaled_h * 8, scaled_w * 8, device=device, dtype=dtype
            ),  # Image space padding mask
            "data_type": data_type,
            "use_cuda_graphs": False,
        }

        # Add condition mask for video2world models
        # Check if this is a Video2World pipeline by looking at pipeline type and data type
        is_video2world = (
            (hasattr(pipe, "__class__") and "Video2World" in pipe.__class__.__name__)
            or (hasattr(pipe.dit, "__class__") and "V1LGD" in pipe.dit.__class__.__name__)
            or (
                data_type == DataType.VIDEO
                and hasattr(pipe, "dit")
                and hasattr(pipe.dit, "forward")
                and "condition_video_input_mask_B_C_T_H_W" in str(pipe.dit.forward.__code__.co_varnames)
            )
        )

        if is_video2world:
            log.debug("Detected Video2World model - adding condition_video_input_mask_B_C_T_H_W")
            inputs["condition_video_input_mask_B_C_T_H_W"] = torch.ones(
                1, 1, scaled_frames, scaled_h, scaled_w, device=device, dtype=dtype
            )
        else:
            log.debug("Detected Text2Image model - no condition mask needed")

        return inputs


def optimize_model(pipe, backend: str = "torch-opt", target_scale: float = 0.25) -> None:
    """Main API for optimizing a model with auto_deploy.

    This function provides the primary interface for optimizing Cosmos models
    using TensorRT-LLM's auto-deploy framework. It handles input generation,
    model preparation, compilation, and replacement.

    Args:
        pipe: Pipeline object containing the model to optimize.
        backend: Compilation backend to use. Supported backends:
            - "torch-opt": TensorRT-LLM optimization (default)
            - "torch-compile": PyTorch 2.0 compilation
            - "torch-cudagraph": CUDA graph optimization
            - "torch-simple": Simple torch compilation
        target_scale: Scale factor for tensor size reduction during compilation.
            Smaller values use less memory but may affect optimization quality.

    Raises:
        RuntimeError: If model compilation or optimization fails.
    """
    log.info(f"Starting model optimization with backend: {backend}")

    # Step 1: Generate inputs based on config
    log.debug("Generating input tensors for compilation...")
    inputs = InputGenerator.generate_from_config(pipe, target_scale=target_scale)

    # Step 2: Infer shape constraints from config
    log.debug("Inferring dynamic shape constraints...")
    shape_info = ShapeInferer.infer_from_config(pipe)
    dynamic_shapes_dims = ShapeInferer.create_dynamic_shapes(shape_info)

    # Build dynamic shapes dict using Dim.STATIC for static dimensions
    # Only include keys that are actually present in the inputs
    dynamic_shapes = {}

    if "x_B_C_T_H_W" in inputs:
        dynamic_shapes["x_B_C_T_H_W"] = (
            Dim.STATIC,
            Dim.STATIC,
            dynamic_shapes_dims["time_dim"],
            dynamic_shapes_dims["height_dim"],
            dynamic_shapes_dims["width_dim"],
        )

    if "timesteps_B_T" in inputs:
        dynamic_shapes["timesteps_B_T"] = (Dim.STATIC, dynamic_shapes_dims["time_dim"])

    if "condition_video_input_mask_B_C_T_H_W" in inputs:
        dynamic_shapes["condition_video_input_mask_B_C_T_H_W"] = (
            Dim.STATIC,
            Dim.STATIC,
            dynamic_shapes_dims["time_dim"],
            dynamic_shapes_dims["height_dim"],
            dynamic_shapes_dims["width_dim"],
        )

    if "crossattn_emb" in inputs:
        # Make sequence dimension dynamic to handle variable text lengths
        # crossattn_emb shape is (batch, seq_len, embed_dim)
        # batch=STATIC, seq_len=DYNAMIC, embed_dim=STATIC
        text_seq_dim = Dim("text_seq_len", min=1, max=2048)  # Allow text sequences from 1 to 2048 tokens
        dynamic_shapes["crossattn_emb"] = (Dim.STATIC, text_seq_dim, Dim.STATIC)

    if "fps" in inputs:
        dynamic_shapes["fps"] = None  # Keep completely static

    if "padding_mask" in inputs:
        dynamic_shapes["padding_mask"] = (
            Dim.STATIC,
            Dim.STATIC,
            dynamic_shapes_dims["pad_height_dim"],
            dynamic_shapes_dims["pad_width_dim"],
        )

    if "data_type" in inputs:
        dynamic_shapes["data_type"] = None  # Non-tensor

    if "use_cuda_graphs" in inputs:
        dynamic_shapes["use_cuda_graphs"] = None  # Non-tensor

    log.debug(f"Dynamic shapes keys: {list(dynamic_shapes.keys())}")
    log.debug(f"Inputs keys: {list(inputs.keys())}")

    # Verify that all input keys have corresponding dynamic shapes
    missing_keys = set(inputs.keys()) - set(dynamic_shapes.keys())
    if missing_keys:
        log.warning(f"Missing dynamic shapes for keys: {missing_keys}")
        for key in missing_keys:
            dynamic_shapes[key] = None  # Add as static

    # Step 4: Prepare model for export
    log.info("Preparing model for export...")
    ModelOptimizer.prepare_model(pipe.dit)

    # Step 5: Benchmark original model
    log.info("Benchmarking original model...")
    baseline_latency = ModelOptimizer.benchmark_model(pipe.dit, inputs)
    log.success(f"Baseline latency: {baseline_latency:.2f} ms")

    # Step 6: Compile model
    log.info("Compiling model...")
    compiled_model = ModelOptimizer.compile_model(pipe.dit, inputs, dynamic_shapes, backend)

    # Step 7: Benchmark compiled model
    log.info("Benchmarking compiled model...")
    with torch.inference_mode():
        optimized_latency = ModelOptimizer.benchmark_model(compiled_model, inputs, num_warmup=2, num_iter=10)

    speedup = baseline_latency / optimized_latency
    log.success(f"Optimized latency: {optimized_latency:.2f} ms")
    log.success(f"Speedup: {speedup:.2f}x")

    # Step 8: Patch compiled model for pipeline compatibility
    log.info("Patching compiled model for pipeline compatibility...")
    compiled_model = ModelOptimizer.patch_compiled_model(compiled_model, pipe.dit)

    # Store compilation input keys for inference
    compiled_model._compilation_input_keys = set(inputs.keys())

    # Step 9: Replace original model
    pipe.dit = compiled_model

    log.success("Model optimization completed successfully!")


def optimize_model_with_context_parallel(pipe, backend: str = "torch-opt", target_scale: float = 0.25) -> None:
    """
    Optimize model while preserving context parallelism capabilities.

    This function temporarily disables context parallelism for compilation,
    then re-enables it on the compiled model for distributed inference.

    Args:
        pipe: Pipeline object containing the model to optimize
        backend: Compilation backend ("torch-opt", "torch-cudagraph", etc.)
        target_scale: Scale factor for tensor size reduction during compilation
    """
    log.info("Starting context-parallel-aware optimization...")

    # Check if context parallelism is currently enabled
    dit_model = pipe.dit
    cp_enabled = getattr(dit_model, "is_context_parallel_enabled", False)
    cp_size = getattr(dit_model, "context_parallel_size", 1) if cp_enabled else 1

    if cp_enabled:
        log.info(f"Context parallelism detected (size={cp_size}). Temporarily disabling for compilation...")

        # Store CP configuration
        cp_config = {
            "size": cp_size,
            "enabled": True,
            # Store any other CP-related attributes
        }

        # Temporarily disable context parallelism
        if hasattr(dit_model, "disable_context_parallel"):
            dit_model.disable_context_parallel()
        else:
            # Fallback: manually disable
            dit_model.is_context_parallel_enabled = False
            dit_model._is_context_parallel_enabled = False
    else:
        cp_config = None

    # Perform standard optimization on single GPU
    optimize_model(pipe, backend, target_scale)

    # Re-enable context parallelism on compiled model if it was originally enabled
    if cp_config is not None:
        log.info(f"Re-enabling context parallelism on compiled model (size={cp_config['size']})...")

        compiled_dit = pipe.dit

        # The compiled model should already have CP methods from patch_compiled_model
        # Just re-enable with the original configuration
        if hasattr(compiled_dit, "enable_context_parallel"):
            compiled_dit.enable_context_parallel(cp_config["size"])
        else:
            # Fallback: manually set attributes
            compiled_dit.context_parallel_size = cp_config["size"]
            compiled_dit.is_context_parallel_enabled = True
            compiled_dit._is_context_parallel_enabled = True

        log.success(f"✅ Compiled model ready for context-parallel inference with {cp_config['size']} GPUs!")

    log.success("Context-parallel-aware optimization completed successfully!")


def ad_optimize_dit(pipe, args) -> None:
    """Unified entry point for DiT optimization with auto-deploy.

    This function provides a unified interface for optimizing DiT models across
    different Cosmos pipelines (Text2Image, Text2World, Video2World) using
    auto-deploy with enhanced memory management and context-parallel awareness.

    Args:
        pipe: Pipeline object (Video2WorldPipeline, Text2ImagePipeline, etc.).
        args: Argument namespace containing optimization configuration:
            - auto_deploy_backend (str): Backend to use for optimization
            - auto_deploy_target_scale (float, optional): Scale factor for
              tensor size reduction (default: 0.25)

    Raises:
        ValueError: If required arguments are missing.
        RuntimeError: If optimization fails.

    Example:
        >>> args = argparse.Namespace()
        >>> args.auto_deploy_backend = "torch-opt"
        >>> args.auto_deploy_target_scale = 0.25
        >>> ad_optimize_dit(pipeline, args)
    """
    # Validate arguments
    if not hasattr(args, "auto_deploy_backend"):
        raise ValueError("args must have 'auto_deploy_backend' attribute")

    # Get target scale from args or use default
    target_scale = getattr(args, "auto_deploy_target_scale", 0.25)

    log.info(f"Starting DiT optimization with backend: {args.auto_deploy_backend}")

    # Measure optimization time
    opt_start = time.time()

    try:
        # Always use context-parallel-aware optimization (handles both single and multi-GPU cases)
        optimize_model_with_context_parallel(pipe=pipe, backend=args.auto_deploy_backend, target_scale=target_scale)

        opt_time = time.time() - opt_start
        log.success(f"DiT optimization completed in {opt_time:.2f} seconds")
    except Exception as e:
        log.error(f"DiT optimization failed: {str(e)}")
        raise RuntimeError(f"Failed to optimize DiT model: {str(e)}") from e
