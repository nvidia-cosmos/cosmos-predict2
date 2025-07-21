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

"""Auto-deploy optimization utilities for Cosmos models."""

import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from imaginaire.utils import log


class ShapeInferer:
    """Infers tensor shapes and dynamic constraints from model configuration."""
    
    @staticmethod
    def infer_from_config(pipe) -> Dict[str, Any]:
        """Infer shape constraints from pipeline configuration."""
        config = pipe.config
        
        # Extract key config parameters
        max_frames = getattr(config.net, 'max_frames', 128)
        max_img_h = getattr(config.net, 'max_img_h', 240)
        max_img_w = getattr(config.net, 'max_img_w', 240)
        state_ch = getattr(config, 'state_ch', 16)
        state_t = getattr(config, 'state_t', 24)
        
        # Get tokenizer compression factor
        spatial_compression = getattr(pipe.tokenizer, 'spatial_compression_factor', 8)
        
        log.info(f"Config-based shape inference:")
        log.info(f"  max_frames: {max_frames}, max_img_h: {max_img_h}, max_img_w: {max_img_w}")
        log.info(f"  state_ch: {state_ch}, state_t: {state_t}")
        log.info(f"  spatial_compression: {spatial_compression}")
        
        return {
            'max_frames': max_frames,
            'max_latent_h': max_img_h,
            'max_latent_w': max_img_w, 
            'max_image_h': max_img_h * spatial_compression,
            'max_image_w': max_img_w * spatial_compression,
            'state_ch': state_ch,
            'state_t': state_t,
            'spatial_compression': spatial_compression,
        }
    
    @staticmethod
    def create_dynamic_shapes(shape_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create dynamic shape specifications from shape info."""
        from torch.export import Dim
        
        # Create dynamic dimensions following PyTorch's suggested constraints
        time_dim = Dim("time", min=1, max=shape_info['max_frames'])
        
        # For latent space tensors (follow 2*_base pattern as suggested by PyTorch)
        _height = Dim('_height', min=8, max=shape_info['max_latent_h'])
        _width = Dim('_width', min=8, max=shape_info['max_latent_w'])
        height_dim = 2 * _height  # PyTorch detected this constraint
        width_dim = 2 * _width    # PyTorch detected this constraint
        
        # For image space tensors (padding_mask operates in image space)
        _pad_height = Dim('_pad_height', min=16, max=shape_info['max_image_h'] // 2)
        _pad_width = Dim('_pad_width', min=32, max=shape_info['max_image_w'] // 2)
        pad_height_dim = 2 * _pad_height
        pad_width_dim = 2 * _pad_width
        
        return {
            'time_dim': time_dim,
            'height_dim': height_dim,
            'width_dim': width_dim,
            'pad_height_dim': pad_height_dim,
            'pad_width_dim': pad_width_dim,
        }


class TensorReducer:
    """Handles aggressive tensor size reduction for memory-efficient compilation."""
    
    @staticmethod
    def reduce_inputs(inputs: Dict[str, Any], target_scale: float = 0.25) -> Dict[str, Any]:
        """
        Reduce tensor sizes while maintaining proper scaling relationships.
        
        Args:
            inputs: Dictionary of input tensors
            target_scale: Scale factor for reduction (0.25 = reduce to 1/4 size)
        """
        reduced_inputs = {}
        original_shapes = {}
        
        for key, tensor in inputs.items():
            if not isinstance(tensor, torch.Tensor):
                reduced_inputs[key] = tensor
                continue
                
            original_shapes[key] = tensor.shape
            
            if key == "x_B_C_T_H_W":
                # Reduce temporal and spatial dims
                B, C, T, H, W = tensor.shape
                new_T = max(4, int(T * target_scale))
                new_H = max(16, int(H * target_scale))
                new_W = max(16, int(W * target_scale))
                reduced_inputs[key] = tensor[:, :, :new_T, :new_H, :new_W]
                
            elif key == "condition_video_input_mask_B_C_T_H_W":
                # Match the main tensor reduction
                B, C, T, H, W = tensor.shape
                new_T = max(4, int(T * target_scale))
                new_H = max(16, int(H * target_scale))
                new_W = max(16, int(W * target_scale))
                reduced_inputs[key] = tensor[:, :, :new_T, :new_H, :new_W]
                
            elif key == "timesteps_B_T":
                # Reduce temporal dimension to match
                B, T = tensor.shape
                new_T = max(4, int(T * target_scale))
                reduced_inputs[key] = tensor[:, :new_T]
                
            elif key == "crossattn_emb":
                # CRITICAL: Don't reduce embedding dimensions - they must match model weights
                reduced_inputs[key] = tensor
                
            elif key == "padding_mask" and tensor.dim() == 4:
                # Maintain 8x scaling relationship with main tensor
                B, C, H, W = tensor.shape
                # Calculate what the main tensor spatial dims will be after reduction
                main_shape = inputs.get("x_B_C_T_H_W")
                if main_shape is not None:
                    _, _, _, main_H, main_W = main_shape.shape
                    target_main_H = max(16, int(main_H * target_scale))
                    target_main_W = max(16, int(main_W * target_scale))
                    # padding_mask should be 8x larger to maintain scaling
                    target_H = min(H, target_main_H * 8)
                    target_W = min(W, target_main_W * 8)
                    reduced_inputs[key] = tensor[:, :, :target_H, :target_W]
                else:
                    # Fallback: reduce by target_scale
                    new_H = max(32, int(H * target_scale))
                    new_W = max(64, int(W * target_scale))
                    reduced_inputs[key] = tensor[:, :, :new_H, :new_W]
                    
            elif key == "fps":
                # Keep scalar tensors unchanged
                reduced_inputs[key] = tensor
                
            else:
                # For other tensors, reduce temporal dimension if present
                if tensor.dim() >= 3 and tensor.shape[2] > 4:
                    new_T = max(4, int(tensor.shape[2] * target_scale))
                    reduced_inputs[key] = tensor[:, :, :new_T]
                else:
                    reduced_inputs[key] = tensor
            
            # Log reduction if shape changed
            if reduced_inputs[key].shape != original_shapes[key]:
                log.info(f"Reduced {key}: {original_shapes[key]} -> {reduced_inputs[key].shape}")
        
        return reduced_inputs


class ModelOptimizer:
    """Handles the core model optimization logic."""
    
    @staticmethod
    def prepare_model(model: nn.Module) -> None:
        """Prepare model for export by patching attention backends and unwrapping checkpoints."""
        # Force torch attention backend to avoid Flash Attention issues during export
        try:
            model.atten_backend = "torch"
        except AttributeError:
            pass

        # Patch all attention modules to use torch backend
        from cosmos_predict2.models.text2image_dit import Attention as _Attn, torch_attention_op
        
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
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
        
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
        """Benchmark model latency."""
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
    def compile_model(model: nn.Module, inputs: Dict[str, Any], dynamic_shapes: Dict[str, Any], backend: str = "torch-opt"):
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
        """Add necessary attributes to compiled model for pipeline compatibility."""
        # Add pipeline compatibility attributes
        compiled_model.disable_context_parallel = lambda: None
        compiled_model._is_context_parallel_enabled = False
        compiled_model.is_context_parallel_enabled = False
        
        # Create wrapper to filter inference inputs
        original_forward = compiled_model.forward
        pipeline_only_fields = {'use_video_condition', 'gt_frames'}
        
        def filtered_forward(x_B_C_T_H_W, timesteps_B_T, **kwargs):
            # Reconstruct inputs dictionary as expected by compiled model
            inference_inputs = {
                'x_B_C_T_H_W': x_B_C_T_H_W,
                'timesteps_B_T': timesteps_B_T,
            }
            
            # Add expected fields, filtering out pipeline-only fields
            expected_fields = ['crossattn_emb', 'fps', 'padding_mask', 'condition_video_input_mask_B_C_T_H_W', 'data_type']
            for field in expected_fields:
                if field in kwargs:
                    inference_inputs[field] = kwargs[field]
            
            return original_forward(**inference_inputs)
        
        compiled_model.forward = filtered_forward
        return compiled_model


def optimize_model(pipe, input_path: str, backend: str = "torch-opt", target_scale: float = 0.25) -> None:
    """
    Main API for optimizing a model with auto_deploy.
    
    Args:
        pipe: Pipeline object containing the model to optimize
        input_path: Path to saved input tensors for compilation
        backend: Compilation backend ("torch-opt", "torch-cudagraph", etc.)
        target_scale: Scale factor for tensor size reduction during compilation
    """
    log.info(f"[Auto-Deploy] Starting model optimization with backend: {backend}")
    
    # Step 1: Load and prepare inputs
    from cosmos_predict2.pipelines.text2image import load_sample_inputs
    inputs = load_sample_inputs(input_path)
    
    # Move inputs to model device
    device = next(pipe.dit.parameters()).device
    for key, tensor in inputs.items():
        if isinstance(tensor, torch.Tensor):
            inputs[key] = tensor.to(device)
    
    # Step 2: Reduce tensor sizes for memory efficiency
    log.info("Reducing input tensor sizes for compilation...")
    inputs = TensorReducer.reduce_inputs(inputs, target_scale=target_scale)
    torch.cuda.empty_cache()
    
    # Step 3: Infer shape constraints from config
    log.info("Inferring dynamic shape constraints from model config...")
    shape_info = ShapeInferer.infer_from_config(pipe)
    dynamic_shapes_spec = ShapeInferer.create_dynamic_shapes(shape_info)
    
    # Build dynamic shapes dict using Dim.STATIC for static dimensions
    from torch.export import Dim
    
    dynamic_shapes = {
        'x_B_C_T_H_W': (Dim.STATIC, Dim.STATIC, dynamic_shapes_spec['time_dim'], dynamic_shapes_spec['height_dim'], dynamic_shapes_spec['width_dim']),
        'timesteps_B_T': (Dim.STATIC, dynamic_shapes_spec['time_dim']),
        'condition_video_input_mask_B_C_T_H_W': (Dim.STATIC, Dim.STATIC, dynamic_shapes_spec['time_dim'], dynamic_shapes_spec['height_dim'], dynamic_shapes_spec['width_dim']),
        'crossattn_emb': None,  # Keep completely static (None is still valid for entire tensors)
        'fps': None,  # Keep completely static (None is still valid for entire tensors)
        'padding_mask': (Dim.STATIC, Dim.STATIC, dynamic_shapes_spec['pad_height_dim'], dynamic_shapes_spec['pad_width_dim']),
        'data_type': None,  # Non-tensor
    }
    
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
    
    # Step 9: Replace original model
    pipe.dit = compiled_model
    log.success("[Auto-Deploy] Model optimization completed successfully!") 