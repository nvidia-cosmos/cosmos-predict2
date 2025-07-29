"""Auto-deploy optimization utilities for Cosmos models.

This module provides streamlined optimization utilities for Cosmos models using 
TensorRT-LLM's auto-deploy framework with real captured DiT inputs.

Key Features:
    - Real DiT input capture and reuse for accurate compilation
    - Pipeline-specific optimization strategies (Text2Image vs Video models)
    - Automatic dynamic shape inference from configuration
    - Context-parallel aware compilation and deployment
    - Comprehensive benchmarking and performance analysis

Functions:
    ad_optimize_dit: Main entry point for DiT optimization
    detect_pipeline_type: Utility to identify pipeline type

Example:
    Basic optimization:
    ```python
    from cosmos_predict2.utils.auto_deploy import ad_optimize_dit
    
    args = argparse.Namespace()
    args.auto_deploy_backend = "torch-opt"
    args.benchmark = True
    ad_optimize_dit(pipeline, args)
    ```
"""

import os
import time
import tempfile
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.library  # For custom op registration
import torch.nn as nn
from torch.export import Dim
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from PIL import Image

from imaginaire.utils import log
from cosmos_predict2.models.text2image_dit import Attention as _Attn, torch_attention_op

# Global counters for tracking SageAttention usage
_sage_attention_success_count = 0
_sage_attention_failure_count = 0

def is_blackwell_gpu() -> bool:
    """Check if the current GPU is a Blackwell architecture GPU."""
    if not torch.cuda.is_available():
        return False
    
    try:
        # Check GPU name for Blackwell identifiers
        gpu_name = torch.cuda.get_device_name().lower()
        blackwell_identifiers = ['b200', 'b100', 'gb200', 'blackwell']
        
        if any(identifier in gpu_name for identifier in blackwell_identifiers):
            return True
            
        # Also check compute capability as fallback
        actual_capability = torch.cuda.get_device_capability()
        # Blackwell GPUs have compute capability 10.0
        return actual_capability >= (10, 0)
        
    except Exception:
        # If we can't determine GPU type, don't override
        return False

# Only override compute capability if we're on Blackwell GPU
if is_blackwell_gpu():
    print("[INFO] Detected Blackwell GPU - overriding compute capability to 10.0 for compatibility")
    torch.cuda.get_device_capability = lambda device=None: (10, 0)  # Compute capability 10.0 = sm_100

PIPELINE_TYPES = {
    "TEXT2IMAGE": "text2image",
    "TEXT2WORLD": "text2world", 
    "VIDEO2WORLD": "video2world"
}

SUPPORTED_BACKENDS = ["torch-opt", "torch-compile", "torch-cudagraph", "torch-simple"]


def get_distributed_info():
    """Get distributed environment information."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, local_rank, world_size


def detect_pipeline_type(pipe) -> str:
    """Detect pipeline type from pipeline class name.
    
    Args:
        pipe: Pipeline object
        
    Returns:
        str: Pipeline type identifier ("text2image", "text2world", "video2world")
    """
    if pipe is None:
        raise TypeError("Pipeline object cannot be None")
    
    class_name = pipe.__class__.__name__
    
    if "Text2Image" in class_name:
        return PIPELINE_TYPES["TEXT2IMAGE"]
    elif "Text2World" in class_name:
        return PIPELINE_TYPES["TEXT2WORLD"]
    elif "Video2World" in class_name:
        return PIPELINE_TYPES["VIDEO2WORLD"]
    
    # Fallback with warning
    log.warning(f"Could not detect pipeline type from {class_name}, defaulting to video2world")
    return PIPELINE_TYPES["VIDEO2WORLD"]


def get_default_dit_input_path(pipeline_type: str, backend: str = "torch-opt") -> str:
    """Get default DiT input path based on pipeline type and backend."""
    paths = {
        PIPELINE_TYPES["TEXT2IMAGE"]: f"/tmp/dit_inputs_text2image_{backend}.pt",
        PIPELINE_TYPES["TEXT2WORLD"]: f"/tmp/dit_inputs_text2world_{backend}.pt", 
        PIPELINE_TYPES["VIDEO2WORLD"]: f"/tmp/dit_inputs_video2world_{backend}.pt"
    }
    return paths.get(pipeline_type, f"/tmp/dit_inputs_{backend}.pt")


def prepare_model_for_export(model: nn.Module, sage_attention_available: bool = False) -> None:
    """Prepare model for auto-deploy optimization by patching attention with SageAttention."""
    log.info("🔧 [PATCH] Preparing model for export with SageAttention backend")
    
    # Get rank for logging
    rank = get_distributed_info()[0]
    
    # Always patch attention modules to use SageAttention
    patched_modules = 0
    
    # Use SageAttention if available, otherwise fallback to basic torch attention
    if sage_attention_available:
        attention_op = sage_attention_op
        attention_name = "sage_attention_op"
        log.info("🧠 [SAGE] Using SageAttention for optimal temporal coherence")
    else:
        attention_op = torch_attention_op
        attention_name = "torch_attention_op"
        log.info("⚠️  [SAGE] SageAttention not available, using torch_attention_op fallback")
    
    log.info(f"🔧 [PATCH] Applying {attention_name} to all attention modules")
    
    for name, mod in model.named_modules():
        if isinstance(mod, _Attn):
            original_backend = getattr(mod, "backend", "unknown")
            
            # CRITICAL: Only use SageAttention for self-attention (like early Cosmos)
            # Cross-attention has GQA incompatibility issues with SageAttention
            is_self_attention = getattr(mod, "is_selfattn", False) or "self" in name.lower()
            
            if sage_attention_available and is_self_attention:
                # Use SageAttention for self-attention layers only
                selected_op = sage_attention_op
                selected_name = "sage_attention_op"
                mod.backend = "torch"
                log.info(f"🧠 [SAGE] {name}: Using SageAttention (self-attention)")
            else:
                # Use torch attention for cross-attention or when SageAttention unavailable
                selected_op = torch_attention_op
                selected_name = "torch_attention_op"
                mod.backend = "torch"
                reason = "cross-attention" if not is_self_attention else "sage unavailable"
                log.info(f"🔧 [TORCH] {name}: Using torch attention ({reason})")
            
            if hasattr(mod, "attn_op"):
                if isinstance(mod.attn_op, torch.nn.Module):
                    delattr(mod, "attn_op")
                    mod.__dict__["attn_op"] = selected_op
                else:
                    mod.attn_op = selected_op
            if hasattr(mod, "atten_backend"):
                mod.atten_backend = "torch"
            
            patched_modules += 1
    
    log.info(f"✅ [PATCH] Successfully patched {patched_modules} attention modules")
    
    # Track usage during inference
    if sage_attention_available:
        log.info("📊 [SAGE] SageAttention is active - will log usage during inference")
    else:
        log.info("📊 [SAGE] Basic torch_attention_op fallback is active")

    # Unwrap checkpoint wrappers
    log.info("🔄 [PATCH] Unwrapping checkpoint wrappers for torch.export compatibility")
    unwrapped_modules = 0
    for name, mod in model.named_modules():
        if hasattr(mod, '_checkpoint_wrapped') or 'checkpoint' in str(type(mod)).lower():
            unwrapped_modules += 1
            log.info(f"🔄 [PATCH] Unwrapping checkpoint: {name}")
            
    log.info(f"✅ [PATCH] Model preparation complete - {patched_modules} attention modules patched, {unwrapped_modules} checkpoint wrappers unwrapped")


def benchmark_model(model: nn.Module, inputs: Dict[str, Any], num_warmup: int = 2, num_iter: int = 10) -> float:
    """Benchmark model inference latency."""
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


def compile_model(model: nn.Module, inputs: Dict[str, Any], dynamic_shapes: Tuple[Any], backend: str) -> nn.Module:
    """Compile model using auto_deploy."""
    from tensorrt_llm._torch.auto_deploy.compile import compile_and_capture
    from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export_to_gm

    # DEBUG: Log exactly what PyTorch receives  
    log.info(f"🔍 [COMPILE] All inputs to torch_export_to_gm ({len(inputs)}): {sorted(inputs.keys())}")
    if dynamic_shapes is not None:
        log.info(f"🔍 [COMPILE] Dynamic shapes tuple length: {len(dynamic_shapes)}")
    else:
        log.info(f"🔍 [COMPILE] Dynamic shapes: None (static compilation)")

    # Export model with all parameters (including gt_frames and use_video_condition)
    gm = torch_export_to_gm(
        model, 
        args=(), 
        kwargs=inputs,  # All parameters as kwargs
        clone=False, 
        dynamic_shapes=dynamic_shapes
    )
    
    log.info(f"✅ torch.export completed successfully")
    
    # Compile the exported graph module with proper args and kwargs
    compiled_model = compile_and_capture(gm, backend=backend, args=(), kwargs=inputs)
    
    log.info(f"✅ Model compilation completed with backend: {backend}")
    
    return compiled_model


def create_dynamic_shapes(inputs: Dict[str, torch.Tensor], max_frames: int) -> Tuple[Any]:
    """Create dynamic shapes for video models (only time dimension dynamic).
    
    Returns tuple format to avoid key mismatch issues with torch.export kwargs.
    The tuple order must match the order of inputs passed to the model.
    
    IMPORTANT: All tensors including gt_frames use sharded frame count since runtime provides sharded tensors.
    """
    # Use consistent sharded time dimension for all tensors
    sharded_time_dim = Dim("time_dim", min=1, max=max_frames)  # For all sharded tensors (3 frames per rank)
    
    # Define the expected order of inputs for the DiT forward method
    # This must match the parameter order in MinimalV1LVGDiT.forward()
    expected_order = [
        'x_B_C_T_H_W', 'timesteps_B_T', 'crossattn_emb',
        'condition_video_input_mask_B_C_T_H_W', 'fps', 'padding_mask',
        'data_type', 'use_cuda_graphs', 'gt_frames', 'use_video_condition'
    ]
    
    dynamic_shapes_tuple = []
    
    for key in expected_order:
        if key in inputs:
            value = inputs[key]
            if isinstance(value, torch.Tensor):
                if len(value.shape) == 5:  # B_C_T_H_W - make time dimension dynamic
                    batch_size = value.shape[0]
                    
                    # ALL 5D tensors (including gt_frames) use the same sharded time dimension
                    log.info(f"🔪 [DYNAMIC] {key} uses sharded_time_dim ({max_frames} frames): {value.shape}")
                    
                    if batch_size > 1:
                        batch_dim = Dim("batch_dim", min=1, max=batch_size)
                        # Format: {dim_index: Dim} for 5D tensors (B_C_T_H_W)
                        dynamic_shapes_tuple.append({
                            0: batch_dim,               # Batch dimension  
                            2: sharded_time_dim         # Time dimension (index 2 in B_C_T_H_W)
                        })
                    else:
                        # Batch size = 1, only time dimension is dynamic
                        dynamic_shapes_tuple.append({2: sharded_time_dim})
                elif len(value.shape) == 2:  # B_T - make time dimension dynamic
                    batch_size = value.shape[0]
                    # 2D tensors are always sharded
                    if batch_size > 1:
                        batch_dim = Dim("batch_dim", min=1, max=batch_size)
                        # Format: {dim_index: Dim} for 2D tensors (B_T)
                        dynamic_shapes_tuple.append({
                            0: batch_dim,               # Batch dimension
                            1: sharded_time_dim         # Time dimension (index 1 in B_T)
                        })
                    else:
                        # Batch size = 1, only time dimension is dynamic
                        dynamic_shapes_tuple.append({1: sharded_time_dim})
                elif len(value.shape) == 1:  # B - make batch dimension dynamic if needed
                    batch_size = value.shape[0]
                    if batch_size > 1:
                        batch_dim = Dim("batch_dim", min=1, max=batch_size)
                        dynamic_shapes_tuple.append({0: batch_dim})
                    else:
                        # Batch size = 1, static
                        dynamic_shapes_tuple.append(None)
                else:
                    # Other tensor shapes - static for now
                    dynamic_shapes_tuple.append(None)
            else:
                # Non-tensor inputs (DataType, bool, etc.) - static
                dynamic_shapes_tuple.append(None)
        else:
            # Input not present - static
            dynamic_shapes_tuple.append(None)
    
    return tuple(dynamic_shapes_tuple)


def patch_compiled_model(compiled_model: nn.Module, original_model: nn.Module) -> nn.Module:
    """Patch compiled model for pipeline compatibility with context parallelism support."""
    
    # Get original context parallelism settings
    original_cp_enabled = getattr(original_model, 'is_context_parallel_enabled', False)
    original_cp_size = getattr(original_model, 'context_parallel_size', 1)
    original_cp_group = getattr(original_model, 'context_parallel_group', None)
    
    # CRITICAL FIX: Detect if we should enable CP based on distributed environment
    # The pipeline enables CP dynamically AFTER auto-deploy, so we need to detect this
    rank, local_rank, world_size = get_distributed_info()
    should_enable_cp = world_size > 1  # Multi-GPU = needs context parallelism
    
    log.info(f"🔧 [PATCH] Original model CP state: enabled={original_cp_enabled}, size={original_cp_size}")
    log.info(f"🌍 [PATCH] Environment: world_size={world_size}, should_enable_cp={should_enable_cp}")
    
    if should_enable_cp:
        # ENABLE context parallelism for the compiled model based on environment
        log.info(f"✅ [PATCH] Enabling context parallelism for compiled model (world_size={world_size})")
        
        # Copy/set context parallelism attributes
        compiled_model.is_context_parallel_enabled = True
        compiled_model._is_context_parallel_enabled = True
        compiled_model.context_parallel_size = world_size  # Use world_size as CP size
        
        # Copy original CP group if available, otherwise it will be set by pipeline
        if original_cp_group is not None:
            compiled_model.context_parallel_group = original_cp_group
            log.info(f"📡 [PATCH] Copied context parallel group to compiled model")
        else:
            log.info("📡 [PATCH] CP group will be set by pipeline dynamically")
        
        # Add working context parallel methods
        def enable_context_parallel(cp_group_or_size=None):
            log.info("✅ [COMPILED] Context parallelism already enabled")
            # If a group is provided, update it
            if cp_group_or_size is not None:
                compiled_model.context_parallel_group = cp_group_or_size
            return True
        
        def disable_context_parallel():
            log.warning("⚠️  [COMPILED] Cannot disable context parallelism for compiled model")
            return False
        
        def set_context_parallel_group(cp_group):
            log.info(f"📡 [COMPILED] Context parallel group updated")
            compiled_model.context_parallel_group = cp_group
            return True
        
        compiled_model.enable_context_parallel = enable_context_parallel
        compiled_model.disable_context_parallel = disable_context_parallel
        compiled_model.set_context_parallel_group = set_context_parallel_group
        
        log.info("✅ [PATCH] Context parallelism enabled for compiled model")
        
    else:
        # No context parallelism needed (single GPU)
        log.info("🖥️  [PATCH] No context parallelism required (single GPU)")
        compiled_model.is_context_parallel_enabled = False
        compiled_model._is_context_parallel_enabled = False
        compiled_model.context_parallel_size = 1
        
        # Add no-op methods
        def enable_context_parallel(cp_group_or_size=None):
            log.info("📋 [COMPILED] No context parallelism available")
            return False
        
        def disable_context_parallel():
            log.info("📋 [COMPILED] Context parallelism already disabled")
            return True
        
        def set_context_parallel_group(cp_group):
            log.info("📋 [COMPILED] No context parallelism to configure")
            return False
        
        compiled_model.enable_context_parallel = enable_context_parallel
        compiled_model.disable_context_parallel = disable_context_parallel
        compiled_model.set_context_parallel_group = set_context_parallel_group
    
    return compiled_model


def capture_dit_inputs(pipe, dit_input_path: str) -> None:
    """Capture real DiT inputs by running minimal inference."""
    log.info("🎬 [CAPTURE] Starting DiT input capture process...")
    
    os.makedirs(os.path.dirname(dit_input_path), exist_ok=True)
    pipe.dit.save_input_path = dit_input_path
    
    pipeline_type = detect_pipeline_type(pipe)
    log.info(f"🎬 [CAPTURE] Pipeline type detected: {pipeline_type}")
    
    try:
        if pipeline_type == PIPELINE_TYPES["TEXT2IMAGE"]:
            # Text2Image inference
            log.info("🎬 [CAPTURE] Running Text2Image capture...")
            pipe(prompt="test capture", aspect_ratio="16:9", seed=0, num_sampling_step=1)
        else:
            # Video inference with temporary input file
            log.info("🎬 [CAPTURE] Running Video pipeline capture...")
            temp_dir = tempfile.mkdtemp()
            temp_image = os.path.join(temp_dir, "temp_input.jpg")
            
            # Create simple test image (64x64 gray)
            img_array = np.ones((64, 64, 3), dtype=np.uint8) * 128
            Image.fromarray(img_array).save(temp_image)
            
            # Run video inference with standard parameters
            pipe(
                input_path=temp_image,
                prompt="test capture", 
                aspect_ratio="16:9",
                seed=0, 
                num_sampling_step=1,
                num_conditional_frames=1
            )
                
    except Exception as e:
        log.error(f"🎬 [CAPTURE] Inference failed during input capture: {e}")
    
    # Verify inputs were successfully captured
    if not os.path.exists(dit_input_path):
        raise FileNotFoundError(f"🎬 [CAPTURE] Failed to capture DiT inputs at {dit_input_path}")
    
    log.info(f"🎬 [CAPTURE] DiT inputs captured to {dit_input_path}")
    
    # Quick verification of captured inputs
    try:
        captured_inputs = torch.load(dit_input_path, map_location='cpu', weights_only=False)
        log.info("🔍 [CAPTURE-VERIFY] Captured input summary:")
        for key, value in captured_inputs.items():
            if isinstance(value, torch.Tensor):
                log.info(f"  {key}: {value.shape} ({value.dtype})")
        
    except Exception as e:
        log.warning(f"⚠️  [CAPTURE] Could not verify captured inputs: {e}")
    
    # Reset save path
    pipe.dit.save_input_path = None


def shard_inputs_for_context_parallel(inputs: Dict[str, Any], rank: int, world_size: int) -> Dict[str, Any]:
    """Shard inputs to match context parallel distribution.
    
    In context parallelism, the time dimension is split across ranks.
    Each rank should compile with inputs that match its runtime shard.
    
    IMPORTANT: ALL tensors including gt_frames are sharded to match runtime behavior.
    The pipeline will provide sharded gt_frames during inference, so compilation must match.
    """
    if world_size == 1:
        return inputs  # No sharding needed for single GPU
    
    sharded_inputs = {}
    
    # First, determine the expected runtime frames per rank
    # In context parallelism, the model expects each rank to process frames_per_rank
    # For Text2World: 24 total frames / 8 ranks = 3 frames per rank
    expected_frames_per_rank = 3  # This is the runtime expectation for Text2World with 8 GPUs
    
    log.info(f"🎯 [RANK {rank}] Expected frames per rank for compilation: {expected_frames_per_rank}")
    
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and len(value.shape) == 5:  # B_C_T_H_W
            B, C, T, H, W = value.shape
            
            if T < expected_frames_per_rank:
                # Input has fewer frames than expected per rank
                # This means we need to pad or repeat the input to match expected size
                log.warning(f"⚠️  [RANK {rank}] Input {key} has {T} frames, but need {expected_frames_per_rank}")
                log.warning(f"   This suggests input was captured incorrectly for multi-GPU context parallelism")
                
                if T == 0:
                    # Create a dummy tensor with the correct size
                    log.warning(f"🔧 [RANK {rank}] Creating dummy {key} with {expected_frames_per_rank} frames")
                    sharded_tensor = torch.zeros(B, C, expected_frames_per_rank, H, W, dtype=value.dtype)
                else:
                    # Repeat the existing frames to reach expected size
                    log.warning(f"🔧 [RANK {rank}] Repeating {key} frames to reach {expected_frames_per_rank}")
                    repeat_factor = (expected_frames_per_rank + T - 1) // T  # Ceiling division
                    repeated = value.repeat(1, 1, repeat_factor, 1, 1)
                    sharded_tensor = repeated[:, :, :expected_frames_per_rank, :, :]
                
                sharded_inputs[key] = sharded_tensor
                log.info(f"🔪 [RANK {rank}] Fixed {key}: {value.shape} → {sharded_tensor.shape}")
                
            elif T == expected_frames_per_rank:
                # Input already has the correct size for one rank
                sharded_inputs[key] = value
                log.info(f"✅ [RANK {rank}] {key} already correctly sized: {value.shape}")
                
            else:
                # Input has more frames - shard normally (including gt_frames)
                # Calculate per-rank time slices for context parallelism
                frames_per_rank = T // world_size
                start_frame = rank * frames_per_rank
                end_frame = start_frame + frames_per_rank
                
                # Shard the time dimension for ALL 5D tensors including gt_frames
                sharded_tensor = value[:, :, start_frame:end_frame, :, :]
                sharded_inputs[key] = sharded_tensor
                
                log.info(f"🔪 [RANK {rank}] Sharded {key}: {value.shape} → {sharded_tensor.shape}")
                log.info(f"   Time slice: frames {start_frame}:{end_frame} of {T}")
            
        elif isinstance(value, torch.Tensor) and len(value.shape) == 2:  # B_T
            B, T = value.shape
            
            if T < expected_frames_per_rank:
                # Pad/repeat for 2D tensors as well
                if T == 0:
                    sharded_tensor = torch.zeros(B, expected_frames_per_rank, dtype=value.dtype)
                else:
                    repeat_factor = (expected_frames_per_rank + T - 1) // T
                    repeated = value.repeat(1, repeat_factor)
                    sharded_tensor = repeated[:, :expected_frames_per_rank]
                
                sharded_inputs[key] = sharded_tensor
                log.info(f"🔧 [RANK {rank}] Fixed {key}: {value.shape} → {sharded_tensor.shape}")
                
            elif T == expected_frames_per_rank:
                sharded_inputs[key] = value
                log.info(f"✅ [RANK {rank}] {key} already correctly sized: {value.shape}")
                
            else:
                # Shard normally for larger tensors
                frames_per_rank = T // world_size
                start_frame = rank * frames_per_rank
                end_frame = start_frame + frames_per_rank
                
                sharded_tensor = value[:, start_frame:end_frame]
                sharded_inputs[key] = sharded_tensor
                
                log.info(f"🔪 [RANK {rank}] Sharded {key}: {value.shape} → {sharded_tensor.shape}")
            
        else:
            # Non-temporal tensors or non-tensors: no sharding needed
            sharded_inputs[key] = value
    
    return sharded_inputs


def optimize_model_with_dit_inputs(pipe, backend: str = "torch-opt", benchmark: bool = False, 
                                 dit_input_path: str = None) -> None:
    """Main optimization function using real DiT inputs."""
    # Detect rank and set correct device for distributed execution
    rank, local_rank, world_size = get_distributed_info()
    
    if world_size > 1:
        log.info(f"🌍 Distributed setup detected: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        torch.cuda.set_device(local_rank)
        log.info(f"📍 Set CUDA device to local_rank={local_rank}")
    else:
        log.info("🖥️  Single GPU setup detected")
    
    # Setup SageAttention early if requested
    sage_attention_available = setup_sage_attention()
    
    # Check if we should skip auto-deploy entirely to preserve temporal coherence
    selective_compilation = os.environ.get("SELECTIVE_COMPILATION", "0") == "1"
    if selective_compilation:
        log.info("🎯 [SELECTIVE] Selective compilation enabled - SKIPPING auto-deploy entirely")
        log.info("🎯 [SELECTIVE] Running baseline model to preserve temporal coherence")
        log.info("🎯 [SELECTIVE] This means no optimization but perfect temporal quality")
        
        # Just set model to eval mode and return - no compilation
        pipe.dit.eval()
        log.info("✅ [SELECTIVE] Model optimization skipped - using baseline for temporal coherence")
        return
    
    # CRITICAL FIX: Set deterministic seeds BEFORE any model modifications
    if world_size > 1:
        log.info(f"🌍 [RANK {rank}] Setting deterministic seeds BEFORE model preparation")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, 'deterministic'):
            torch.backends.cudnn.deterministic = True
        log.info(f"🎲 [RANK {rank}] Deterministic seeds set for consistent model preparation")
    
    # Determine input path
    if dit_input_path is None:
        pipeline_type = detect_pipeline_type(pipe)
        dit_input_path = get_default_dit_input_path(pipeline_type, backend)
    
    # Load or capture inputs (two-pass system preferred)
    if os.path.exists(dit_input_path):
        log.info(f"✅ Found existing DiT inputs at: {dit_input_path}")
        log.info("📋 Using two-pass system (recommended for multi-GPU)")
        real_inputs = torch.load(dit_input_path, map_location='cpu', weights_only=False)
    else:
        log.warning(f"⚠️  DiT inputs not found at: {dit_input_path}")
        log.warning("📋 Falling back to inline capture (not recommended for multi-GPU)")
        log.info("💡 Consider running: python scripts/capture_dit_inputs_single_gpu.py --pipeline <type> --backend <backend>")
        capture_dit_inputs(pipe, dit_input_path)
        real_inputs = torch.load(dit_input_path, map_location='cpu', weights_only=False)
    
    # Debug: Print loaded input shapes
    log.info("🔍 [DEBUG] Loaded DiT input shapes:")
    for key, value in real_inputs.items():
        if isinstance(value, torch.Tensor):
            log.info(f"  {key}: {value.shape} ({value.dtype})")
        else:
            log.info(f"  {key}: {type(value)} = {value}")
    
    # CRITICAL FIX: Shard inputs for context parallel compilation
    if world_size > 1:
        log.info(f"🔪 [RANK {rank}] Sharding inputs for context-parallel compilation")
        sharded_inputs = shard_inputs_for_context_parallel(real_inputs, rank, world_size)
        log.info(f"✅ [RANK {rank}] Using sharded inputs for compilation (matches runtime)")
    else:
        sharded_inputs = real_inputs
    
    # Move inputs to GPU with rank-aware device selection
    device = f"cuda:{local_rank}" if world_size > 1 else "cuda:0"
    log.info(f"📍 Using device: {device}")
    
    gpu_inputs = {}
    for key, value in sharded_inputs.items():
        if isinstance(value, torch.Tensor):
            # Clone tensors for multi-GPU to avoid memory aliasing in CUDA graphs
            if world_size > 1:
                gpu_inputs[key] = value.clone().to(device)
            else:
                gpu_inputs[key] = value.to(device)
        else:
            gpu_inputs[key] = value
    
    log.info("🔍 [DEBUG] GPU input shapes after sharding and device move:")
    for key, value in gpu_inputs.items():
        if isinstance(value, torch.Tensor):
            log.info(f"  {key}: {value.shape} ({value.dtype}) on {value.device}")
        else:
            log.info(f"  {key}: {type(value)} = {value}")
    
    # CRITICAL: Include ALL parameters needed for compilation including temporal conditioning
    # gt_frames and use_video_condition are essential for valid video generation
    compilation_input_keys = {
        'x_B_C_T_H_W', 'timesteps_B_T', 'crossattn_emb',
        'condition_video_input_mask_B_C_T_H_W', 'fps', 'padding_mask',
        'data_type', 'use_cuda_graphs',
        # CRITICAL: These control temporal conditioning logic in the pipeline
        'gt_frames', 'use_video_condition'
    }
    
    # Filter inputs for compilation - include all necessary parameters
    filtered_gpu_inputs = {k: v for k, v in gpu_inputs.items() if k in compilation_input_keys}
    
    log.info(f"🔍 [DEBUG] Final compilation inputs: {sorted(filtered_gpu_inputs.keys())}")
    
    # Prepare model for export - simplified to use SageAttention everywhere
    log.info(f"🔧 [RANK {rank}] Preparing model for export with SageAttention")
    
    # CRITICAL: Set model to evaluation mode before compilation
    pipe.dit.eval()
    log.info(f"✅ [RANK {rank}] Model set to evaluation mode")
    
    prepare_model_for_export(pipe.dit, sage_attention_available=sage_attention_available)
    
    # CRITICAL FIX: Enable dynamic shapes for video models with context parallel support
    if world_size > 1:
        # For multi-GPU Text2World: Each rank processes 3 frames (24 total / 8 ranks = 3)
        expected_frames_per_rank = 3
        log.info(f"🎯 [RANK {rank}] Creating dynamic shapes for {expected_frames_per_rank} frames per rank (Text2World 8-GPU)")
        
        # DEBUG: Log exact input keys and count before creating dynamic shapes
        log.info(f"🔍 [DEBUG] Input keys for dynamic shapes ({len(filtered_gpu_inputs)}): {sorted(filtered_gpu_inputs.keys())}")
        
        # Re-enable dynamic shapes with correct format for multi-GPU temporal processing
        dynamic_shapes = create_dynamic_shapes(filtered_gpu_inputs, expected_frames_per_rank)
        
        log.info(f"🔍 [DEBUG] Dynamic shapes tuple length: {len(dynamic_shapes)}")
    else:
        # For single GPU: use full frame count from input
        max_frames = 24  # Default for video models
        for key, value in filtered_gpu_inputs.items():
            if isinstance(value, torch.Tensor) and len(value.shape) == 5:
                max_frames = value.shape[2]  # T dimension
                break
        
        log.info(f"🎯 Creating dynamic shapes for single GPU with max_frames={max_frames}")
        dynamic_shapes = create_dynamic_shapes(filtered_gpu_inputs, max_frames)
    
    # Compile model
    log.info(f"Compiling model...")
    log.info(f"🔨 [RANK {rank}] Compiling model on {device} (deterministic, sharded inputs)")
    compiled_model = compile_model(pipe.dit, filtered_gpu_inputs, dynamic_shapes, backend)
    
    # Patch compiled model for pipeline compatibility first
    compiled_model = patch_compiled_model(compiled_model, pipe.dit)
    
    # Verify CUDA graph consistency across ranks
    if world_size > 1:
        # Compute hash of compiled model's state for verification
        state_dict = compiled_model.state_dict()
        # Convert all tensors to bytes and hash
        import hashlib
        hasher = hashlib.md5()
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            if isinstance(tensor, torch.Tensor):
                # Handle bfloat16 tensors by converting to float32 first
                if tensor.dtype == torch.bfloat16:
                    tensor_bytes = tensor.float().cpu().numpy().tobytes()
                else:
                    tensor_bytes = tensor.cpu().numpy().tobytes()
                hasher.update(tensor_bytes)
            else:
                hasher.update(str(tensor).encode())
        model_hash = hasher.hexdigest()
        
        log.info(f"🔍 [RANK {rank}] Compiled model hash: {model_hash[:16]}...")
        
        # Gather all hashes at rank 0 for comparison
        if torch.distributed.is_initialized():
            all_hashes = [None] * world_size
            torch.distributed.all_gather_object(all_hashes, model_hash)
            
            if rank == 0:
                unique_hashes = set(all_hashes)
                if len(unique_hashes) == 1:
                    log.info(f"✅ [VERIFICATION] All ranks have IDENTICAL compiled models! Hash: {model_hash[:16]}...")
                else:
                    log.error(f"❌ [VERIFICATION] Ranks have DIFFERENT compiled models!")
                    for i, h in enumerate(all_hashes):
                        log.error(f"   Rank {i}: {h[:16]}...")
                    log.error("🚨 This will cause output regression!")
    
    # Store compilation input keys and add SAFER input filtering
    compilation_input_keys = set(filtered_gpu_inputs.keys())  # Use filtered keys for compilation
    
    # Store the original forward method before replacing it
    original_compiled_forward = compiled_model.forward
    
    def filtered_forward(*args, **kwargs):
        """Forward wrapper that filters inputs for compiled model.
        
        The compiled model now includes all parameters including gt_frames and use_video_condition.
        """
        # All parameters that the compiled model expects (including temporal conditioning)
        compilation_input_keys = {
            'x_B_C_T_H_W', 'timesteps_B_T', 'crossattn_emb',
            'condition_video_input_mask_B_C_T_H_W', 'fps', 'padding_mask',
            'data_type', 'use_cuda_graphs', 'gt_frames', 'use_video_condition'
        }
        
        # DEBUG: Log ALL incoming kwargs to see what the pipeline is actually passing
        log.info(f"🔍 [RUNTIME-ALL] Incoming kwargs ({len(kwargs)}): {sorted(kwargs.keys())}")
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                log.info(f"  {key}: {value.shape} ({value.dtype}) on {value.device}")
            else:
                log.info(f"  {key}: {type(value)} = {value}")
        
        # Filter kwargs to only include compiled parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in compilation_input_keys}
        
        # DEBUG: Log filtered kwargs to see what's being passed to compiled model
        log.info(f"🔍 [RUNTIME-FILTERED] Filtered kwargs ({len(filtered_kwargs)}): {sorted(filtered_kwargs.keys())}")
        
        # Debug log for temporal conditioning parameters (only if they exist)
        if 'gt_frames' in filtered_kwargs and filtered_kwargs['gt_frames'] is not None:
            gt_frames_tensor = filtered_kwargs['gt_frames']
            log.info(f"✅ [RUNTIME] gt_frames present: {gt_frames_tensor.shape} on {gt_frames_tensor.device}")
            # Log some statistics about gt_frames content
            log.info(f"✅ [RUNTIME] gt_frames stats: min={gt_frames_tensor.min().item():.4f}, max={gt_frames_tensor.max().item():.4f}, mean={gt_frames_tensor.mean().item():.4f}")
            # Note: gt_frames is now expected to be sharded (3 frames per rank) to match compilation
            if gt_frames_tensor.shape[2] == 3:
                log.info(f"✅ [RUNTIME] gt_frames correctly sharded to 3 frames per rank (matches compilation)")
            else:
                log.warning(f"⚠️  [RUNTIME] gt_frames has {gt_frames_tensor.shape[2]} frames, expected 3 for this rank")
        else:
            log.warning(f"⚠️  [RUNTIME] gt_frames MISSING or None in filtered kwargs!")
            
        if 'use_video_condition' in filtered_kwargs and filtered_kwargs['use_video_condition'] is not None:
            use_vid_cond = filtered_kwargs['use_video_condition']
            log.info(f"✅ [RUNTIME] use_video_condition: {use_vid_cond} (type: {type(use_vid_cond)})")
        else:
            log.warning(f"⚠️  [RUNTIME] use_video_condition MISSING or None in filtered kwargs!")
        
        # DEBUG: Log what gets excluded from compilation
        excluded_keys = set(kwargs.keys()) - compilation_input_keys
        if excluded_keys:
            log.info(f"🚫 [RUNTIME] Excluded from compilation: {sorted(excluded_keys)}")
        
        # Call the ORIGINAL compiled model's forward method (not the replaced one)
        return original_compiled_forward(*args, **filtered_kwargs)
    
    compiled_model.forward = filtered_forward
    compiled_model._compilation_input_keys = compilation_input_keys
    
    # Benchmark compiled model AFTER patching and filtering
    if benchmark and baseline_latency is not None:
        log.info("Benchmarking compiled model...")
        with torch.inference_mode():
            optimized_latency = benchmark_model(compiled_model, filtered_gpu_inputs)
        
        speedup = baseline_latency / optimized_latency if optimized_latency > 0 else 0
        log.info(f"Optimized latency: {optimized_latency:.2f} ms")
        log.info(f"Speedup: {speedup:.2f}x")
    
    # Replace original model
    pipe.dit = compiled_model
    log.info("Model optimization completed successfully!")


def ad_optimize_dit(pipe, args) -> None:
    """Unified entry point for DiT optimization with auto-deploy.

    Args:
        pipe: Pipeline object (Text2ImagePipeline, Text2WorldPipeline, Video2WorldPipeline)
        args: Argument namespace with:
            - auto_deploy_backend (str): Backend to use for optimization
            - benchmark (bool, optional): Whether to benchmark the models
            - dit_input_path (str, optional): Path to save/load DiT inputs

    Example:
        >>> args = argparse.Namespace()
        >>> args.auto_deploy_backend = "torch-opt"
        >>> args.benchmark = True
        >>> ad_optimize_dit(pipeline, args)
    """
    if not hasattr(args, "auto_deploy_backend"):
        raise ValueError("args must have 'auto_deploy_backend' attribute")
    
    backend = args.auto_deploy_backend
    benchmark = getattr(args, "benchmark", False)
    dit_input_path = getattr(args, "dit_input_path", None)
    
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Backend '{backend}' not supported. Must be one of: {SUPPORTED_BACKENDS}")
    
    try:
        log.info("Starting DiT optimization...")
        
        # Optimize model (context parallelism state handled in patch_compiled_model)
        optimize_model_with_dit_inputs(pipe, backend, benchmark, dit_input_path)
        
        log.info("DiT optimization completed successfully!")
        
    except Exception as e:
        log.error(f"DiT optimization failed: {str(e)}")
        raise RuntimeError(f"Failed to optimize DiT model: {str(e)}") from e


def setup_sage_attention():
    """Setup SageAttention as a custom op to prevent torch.export tracing."""
    try:
        from sageattention import sageattn
        
        log.info("✅ SageAttention imported successfully")
        
        # Read temporal coherence settings from environment (with video-optimized defaults)
        use_smooth_k = os.environ.get("SAGE_SMOOTH_K", "true").lower() == "true"
        use_smooth_v = os.environ.get("SAGE_SMOOTH_V", "true").lower() == "true"  # Default true for video
        sage_tensor_layout = os.environ.get("SAGE_TENSOR_LAYOUT", "HND")
        sage_is_causal = os.environ.get("SAGE_IS_CAUSAL", "false").lower() == "true"
        sage_qk_quant_gran = os.environ.get("SAGE_QK_QUANT_GRAN", "per_thread")
        sage_pv_accum_dtype = os.environ.get("SAGE_PV_ACCUM_DTYPE", "fp32+fp32")  # CRITICAL for video coherence
        
        log.info(f"🎬 [SAGE CONFIG] smooth_k={use_smooth_k}, smooth_v={use_smooth_v}")
        log.info(f"🎬 [SAGE CONFIG] pv_accum_dtype={sage_pv_accum_dtype}, qk_quant_gran={sage_qk_quant_gran}")
        log.info(f"🎬 [SAGE CONFIG] tensor_layout={sage_tensor_layout}, is_causal={sage_is_causal}")
        
        # Register SageAttention as a custom operator with video-optimized settings
        @torch.library.custom_op("sage::attention", mutates_args=())
        def scaled_dot_product_attention(
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor
        ) -> torch.Tensor:
            """Custom op wrapper for SageAttention optimized for video temporal coherence."""
            # Use video-optimized settings based on actual API
            return sageattn(
                q, k, v, 
                tensor_layout=sage_tensor_layout,
                is_causal=sage_is_causal,
                qk_quant_gran=sage_qk_quant_gran,     # per_thread for finest granularity
                pv_accum_dtype=sage_pv_accum_dtype,   # fp32 for best temporal coherence
                smooth_k=use_smooth_k,                # K matrix smoothing
                smooth_v=use_smooth_v,                # V matrix smoothing (critical for video)
                return_lse=False                      # Not needed for video generation
            )
        
        @scaled_dot_product_attention.register_fake
        def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            """Fake implementation for torch.export."""
            # Simulate output shape: (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, hidden_dim)
            batch_size, seq_len, num_heads, head_dim = q.shape
            return torch.empty(batch_size, seq_len, num_heads * head_dim, dtype=q.dtype, device=q.device)
        
        global _sage_attention_success_count, _sage_attention_failure_count
        _sage_attention_success_count = 0
        _sage_attention_failure_count = 0
        
        log.info("✅ SageAttention custom op registered successfully with video-optimized settings")
        log.info("🎬 [CRITICAL] Using pv_accum_dtype=fp32 and smooth_v=True for temporal coherence")
        return scaled_dot_product_attention
        
    except ImportError as e:
        log.warning(f"⚠️  SageAttention not available: {e}")
        return None
    except Exception as e:
        log.error(f"❌ Failed to setup SageAttention: {e}")
        return None


def sage_attention_op(q_B_S_H_D: torch.Tensor, k_B_S_H_D: torch.Tensor, v_B_S_H_D: torch.Tensor) -> torch.Tensor:
    """
    SageAttention-based attention operation using custom op for torch.export compatibility.
    
    Optimized for video temporal coherence with smooth_k=True and video-specific settings.
    Uses custom op to prevent torch.export from tracing through SageAttention internals.
    """
    global _sage_attention_success_count, _sage_attention_failure_count
    
    try:
        # Get input shapes
        B, S, H, D = q_B_S_H_D.shape
        
        # Detailed logging for debugging
        log.debug(f"🧠 [SAGE] Input tensors: q{q_B_S_H_D.shape}, k{k_B_S_H_D.shape}, v{v_B_S_H_D.shape}")
        log.debug(f"🧠 [SAGE] Input dtypes: q={q_B_S_H_D.dtype}, k={k_B_S_H_D.dtype}, v={v_B_S_H_D.dtype}")
        log.debug(f"🧠 [SAGE] Input devices: q={q_B_S_H_D.device}, k={k_B_S_H_D.device}, v={v_B_S_H_D.device}")
        
        # Convert to SageAttention's expected format: B_S_H_D -> B_H_S_D
        # Documentation: (batch_size, head_num, seq_len, head_dim) for tensor_layout="HND"
        q_B_H_S_D = q_B_S_H_D.transpose(1, 2).contiguous()
        k_B_H_S_D = k_B_S_H_D.transpose(1, 2).contiguous()
        v_B_H_S_D = v_B_S_H_D.transpose(1, 2).contiguous()
        
        log.debug(f"🧠 [SAGE] Transposed to HND format: q{q_B_H_S_D.shape}, k{k_B_H_S_D.shape}, v{v_B_H_S_D.shape}")
        
        # Use SageAttention custom op with temporal coherence settings
        # The custom op includes smooth_k=True for video temporal coherence
        attn_output_B_H_S_D = torch.ops.sage.attention(q_B_H_S_D, k_B_H_S_D, v_B_H_S_D)
        
        _sage_attention_success_count += 1
        if _sage_attention_success_count % 10 == 0:  # Log every 10 successes
            log.info(f"✅ [SAGE] SageAttention custom op working! Success count: {_sage_attention_success_count}")
        
        log.debug(f"✅ [SAGE] SageAttention custom op succeeded, output shape: {attn_output_B_H_S_D.shape}")
        
        # Convert back to original format: B_H_S_D -> B_S_H_D
        attn_output_B_S_H_D = attn_output_B_H_S_D.transpose(1, 2).contiguous()
        
        # Reshape to output format: B_S_H_D -> B_S_(H*D)
        result_B_S_HD = attn_output_B_S_H_D.reshape(B, S, H * D)
        
        log.debug(f"✅ [SAGE] Final result shape: {result_B_S_HD.shape}")
        
        return result_B_S_HD
        
    except Exception as e:
        _sage_attention_failure_count += 1
        log.warning(f"⚠️  SageAttention custom op failed #{_sage_attention_failure_count}: {type(e).__name__}: {e}")
        if _sage_attention_failure_count <= 5:  # Only show traceback for first few failures
            log.exception(f"🔍 [SAGE] Full traceback for failure #{_sage_attention_failure_count}")
        log.info(f"🔄 [SAGE] Falling back to torch_attention_op")
        return torch_attention_op(q_B_S_H_D, k_B_S_H_D, v_B_S_H_D)


def get_sage_attention_stats():
    """Get SageAttention usage statistics."""
    global _sage_attention_success_count, _sage_attention_failure_count
    total = _sage_attention_success_count + _sage_attention_failure_count
    success_rate = (_sage_attention_success_count / total * 100) if total > 0 else 0
    return {
        'success_count': _sage_attention_success_count,
        'failure_count': _sage_attention_failure_count,
        'total_calls': total,
        'success_rate': success_rate
    }
