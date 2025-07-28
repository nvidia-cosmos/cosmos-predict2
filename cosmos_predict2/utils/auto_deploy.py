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
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.export import Dim
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from PIL import Image

from imaginaire.utils import log
from cosmos_predict2.models.text2image_dit import Attention as _Attn, torch_attention_op

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

# Configuration Constants
SUPPORTED_BACKENDS = {"torch-opt", "torch-compile", "torch-cudagraph", "torch-simple"}
PIPELINE_TYPES = {
    "TEXT2IMAGE": "text2image",
    "TEXT2WORLD": "text2world", 
    "VIDEO2WORLD": "video2world"
}


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


def prepare_model_for_export(model: nn.Module) -> None:
    """Prepare model for auto-deploy optimization by patching attention and removing wrappers."""
    # Force torch attention backend
    try:
        model.atten_backend = "torch"
    except AttributeError:
        pass

    # Patch attention modules
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

    # Unwrap checkpoint wrappers
    for i, block in enumerate(model.blocks):
        if isinstance(block, CheckpointWrapper):
            model.blocks[i] = block._checkpoint_wrapped_module

    if hasattr(model, "final_layer") and isinstance(model.final_layer, CheckpointWrapper):
        model.final_layer = model.final_layer._checkpoint_wrapped_module

    # Disable context parallelism for export
    if hasattr(model, "disable_context_parallel"):
        model.disable_context_parallel()


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


def compile_model(model: nn.Module, inputs: Dict[str, Any], dynamic_shapes: Dict[str, Any], backend: str) -> nn.Module:
    """Compile model using auto_deploy."""
    from tensorrt_llm._torch.auto_deploy.compile import compile_and_capture
    from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export_to_gm

    gm = torch_export_to_gm(model, args=(), kwargs=inputs, clone=False, dynamic_shapes=dynamic_shapes)
    gm_opt = compile_and_capture(gm, backend=backend, args=(), kwargs=inputs)

    return gm_opt


def create_dynamic_shapes(inputs: Dict[str, torch.Tensor], max_frames: int) -> Dict[str, Any]:
    """Create dynamic shapes for video models (only time dimension dynamic)."""
    time_dim = Dim("time_dim", min=1, max=max_frames)
    dynamic_shapes = {}
    
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if len(value.shape) == 5:  # B_C_T_H_W tensors
                dynamic_shapes[key] = (Dim.STATIC, Dim.STATIC, time_dim, Dim.STATIC, Dim.STATIC)
            elif len(value.shape) == 2:  # B_T tensors
                dynamic_shapes[key] = (Dim.STATIC, time_dim)
            else:
                dynamic_shapes[key] = None  # Static for other tensors
        else:
            dynamic_shapes[key] = None
    
    return dynamic_shapes


def patch_compiled_model(compiled_model: nn.Module, original_model: nn.Module) -> nn.Module:
    """Patch compiled model for pipeline compatibility."""
    # Preserve basic context parallel attributes (but not distributed groups)
    cp_enabled = getattr(original_model, "is_context_parallel_enabled", False)
    cp_size = getattr(original_model, "context_parallel_size", 1)
    
    # Safely copy basic attributes
    compiled_model._is_context_parallel_enabled = cp_enabled
    compiled_model.is_context_parallel_enabled = cp_enabled
    compiled_model.context_parallel_size = cp_size
    
    # WARNING: Do not copy context_parallel_group as it may cause distributed issues
    # compiled_model.context_parallel_group = None  # Safer to not transfer group objects
    
    if cp_enabled:
        log.warning("⚠️  Context parallelism was enabled on original model.")
        log.warning("⚠️  Compiled model preserves CP attributes but may need group re-initialization.")
        log.warning("⚠️  Consider disabling context parallelism if you encounter distributed issues.")

    # Add simplified context parallel methods (flags only, no distributed group management)
    def enable_context_parallel(cp_group_or_size=None):
        if cp_group_or_size is not None and isinstance(cp_group_or_size, int):
            compiled_model.context_parallel_size = cp_group_or_size
        compiled_model.is_context_parallel_enabled = True
        compiled_model._is_context_parallel_enabled = True
        log.info(f"✅ Context parallelism enabled on compiled model (size={compiled_model.context_parallel_size})")

    def disable_context_parallel():
        compiled_model.is_context_parallel_enabled = False
        compiled_model._is_context_parallel_enabled = False
        log.info("❌ Context parallelism disabled on compiled model")

    def set_context_parallel_group(cp_group):
        # Allow setting but warn about potential issues
        compiled_model.context_parallel_group = cp_group
        log.warning("⚠️  Set context_parallel_group on compiled model - ensure distributed consistency")

    compiled_model.enable_context_parallel = enable_context_parallel
    compiled_model.disable_context_parallel = disable_context_parallel
    compiled_model.set_context_parallel_group = set_context_parallel_group

    return compiled_model


def capture_dit_inputs(pipe, dit_input_path: str) -> None:
    """Capture real DiT inputs by running minimal inference."""
    os.makedirs(os.path.dirname(dit_input_path), exist_ok=True)
    pipe.dit.save_input_path = dit_input_path
    
    pipeline_type = detect_pipeline_type(pipe)
    
    try:
        if pipeline_type == PIPELINE_TYPES["TEXT2IMAGE"]:
            # Text2Image inference
            pipe(prompt="test capture", aspect_ratio="16:9", seed=0, num_sampling_step=1)
        else:
            # Video inference with temporary input file
            temp_dir = tempfile.mkdtemp()
            temp_image = os.path.join(temp_dir, "temp_input.jpg")
            
            # Create simple test image (64x64 gray)
            img_array = np.ones((64, 64, 3), dtype=np.uint8) * 128
            Image.fromarray(img_array).save(temp_image)
            
            # Run video inference
            pipe(
                input_path=temp_image,
                prompt="test capture", 
                aspect_ratio="16:9",
                seed=0, 
                num_sampling_step=1,
                num_conditional_frames=1
            )
                
    except Exception as e:
        log.error(f"Inference failed during input capture: {e}")
    
    # Verify inputs were successfully captured
    if not os.path.exists(dit_input_path):
        raise FileNotFoundError(f"Failed to capture DiT inputs at {dit_input_path}")
    
    log.info(f"Captured DiT inputs to {dit_input_path}")


def optimize_model_with_dit_inputs(pipe, backend: str = "torch-opt", benchmark: bool = False, 
                                 dit_input_path: str = None) -> None:
    """Main optimization function using real DiT inputs."""
    # Detect rank and set correct device for distributed execution
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        log.info(f"🌍 Distributed setup detected: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        torch.cuda.set_device(local_rank)
        log.info(f"📍 Set CUDA device to local_rank={local_rank}")
    else:
        log.info("🖥️  Single GPU setup detected")
    
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
    
    # Move inputs to GPU with rank awareness
    if world_size > 1:
        # Use local_rank device for distributed setup
        device = torch.device(f"cuda:{local_rank}")
        log.info(f"📍 Using rank-aware device: {device}")
    else:
        # Fallback to model's device for single GPU
        device = next(pipe.dit.parameters()).device
        log.info(f"📍 Using model device: {device}")
    
    dtype = next(pipe.dit.parameters()).dtype
    gpu_inputs = {}
    for key, value in real_inputs.items():
        if isinstance(value, torch.Tensor):
            if world_size > 1:
                # Clone tensors for rank isolation to prevent memory aliasing in CUDA graphs
                gpu_inputs[key] = value.clone().to(device=device, dtype=dtype)
                log.info(f"🔄 [RANK {rank}] Cloned and moved {key} to {device}")
            else:
                # Single GPU - no need to clone
                gpu_inputs[key] = value.to(device=device, dtype=dtype)
        else:
            gpu_inputs[key] = value
    
    # Debug: Print GPU input shapes after conversion
    log.info("🔍 [DEBUG] GPU DiT input shapes:")
    for key, value in gpu_inputs.items():
        if isinstance(value, torch.Tensor):
            log.info(f"  {key}: {value.shape} ({value.dtype})")
        else:
            log.info(f"  {key}: {type(value)} = {value}")
    
    # Prepare model
    prepare_model_for_export(pipe.dit)
    
    # Create dynamic shapes
    pipeline_type = detect_pipeline_type(pipe)
    if pipeline_type == PIPELINE_TYPES["TEXT2IMAGE"]:
        dynamic_shapes = None
        log.info("Using static shapes for Text2Image")
    else:
        # Temporarily disable dynamic shapes to avoid kwargs mismatch with new parameters
        dynamic_shapes = None
        log.info("Using static shapes for video models (temporary fix for kwargs mismatch)")
    
    # Benchmark original model
    baseline_latency = None
    if benchmark:
        log.info("Benchmarking original model...")
        baseline_latency = benchmark_model(pipe.dit, gpu_inputs)
        log.info(f"Baseline latency: {baseline_latency:.2f} ms")
        log.info(f"Backend: {backend}, Dynamic shapes: {dynamic_shapes is not None}")
    
    # Compile model
    log.info("Compiling model...")
    compiled_model = compile_model(pipe.dit, gpu_inputs, dynamic_shapes, backend)
    
    if compiled_model is None:
        raise RuntimeError("Model compilation returned None")
    
    # Patch compiled model for pipeline compatibility first
    compiled_model = patch_compiled_model(compiled_model, pipe.dit)
    
    # Store compilation input keys and add input filtering
    compilation_input_keys = set(gpu_inputs.keys())
    original_forward = compiled_model.forward
    
    def filtered_forward(*args, **kwargs):
        """Filter inputs to only include keys that were used during compilation."""
        # Filter kwargs to only include compilation keys
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in compilation_input_keys}
        return original_forward(*args, **filtered_kwargs)
    
    compiled_model.forward = filtered_forward
    compiled_model._compilation_input_keys = compilation_input_keys
    
    # Benchmark compiled model AFTER patching and filtering
    if benchmark and baseline_latency is not None:
        log.info("Benchmarking compiled model...")
        with torch.inference_mode():
            optimized_latency = benchmark_model(compiled_model, gpu_inputs)
        
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
