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

import argparse
import json
import os
import numpy as np
from tensorrt_llm._torch.auto_deploy.compile import compile_and_capture
from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export_to_gm
from cosmos_predict2.pipelines.text2image import load_sample_inputs
import torch
import torch.distributed

from cosmos_predict2.pipelines.text2image import Text2ImagePipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.models.text2image_dit import torch_attention_op

# Import functionality from other example scripts
from examples.text2image import process_single_generation as process_single_image_generation
from examples.text2image import setup_pipeline as setup_text2image_pipeline
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT, cleanup_distributed
from examples.video2world import process_single_generation as process_single_video_generation
from examples.video2world import setup_pipeline as setup_video2world_pipeline
from imaginaire.utils import log
import time

_DEFAULT_POSITIVE_PROMPT = "An autonomous welding robot arm operating inside a modern automotive factory, sparks flying as it welds a car frame with precision under bright overhead lights."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text to World Generation with Cosmos Predict2")
    # Common arguments between text2image and video2world
    parser.add_argument(
        "--model_size",
        choices=["2B", "14B"],
        default="2B",
        help="Size of the model to use for text2world generation",
    )
    parser.add_argument("--prompt", type=str, default=_DEFAULT_POSITIVE_PROMPT, help="Text prompt for generation")
    parser.add_argument(
        "--batch_input_json",
        type=str,
        default=None,
        help="Path to JSON file containing batch inputs. Each entry should have 'prompt' and 'output_video' fields.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative text prompt for video2world generation",
    )
    parser.add_argument(
        "--aspect_ratio",
        choices=["1:1", "4:3", "3:4", "16:9", "9:16"],
        default="16:9",
        type=str,
        help="Aspect ratio of the generated output (width:height)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/generated_video.mp4",
        help="Path to save the generated video (include file extension)",
    )
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference for both text2image and video2world parts",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the generation in benchmark mode. It means that generation will be rerun a few times and the average generation time will be shown.",
    )

    # Text2image specific arguments
    parser.add_argument("--use_cuda_graphs", action="store_true", help="Use CUDA Graphs for the text2image inference.")

    # Video2world specific arguments
    parser.add_argument(
        "--resolution",
        choices=["480", "720"],
        default="720",
        type=str,
        help="Resolution of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--fps",
        choices=[10, 16],
        default=16,
        type=int,
        help="FPS of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--dit_path_text2image",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained text2image models.",
    )
    parser.add_argument(
        "--dit_path_video2world",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained video2world models.",
    )
    parser.add_argument(
        "--load_ema",
        action="store_true",
        help="Use EMA weights for generation.",
    )
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value for video generation")
    parser.add_argument("--offload_guardrail", action="store_true", help="Offload guardrail to CPU to save GPU memory")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )
    parser.add_argument(
        "--offload_prompt_refiner", action="store_true", help="Offload prompt refiner to CPU to save GPU memory"
    )
    parser.add_argument(
        "--natten",
        action="store_true",
        help="Run the sparse attention variant (with NATTEN).",
    )
    # --- Auto-Deploy helper flags (Text2Video DiT) ---
    parser.add_argument(
        "--save_dit_input_path",
        type=str,
        default=None,
        help="Path to save the DiT input dict (for later compilation).",
    )
    parser.add_argument(
        "--load_dit_input_path",
        type=str,
        default=None,
        help="Path to a saved DiT input dict; if provided, the script compiles the Video2World DiT with auto-deploy before generation.",
    )
    return parser.parse_args()


def generate_first_frames(text2image_pipe: Text2ImagePipeline, args: argparse.Namespace) -> list:
    """
    Generate first frames using the text2image pipeline.
    Returns a list of batch items containing prompt, output video path, and temp image path.
    """
    from megatron.core import parallel_state

    from imaginaire.utils.distributed import barrier, get_rank

    batch_items = []

    # Check if we're in a multi-GPU distributed environment
    is_distributed = parallel_state.is_initialized() and torch.distributed.is_initialized()
    rank = get_rank() if is_distributed else 0

    # Only rank 0 should run text2image generation to avoid OOM when CP is disabled
    if rank == 0 and text2image_pipe is not None:
        if args.batch_input_json is not None:
            # Process batch inputs from JSON file
            log.info(f"Loading batch inputs from JSON file: {args.batch_input_json}")
            with open(args.batch_input_json, "r") as f:
                batch_inputs = json.load(f)

            # Generate all the first frames first
            for idx, item in enumerate(batch_inputs):
                log.info(f"Generating first frame {idx + 1}/{len(batch_inputs)}")
                prompt = item.get("prompt", "")
                output_video = item.get("output_video", f"output_{idx}.mp4")

                if not prompt:
                    log.warning(f"Skipping item {idx}: Missing prompt")
                    continue

                # Save the generated first frame with a temporary name based on the output video path
                temp_image_name = os.path.splitext(output_video)[0] + "_temp.jpg"

                # Use the imported process_single_image_generation function
                if process_single_image_generation(
                    pipe=text2image_pipe,
                    prompt=prompt,
                    output_path=temp_image_name,
                    negative_prompt=args.negative_prompt,
                    aspect_ratio=args.aspect_ratio,
                    seed=args.seed,
                    use_cuda_graphs=args.use_cuda_graphs,
                    benchmark=args.benchmark,
                ):
                    # Save the item for the second stage
                    batch_items.append(
                        {"prompt": prompt, "output_video": output_video, "temp_image_path": temp_image_name}
                    )
        else:
            # Single item processing
            temp_image_path = os.path.splitext(args.save_path)[0] + "_temp.jpg"

            if args.use_cuda_graphs:
                log.warning(
                    "Using CUDA Graphs for a single inference call may not be beneficial because of overhead of Graphs creation."
                )

            # Use the imported process_single_image_generation function
            if process_single_image_generation(
                pipe=text2image_pipe,
                prompt=args.prompt,
                output_path=temp_image_path,
                negative_prompt=args.negative_prompt,
                aspect_ratio=args.aspect_ratio,
                seed=args.seed,
                use_cuda_graphs=args.use_cuda_graphs,
                benchmark=args.benchmark,
            ):
                # Add single item to batch_items for consistent processing
                batch_items.append(
                    {"prompt": args.prompt, "output_video": args.save_path, "temp_image_path": temp_image_path}
                )

        log.info(f"Rank 0: Generated {len(batch_items)} first frames")
    else:
        # Non-rank-0 processes: just wait for broadcast
        log.info(f"Rank {rank}: Waiting for batch_items from rank 0")
        batch_items = []  # Initialize empty list for non-rank-0 processes

    # Broadcast batch_items from rank 0 to all other ranks using PyTorch's broadcast_object_list
    if is_distributed:
        batch_items_list = [batch_items]  # Wrap in list for broadcast_object_list
        torch.distributed.broadcast_object_list(batch_items_list, src=0)
        batch_items = batch_items_list[0]  # Extract the broadcasted list

        if rank != 0:
            log.info(f"Rank {rank}: Received {len(batch_items)} batch items from rank 0")

        barrier()
        log.info(f"Rank {rank}: Synchronized after batch_items broadcast")

    return batch_items


def generate_videos(video2world_pipe: Video2WorldPipeline, batch_items: list, args: argparse.Namespace) -> None:
    """
    Generate videos from first frames using the video2world pipeline.
    """
    # Process all items for video generation
    for idx, item in enumerate(batch_items):
        log.info(f"Generating video from first frame {idx + 1}/{len(batch_items)}")
        prompt = item["prompt"]
        output_video = item["output_video"]
        temp_image_path = item["temp_image_path"]

        # Use the imported process_single_video_generation function
        process_single_video_generation(
            pipe=video2world_pipe,
            input_path=temp_image_path,
            prompt=prompt,
            output_path=output_video,
            negative_prompt=args.negative_prompt,
            aspect_ratio=args.aspect_ratio,
            num_conditional_frames=1,  # Always use 1 frame for text2world
            guidance=args.guidance,
            seed=args.seed,
            benchmark=args.benchmark,
            use_cuda_graphs=args.use_cuda_graphs,
        )

        # Clean up the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            log.success(f"Cleaned up temporary image: {temp_image_path}")


# ---------------------------------------------------------------------------
# Auto-Deploy support for the Video2World DiT inside text2world pipeline
# ---------------------------------------------------------------------------

def ad_optimize_t2v_dit(pipe: Video2WorldPipeline, args: argparse.Namespace):
    """Compile & optimise the Video2World DiT with torch.export + torch-opt backend."""

    print("[Auto-Deploy] Starting Video2World DiT optimization…")
    
    # Set environment variable to handle memory fragmentation
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Clear GPU cache before optimization
    torch.cuda.empty_cache()
    
    dit_model = pipe.dit

    print("type(dit_model)", type(dit_model))

    # Ensure Flash-Attention kernels are not invoked during export.
    try:
        dit_model.atten_backend = "torch"
    except AttributeError:
        pass

    from cosmos_predict2.models.text2image_dit import Attention as _Attn

    for name, mod in dit_model.named_modules():
        if isinstance(mod, _Attn):
            mod.backend = "torch"
            if hasattr(mod, "attn_op"):
                if isinstance(mod.attn_op, torch.nn.Module):
                    # Remove the existing sub-module so _modules dict accepts a plain function replacement
                    delattr(mod, "attn_op")
                    mod.__dict__["attn_op"] = torch_attention_op
                else:
                    mod.attn_op = torch_attention_op
                # Some Attention wrappers stash the backend string under a different name
                if hasattr(mod, "atten_backend"):
                    mod.atten_backend = "torch"
            log.debug(f"Patched Attention backend on module {name}")

    # Unwrap checkpoint wrappers
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    for i, block in enumerate(dit_model.blocks):
        if isinstance(block, CheckpointWrapper):
            dit_model.blocks[i] = block._checkpoint_wrapped_module

    if hasattr(dit_model, "final_layer") and isinstance(dit_model.final_layer, CheckpointWrapper):
        dit_model.final_layer = dit_model.final_layer._checkpoint_wrapped_module

    # Turn off context parallel if present
    if hasattr(dit_model, "disable_context_parallel"):
        dit_model.disable_context_parallel()

    # Prepare inputs
    inputs = load_sample_inputs(args.load_dit_input_path)
    
    # Move inputs to GPU if they're on CPU (to avoid device mismatch)
    device = next(dit_model.parameters()).device
    for key, tensor in inputs.items():
        if isinstance(tensor, torch.Tensor):
            inputs[key] = tensor.to(device)
    
    # Check available GPU memory and aggressively reduce input size for compilation
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    allocated_memory = torch.cuda.memory_allocated(device) / (1024**2)
    print(f"Current GPU memory allocated: {allocated_memory:.0f} MB")
    
    # Always reduce input tensor sizes for compilation to save memory
    # The exact tensor sizes don't matter for graph capture, only the shapes and operations
    log.info("Reducing input tensor sizes for efficient compilation...")
    original_shapes = {}
    
    for key, tensor in inputs.items():
        if isinstance(tensor, torch.Tensor):
            original_shapes[key] = tensor.shape
            
            # Aggressively reduce tensor sizes for compilation
            # IMPORTANT: Keep spatial dimensions consistent between related tensors
            if key == "x_B_C_T_H_W":
                # Reduce from [1, 16, 24, 88, 160] to [1, 16, 8, 32, 64] - much smaller
                target_h, target_w = 32, 64
                inputs[key] = tensor[:, :, :8, :target_h, :target_w]
            elif key == "condition_video_input_mask_B_C_T_H_W":
                # Match the reduced x_B_C_T_H_W dimensions exactly
                inputs[key] = tensor[:, :, :8, :32, :64]
            elif key == "timesteps_B_T":
                # Reduce temporal dimension to match
                inputs[key] = tensor[:, :8]
            elif key == "crossattn_emb":
                # CRITICAL: Don't reduce crossattn_emb sequence dimension - it must match the model's linear layer weights
                # The shape [1, 512, 1024] means: batch=1, seq_len=512, embed_dim=1024
                # The embed_dim (1024) MUST match the model's k_proj weight matrix input size
                # We can only safely reduce batch size, not sequence or embedding dimensions
                pass  # Keep crossattn_emb unchanged to avoid matrix multiplication errors
            elif key == "padding_mask" and tensor.dim() == 4:
                # CRITICAL: padding_mask spatial dimensions might have a fixed relationship to x_B_C_T_H_W
                # Original: padding_mask [1, 1, 704, 1280], x_B_C_T_H_W [1, 16, 24, 88, 160]
                # Ratio: 704/88 = 8, 1280/160 = 8 (8x scaling factor)
                # During compilation: x_B_C_T_H_W -> [1, 16, 8, 32, 64]
                # So padding_mask should -> [1, 1, 8*32, 8*64] = [1, 1, 256, 512] to maintain the 8x ratio
                target_h, target_w = 32 * 8, 64 * 8  # Maintain 8x scaling relationship
                inputs[key] = tensor[:, :, :target_h, :target_w]  # [1, 1, 256, 512]
            elif tensor.dim() >= 3 and tensor.shape[2] > 8:
                # For any other temporal tensors, reduce T dimension (but skip crossattn_emb)
                if key != "crossattn_emb":
                    inputs[key] = tensor[:, :, :8]
                
            if tensor.shape != inputs[key].shape:
                log.info(f"Reduced {key}: {original_shapes[key]} -> {inputs[key].shape}")
    
    # Force garbage collection and cache clearing
    del original_shapes
    gc.collect()
    torch.cuda.empty_cache()
    
    # Check memory usage after reduction
    allocated_after = torch.cuda.memory_allocated(device) / (1024**2)
    print(f"GPU memory after tensor reduction: {allocated_after:.0f} MB")

    # Baseline latency (reduced iterations to save memory)
    for _ in range(2):  # Reduced warmup iterations
        _ = dit_model.forward(**inputs)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # Clear cache between warmup and measurement
    
    lat_before = []
    for _ in range(10):  # Reduced measurement iterations
        t0 = time.time(); _ = dit_model.forward(**inputs); torch.cuda.synchronize(); lat_before.append((time.time()-t0)*1000)
    avg_before = np.mean(lat_before)
    
    # Clear intermediate results
    torch.cuda.empty_cache()
    log.success(f"Average latency before optimisation: {avg_before:.2f} ms")

    # Export & compile
    print(f"Compilation input structure: {list(inputs.keys())}")
    print(f"Input shapes: {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in inputs.items()]}")
    
    # Force dynamic shapes to handle variable temporal AND spatial dimensions during inference
    print("Using dynamic shapes with PyTorch's suggested constraints...")
    from torch.export import Dim
    time_dim = Dim("time", min=1, max=120)     # Temporal dimension
    # Use PyTorch's suggested height/width constraints: height = 2*_height, width = 2*_width
    _height = Dim('_height', min=8, max=120)   # Base height dimension for latent tensors
    _width = Dim('_width', min=8, max=120)     # Base width dimension for latent tensors
    height_dim = 2 * _height                   # height = 2*_height (latent space)
    width_dim = 2 * _width                     # width = 2*_width (latent space)
    
    # padding_mask has a different scaling (8x) compared to latent tensors, so needs separate dims
    _pad_height = Dim('_pad_height', min=16, max=640)  # Base padding height (8x larger than latent)
    _pad_width = Dim('_pad_width', min=32, max=640)    # Base padding width (8x larger than latent)
    pad_height_dim = 2 * _pad_height           # padding height = 2*_pad_height
    pad_width_dim = 2 * _pad_width             # padding width = 2*_pad_width
    
    dynamic_shapes = {
        'x_B_C_T_H_W': (None, None, time_dim, height_dim, width_dim),  # T, H, W are dynamic (latent space)
        'timesteps_B_T': (None, time_dim),  # Only T is dynamic
        'condition_video_input_mask_B_C_T_H_W': (None, None, time_dim, height_dim, width_dim),  # T, H, W are dynamic (latent space)
        'crossattn_emb': None,  # Keep completely static
        'fps': None,  # Keep static
        'padding_mask': (None, None, pad_height_dim, pad_width_dim),  # H, W are dynamic (image space, separate scaling)
        'data_type': None,  # Non-tensor
    }
    
    print(f"Using dynamic shapes: {dynamic_shapes}")
    gm = torch_export_to_gm(dit_model, args=(), kwargs=inputs, clone=False, dynamic_shapes=dynamic_shapes)
    gm_opt = compile_and_capture(gm, backend="torch-opt", args=(), kwargs=inputs)

    # Post-opt latency (reduced iterations to save memory)
    with torch.inference_mode():
        for _ in range(2): _ = gm_opt.forward(**inputs); torch.cuda.synchronize()  # Reduced warmup
        torch.cuda.empty_cache()
    lat_after = []
    with torch.inference_mode():
        for _ in range(10):  # Reduced measurement iterations
            t0 = time.time(); _ = gm_opt.forward(**inputs); torch.cuda.synchronize(); lat_after.append((time.time()-t0)*1000)
    avg_after = np.mean(lat_after)
    log.success(f"Average latency after optimisation: {avg_after:.2f} ms")
    log.success(f"Speed-up: {avg_before/avg_after:.2f}x")

    # Patch helper & replace
    gm_opt.disable_context_parallel = lambda: None
    gm_opt._is_context_parallel_enabled = False  # Private state variable
    gm_opt.is_context_parallel_enabled = False   # Public interface (simplified for compiled model)
    
    # Create a wrapper that matches the exact calling signature used during compilation
    original_forward = gm_opt.forward
    
    def filtered_forward(x_B_C_T_H_W, timesteps_B_T, **kwargs):
        print(f"Inference call - Positional args: x_B_C_T_H_W.shape={x_B_C_T_H_W.shape}, timesteps_B_T.shape={timesteps_B_T.shape}")
        print(f"Inference call - Keyword args: {list(kwargs.keys())}")
        
        # Reconstruct the inputs dictionary exactly as it was during compilation
        # The compiled model expects: args=(), kwargs=full_input_dict
        inference_inputs = {
            'x_B_C_T_H_W': x_B_C_T_H_W,
            'timesteps_B_T': timesteps_B_T,
        }
        
        # Add the expected keyword arguments
        expected_fields = ['crossattn_emb', 'fps', 'padding_mask', 'condition_video_input_mask_B_C_T_H_W', 'data_type']
        for field in expected_fields:
            if field in kwargs:
                inference_inputs[field] = kwargs[field]
        
        print(f"Calling compiled model with kwargs: {list(inference_inputs.keys())}")
        
        # Call the compiled model with the exact same signature as during compilation: args=(), kwargs=inputs
        return original_forward(**inference_inputs)
    
    gm_opt.forward = filtered_forward
    pipe.dit = gm_opt
    print("[Auto-Deploy] Video2World DiT optimisation completed")


if __name__ == "__main__":
    args = parse_args()
    try:
        from megatron.core import parallel_state

        from imaginaire.utils.distributed import get_rank

        if args.benchmark:
            log.warning(
                "Running in benchmark mode. Each generation will be rerun a couple of times and the average generation time will be shown."
            )

        # Check if we're in a multi-GPU distributed environment
        is_distributed = parallel_state.is_initialized() and torch.distributed.is_initialized()
        rank = get_rank() if is_distributed else 0

        # Step 1: Initialize text2image pipeline and generate all first frames
        # Only rank 0 initializes the text2image pipeline to avoid OOM
        text2image_pipe = None
        text_encoder = None

        log.info("Step 1: Initializing text2image pipeline...")
        args.dit_path = args.dit_path_text2image
        text2image_pipe = setup_text2image_pipeline(args)

        # Handle the case where setup_text2image_pipeline returns None for non-rank-0 processes
        if text2image_pipe is not None:
            # Store text encoder for later use (only on rank 0)
            text_encoder = text2image_pipe.text_encoder
            log.info("Rank 0: Text2image pipeline initialized successfully")
        else:
            # Non-rank-0 processes get None
            text_encoder = None
            log.info(f"Rank {rank}: Text2image pipeline setup returned None (expected for non-rank-0)")

        # Generate first frames (only rank 0 does actual generation)
        log.info("Step 1: Generating first frames...")
        batch_items = generate_first_frames(text2image_pipe, args)

        # Clean up text2image pipeline on rank 0
        if text2image_pipe is not None:
            log.info("Step 1 complete. Cleaning up text2image pipeline to free memory...")
            del text2image_pipe
            torch.cuda.empty_cache()

        # Step 2: Initialize video2world pipeline and generate videos
        log.info("Step 2: Initializing video2world pipeline...")

        # For non-rank-0 processes, let video2world create its own text encoder
        # This avoids the complexity of broadcasting the text encoder object across ranks
        if is_distributed and rank != 0:
            text_encoder = None
            log.info(f"Rank {rank}: Will create new text encoder for video2world pipeline")

        # Pass all video2world relevant arguments and the text encoder
        args.dit_path = args.dit_path_video2world
        video2world_pipe = setup_video2world_pipeline(args, text_encoder=text_encoder)

        # ------------------------------------------------------------------
        # Optional: compile or capture DiT inputs
        # ------------------------------------------------------------------
        if args.load_dit_input_path:
            # Temporarily offload some pipeline components to save memory during optimization
            log.info("Temporarily offloading pipeline components for auto_deploy optimization...")
            
            # Store components that can be restored later
            backup_components = {}
            if hasattr(video2world_pipe, 'text_encoder') and video2world_pipe.text_encoder is not None:
                backup_components['text_encoder'] = video2world_pipe.text_encoder
                video2world_pipe.text_encoder = None
            
            if hasattr(video2world_pipe, 'vae') and video2world_pipe.vae is not None:
                backup_components['vae'] = video2world_pipe.vae
                video2world_pipe.vae = None
                
            if hasattr(video2world_pipe, 'guardrail') and video2world_pipe.guardrail is not None:
                backup_components['guardrail'] = video2world_pipe.guardrail
                video2world_pipe.guardrail = None
            
            # Clear cache after offloading
            torch.cuda.empty_cache()
            log.info("Starting auto_deploy optimization with reduced memory footprint...")
            
            try:
                ad_optimize_t2v_dit(video2world_pipe, args)
                log.success("Auto_deploy optimization completed successfully!")
            finally:
                # Restore components
                log.info("Restoring pipeline components after optimization...")
                for attr_name, component in backup_components.items():
                    setattr(video2world_pipe, attr_name, component)
                torch.cuda.empty_cache()
                
        elif args.save_dit_input_path:
            # Set the path to save inputs during the next inference run
            video2world_pipe.dit.set_save_input_path(args.save_dit_input_path)
            log.info(f"Video2World DiT will save inputs to {args.save_dit_input_path} during the next forward pass")

        # Generate videos
        log.info("Step 2: Generating videos from first frames...")
        generate_videos(video2world_pipe, batch_items, args)

        # Clean up video2world pipeline
        log.info("All done. Cleaning up video2world pipeline...")
        del video2world_pipe
        torch.cuda.empty_cache()

    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()
