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

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch
import torch.distributed

from cosmos_predict2.pipelines.video2video_sdedit import Text2ImageSDEditPipeline, Video2WorldSDEditPipeline
from cosmos_predict2.configs.base.config_video2world import (
    PREDICT2_VIDEO2WORLD_PIPELINE_2B,
    PREDICT2_VIDEO2WORLD_PIPELINE_14B,
    PREDICT2_VIDEO2WORLD_WITH_NATTEN_PIPELINE_2B,
    PREDICT2_VIDEO2WORLD_WITH_NATTEN_PIPELINE_14B,
)
from cosmos_predict2.configs.base.config_text2image import (
    PREDICT2_TEXT2IMAGE_PIPELINE_2B,
    PREDICT2_TEXT2IMAGE_PIPELINE_14B,
)

# Import functionality from other example scripts
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT, cleanup_distributed, validate_input_file

from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video, save_text_prompts
from imaginaire.utils.easy_io import easy_io

_DEFAULT_POSITIVE_PROMPT = "A point-of-view video shot from inside a vehicle, capturing a snowy suburban street in the winter filled with snow on the side of the road."


import os, textwrap, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

def concatenate_videos_with_title(output_video_path: str,
                                  input_video_path: str,
                                  title: str,
                                  fps: int = 16):
    """Concatenate two videos horizontally and add a multi-line title above them."""
    try:
        log.info("Loading videos…")
        out_frames, _ = easy_io.load(output_video_path)
        in_frames,  _ = easy_io.load(input_video_path)

        # Determine target frame size
        out_h, out_w = out_frames.shape[1:3]
        in_h,  in_w  =  in_frames.shape[1:3]
        content_h = max(out_h, in_h)
        content_w = out_w + in_w
        
        # Ensure dimensions are even for H.264 compatibility
        if content_h % 2 != 0:
            content_h += 1
        if content_w % 2 != 0:
            content_w += 1

        # Setup font and wrap title text
        try:
            font = ImageFont.load_default(size=28)
        except Exception:
            font = ImageFont.load_default()

        draw_dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        max_text_width = content_w - 20
        wrapped_lines = []
        for word in title.split():
            if not wrapped_lines:
                wrapped_lines.append(word)
                continue
            test = wrapped_lines[-1] + " " + word
            if draw_dummy.textlength(test, font=font) <= max_text_width:
                wrapped_lines[-1] = test
            else:
                wrapped_lines.append(word)

        # Calculate title area height
        bbox = draw_dummy.textbbox((0, 0), "Ag", font=font)
        line_h = bbox[3] - bbox[1]
        title_h = line_h * len(wrapped_lines) + 20
        if title_h % 2 != 0:
            title_h += 1

        # Render title banner
        title_img = Image.new("RGB", (content_w, title_h), "black")
        draw = ImageDraw.Draw(title_img)
        y = 10
        for line in wrapped_lines:
            text_w = draw.textlength(line, font=font)
            x = (content_w - text_w) // 2
            draw.text((x, y), line, font=font, fill="white")
            y += line_h

        title_array = np.asarray(title_img)

        # Process frames
        num_frames = min(len(out_frames), len(in_frames))
        log.info(f"Processing {num_frames} frames…")

        frames = []
        for i in range(num_frames):
            in_f = in_frames[i]
            out_f = out_frames[i]

            # Resize to common height if needed
            if in_f.shape[0] != content_h:
                in_f = np.array(Image.fromarray(in_f).resize((in_w, content_h)))
            if out_f.shape[0] != content_h:
                out_f = np.array(Image.fromarray(out_f).resize((out_w, content_h)))

            combined = np.concatenate([in_f, out_f], axis=1)
            full_f = np.concatenate([title_array, combined], axis=0)
            frames.append(full_f)

        video_np = np.stack(frames)
        video_t = torch.from_numpy(video_np).permute(3,0,1,2) / 255.0

        # Save video
        dst = os.path.splitext(output_video_path)[0] + "_concatenated.mp4"
        log.info(f"Saving to {dst}")
        save_image_or_video(video_t.float(), dst, fps=fps)
        log.success("Done.")
        return dst

    except Exception as e:
        log.error(f"Concatenation failed: {e}")
        return None

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

    parser.add_argument("--input_video_path", type=str, default="assets/video2world/input3.mp4", help="Path to the input video")
    parser.add_argument("--text2image_edit_strength", type=float, default=0.4, help="Strength of the text2image edit (0.0-1.0)")
    parser.add_argument("--video2world_edit_strength", type=float, default=0.8, help="Strength of the video2world edit (0.0-1.0)")
    return parser.parse_args()

def setup_video2world_pipeline(args: argparse.Namespace, text_encoder=None):
    log.info(f"Using model size: {args.model_size}")
    if hasattr(args, "natten") and args.natten:
        assert args.model_size in ["2B", "14B"]
        config = (
            PREDICT2_VIDEO2WORLD_WITH_NATTEN_PIPELINE_2B
            if args.model_size == "2B"
            else PREDICT2_VIDEO2WORLD_WITH_NATTEN_PIPELINE_14B
        )

        config.resolution = args.resolution

        if args.fps == 10:
            config.state_t = 16

        if args.resolution != "720":
            raise NotImplementedError("Cosmos-Predict2 + NATTEN only supports 720p inference at the moment.")

        if args.aspect_ratio != "16:9":
            raise NotImplementedError("Cosmos-Predict2 + NATTEN only supports 16:9 aspect ratio at the moment.")

        dit_path = (
            f"checkpoints/nvidia/Cosmos-Predict2-{args.model_size}-Video2World/model-720p-{args.fps}fps-natten.pt"
        )

    elif args.model_size == "2B":
        config = PREDICT2_VIDEO2WORLD_PIPELINE_2B

        config.resolution = args.resolution
        if args.fps == 10:  # default is 16 so no need to change config
            config.state_t = 16

        dit_path = f"checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-{args.resolution}p-{args.fps}fps.pt"
    elif args.model_size == "14B":
        config = PREDICT2_VIDEO2WORLD_PIPELINE_14B

        config.resolution = args.resolution
        if args.fps == 10:  # default is 16 so no need to change config
            config.state_t = 16

        dit_path = f"checkpoints/nvidia/Cosmos-Predict2-14B-Video2World/model-{args.resolution}p-{args.fps}fps.pt"
    else:
        raise ValueError("Invalid model size. Choose either '2B' or '14B'.")
    if hasattr(args, "dit_path") and args.dit_path:
        dit_path = args.dit_path

    log.info(f"Using dit_path: {dit_path}")

    # Only set up text encoder path if no encoder is provided
    text_encoder_path = None if text_encoder is not None else "checkpoints/google-t5/t5-11b"
    if text_encoder is not None:
        log.info("Using provided text encoder")
    else:
        log.info(f"Using text encoder from: {text_encoder_path}")

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment for multi-GPU inference
    if hasattr(args, "num_gpus") and args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")

        # Check if distributed environment is already initialized
        if not parallel_state.is_initialized():
            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
            log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")
        else:
            log.info("Distributed environment already initialized, skipping initialization")
            # Check if we need to reinitialize with different context parallel size
            current_cp_size = parallel_state.get_context_parallel_world_size()
            if current_cp_size != args.num_gpus:
                log.warning(f"Context parallel size mismatch: current={current_cp_size}, requested={args.num_gpus}")
                log.warning("Using existing context parallel configuration")
            else:
                log.info(f"Using existing context parallel group with {current_cp_size} GPUs")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False
    config.guardrail_config.offload_model_to_cpu = args.offload_guardrail

    # Disable prompt refiner if requested
    if args.disable_prompt_refiner:
        log.warning("Prompt refiner is disabled")
        config.prompt_refiner_config.enabled = False
    config.prompt_refiner_config.offload_model_to_cpu = args.offload_prompt_refiner

    # Load models
    log.info(f"Initializing Video2WorldSDEditPipeline with model size: {args.model_size}")
    pipe = Video2WorldSDEditPipeline.from_config(
        config=config,
        dit_path=dit_path,
        text_encoder_path=text_encoder_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=args.load_ema,
        load_prompt_refiner=True,
    )

    # Set the provided text encoder if one was passed
    if text_encoder is not None:
        pipe.text_encoder = text_encoder

    return pipe

def setup_text2image_pipeline(args: argparse.Namespace, text_encoder=None) -> Text2ImageSDEditPipeline:
    log.info(f"Using model size: {args.model_size}")
    if args.model_size == "2B":
        config = PREDICT2_TEXT2IMAGE_PIPELINE_2B
        dit_path = "checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt"
    elif args.model_size == "14B":
        config = PREDICT2_TEXT2IMAGE_PIPELINE_14B
        dit_path = "checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt"
    else:
        raise ValueError("Invalid model size. Choose either '2B' or '14B'.")
    if hasattr(args, "dit_path") and args.dit_path:
        dit_path = args.dit_path

    log.info(f"Using dit_path: {dit_path}")
    # Only set up text encoder path if no encoder is provided
    text_encoder_path = None if text_encoder is not None else "checkpoints/google-t5/t5-11b"
    if text_encoder is not None:
        log.info("Using provided text encoder")
    else:
        log.info(f"Using text encoder from: {text_encoder_path}")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False
    config.guardrail_config.offload_model_to_cpu = args.offload_guardrail

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Check if we're in a distributed environment (called from text2world)
    is_distributed = parallel_state.is_initialized() and torch.distributed.is_initialized()

    if is_distributed:
        # We're in a multi-GPU text2world context - only initialize on rank 0
        from imaginaire.utils.distributed import get_rank

        rank = get_rank()

        if rank == 0:
            log.info("Rank 0: Initializing Text2ImagePipeline for text2world")
            # Load models only on rank 0
            log.info(f"Initializing Text2ImagePipeline with model size: {args.model_size}")
            pipe = Text2ImageSDEditPipeline.from_config(
                config=config,
                dit_path=dit_path,
                device="cuda",
                torch_dtype=torch.bfloat16,
                load_ema_to_reg=args.load_ema,
            )
            return pipe
        else:
            log.info(f"Rank {rank}: Skipping Text2ImagePipeline initialization - will wait for rank 0")
            return None  # Return None for non-rank-0 processes
    else:
        # We're running as standalone text2image script
        # Only initialize distributed if num_gpus > 1 AND we're running standalone
        if hasattr(args, "num_gpus") and args.num_gpus > 1:
            log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")

            # Check if distributed environment is already initialized
            if not parallel_state.is_initialized():
                distributed.init()
                parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
                log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")
            else:
                log.info("Distributed environment already initialized, skipping initialization")
                # Check if we need to reinitialize with different context parallel size
                current_cp_size = parallel_state.get_context_parallel_world_size()
                if current_cp_size != args.num_gpus:
                    log.warning(f"Context parallel size mismatch: current={current_cp_size}, requested={args.num_gpus}")
                    log.warning("Using existing context parallel configuration")
                else:
                    log.info(f"Using existing context parallel group with {current_cp_size} GPUs")

        # Load models for standalone execution
        log.info(f"Initializing Text2ImagePipeline with model size: {args.model_size}")
        pipe = Text2ImageSDEditPipeline.from_config(
            config=config,
            dit_path=dit_path,
            text_encoder_path=text_encoder_path,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=args.load_ema,
        )

        # Set the provided text encoder if one was passed
        if text_encoder is not None:
            pipe.text_encoder = text_encoder

        return pipe


def process_single_image_generation(
    pipe: Text2ImageSDEditPipeline,
    prompt: str,
    output_path: str,
    negative_prompt: str,
    aspect_ratio: str,
    seed: int,
    use_cuda_graphs: bool,
    benchmark: bool,
    edit_strength: float,
    input_video_path: str,
) -> bool:
    log.info(f"Running Text2ImagePipeline\nprompt: {prompt}")

    # When benchmarking, run inference 4 times, exclude the 1st due to warmup and average time.
    num_repeats = 4 if benchmark else 1
    time_sum = 0
    for i in range(num_repeats):
        # Generate image
        if benchmark and i > 0:
            torch.cuda.synchronize()
            start_time = time.time()
        image = pipe(
            prompt=prompt,
            edit_strength=edit_strength,
            input_video_path=input_video_path,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
        )
        if benchmark and i > 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            time_sum += elapsed
            log.info(f"[iter {i} / {num_repeats - 1}] Generation time: {elapsed:.1f} seconds.")
    if benchmark:
        time_avg = time_sum / (num_repeats - 1)
        log.critical(f"The benchmarked generation time for Text2ImagePipeline is {time_avg:.1f} seconds.")

    if image is not None:
        # save the generated image
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving generated image to: {output_path}")
        save_image_or_video(image, output_path)
        log.success(f"Successfully saved image to: {output_path}")
        # save the prompts used to generate the video
        output_prompt_path = os.path.splitext(output_path)[0] + ".txt"
        prompts_to_save = {"prompt": prompt, "negative_prompt": negative_prompt}
        save_text_prompts(prompts_to_save, output_prompt_path)
        log.success(f"Successfully saved prompt file to: {output_prompt_path}")
        return True
    return False

def process_single_video_generation(
    pipe: Video2WorldSDEditPipeline,
    input_path: str,
    prompt: str,
    output_path: str,
    negative_prompt: str,
    aspect_ratio: str,
    num_conditional_frames: int,
    guidance: float,
    seed: int,
    benchmark: bool = False,
    use_cuda_graphs: bool = False,
    edit_strength: float= 0.8,
    image_edit_strength: float = 0.4,
    input_video_path: str = "",
) -> bool:
    # Validate input file
    if not validate_input_file(input_path, num_conditional_frames):
        log.warning(f"Input file validation failed: {input_path}")
        return False

    log.info(f"Running Video2WorldPipeline\ninput: {input_path}\nprompt: {prompt}")

    num_repeats = 4 if benchmark else 1
    time_sum = 0
    for i in range(num_repeats):
        if benchmark and i > 0:
            torch.cuda.synchronize()
            start_time = time.time()
        video, input_video, prompt_used = pipe(
            prompt=prompt,
            edit_strength=edit_strength,	
	        input_video_path=input_video_path,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            input_path=input_path,
            num_conditional_frames=num_conditional_frames,
            guidance=guidance,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
            return_prompt=True,
        )
        if benchmark and i > 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            time_sum += elapsed
            log.info(f"[iter {i} / {num_repeats - 1}] Generation time: {elapsed:.1f} seconds.")
    if benchmark:
        time_avg = time_sum / (num_repeats - 1)
        log.critical(f"Average generation time for Video2WorldPipeline is {time_avg:.1f} seconds.")

    if video is not None:
        # save the generated video
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving the generated video to: {output_path}")
        if pipe.config.state_t == 16:
            fps = 10
        else:
            fps = 16

        save_image_or_video(video, output_path, fps=fps)
        log.success(f"Successfully saved video to: {output_path}")

        # Save the input video
        input_video_save_path = output_path.replace(".mp4", "_input.mp4")
        save_image_or_video(input_video, input_video_save_path, fps=fps)
        log.success(f"Successfully saved input video to: {input_video_save_path}")

        # Create concatenated video with title
        concatenated_path = concatenate_videos_with_title(
            output_video_path=output_path,
            input_video_path=input_video_save_path,
            title=f'{prompt} (edit strength: img: {image_edit_strength}, vid: {edit_strength})',
            fps=fps
        )
        
        if concatenated_path:
            log.success(f"Successfully created concatenated video: {concatenated_path}")

        # save the prompts used to generate the video
        output_prompt_path = os.path.splitext(output_path)[0] + ".txt"
        prompts_to_save = {"prompt": prompt, "negative_prompt": negative_prompt}
        if (
            pipe.prompt_refiner is not None
            and getattr(pipe.config, "prompt_refiner_config", None) is not None
            and getattr(pipe.config.prompt_refiner_config, "enabled", False)
        ):
            prompts_to_save["refined_prompt"] = prompt_used
        save_text_prompts(prompts_to_save, output_prompt_path)
        log.success(f"Successfully saved prompt file to: {output_prompt_path}")

        return True
    return False


def generate_first_frames(text2image_pipe: Text2ImageSDEditPipeline, args: argparse.Namespace) -> list:
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
                    edit_strength=args.text2image_edit_strength,
	    input_video_path=args.input_video_path,
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
                edit_strength=args.text2image_edit_strength,
	            input_video_path=args.input_video_path,
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

def generate_videos(video2world_pipe: Video2WorldSDEditPipeline, batch_items: list, args: argparse.Namespace) -> None:
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
            edit_strength=args.video2world_edit_strength,
            image_edit_strength=args.text2image_edit_strength,
            input_video_path=args.input_video_path,
        )

        # # Clean up the temporary image file
        # if os.path.exists(temp_image_path):
        #     os.remove(temp_image_path)
        #     log.success(f"Cleaned up temporary image: {temp_image_path}")


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
