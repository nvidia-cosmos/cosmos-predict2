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
from typing import Tuple

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from megatron.core import parallel_state
from tqdm import tqdm

from cosmos_predict2.configs.base.config_text2image import (
    PREDICT2_TEXT2IMAGE_PIPELINE_2B,
    PREDICT2_TEXT2IMAGE_PIPELINE_14B,
)
from cosmos_predict2.configs.base.config_video2world import (
    PREDICT2_VIDEO2WORLD_PIPELINE_2B,
    PREDICT2_VIDEO2WORLD_PIPELINE_14B,
)
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

# Import functionality from other example scripts
from examples.text2image import process_single_generation as process_single_image_generation
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT
from examples.video2world import process_single_generation as process_single_video_generation
from imaginaire.utils import distributed, log, misc

_DEFAULT_POSITIVE_PROMPT = "An autonomous welding robot arm operating inside a modern automotive factory, sparks flying as it welds a car frame with precision under bright overhead lights."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text to World Generation with Cosmos Predict2")
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/generated_video.mp4",
        help="Path to save the generated video (include file extension)",
    )
    parser.add_argument("--use_cuda_graphs", action="store_true", help="Use CUDA Graphs for the text2image inference.")
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value for video generation")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference in the video2world part",
    )
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the generation in benchmark mode. It means that generation will be rerun a few times and the average generation time will be shown.",
    )
    return parser.parse_args()


def setup_pipeline(args: argparse.Namespace) -> Tuple[Text2ImagePipeline, Video2WorldPipeline]:
    log.info(f"Using model size: {args.model_size}")
    if args.model_size == "2B":
        # Config for image model
        config_text2image = PREDICT2_TEXT2IMAGE_PIPELINE_2B
        dit_path_text2image = "checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt"

        # Config for world model
        config_video2world = PREDICT2_VIDEO2WORLD_PIPELINE_2B
        dit_path_video2world = "checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt"

    elif args.model_size == "14B":
        # Config for image model
        config_text2image = PREDICT2_TEXT2IMAGE_PIPELINE_14B
        dit_path_text2image = "checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt"

        # Config for world model
        config_video2world = PREDICT2_VIDEO2WORLD_PIPELINE_14B
        dit_path_video2world = "checkpoints/nvidia/Cosmos-Predict2-14B-Video2World/model-720p-16fps.pt"

    else:
        raise ValueError("Invalid model size. Choose either '2B' or '14B'.")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config_text2image.guardrail_config.enabled = False
        config_video2world.guardrail_config.enabled = False

    # Disable prompt refiner if requested
    if args.disable_prompt_refiner:
        log.warning("Prompt refiner is disabled")
        config_video2world.prompt_refiner_config.enabled = False

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # First generate images from text
    log.info(f"Initializing Text2ImagePipeline with model size: {args.model_size}")
    text2image_pipe = Text2ImagePipeline.from_config(
        config=config_text2image,
        dit_path=dit_path_text2image,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    # Initialize distributed environment for multi-GPU inference (for video2world part)
    if hasattr(args, "num_gpus") and args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")

    # We'll initialize the video2world pipeline with the text_encoder from text2image
    text_encoder = text2image_pipe.text_encoder

    # Now, initialize video2world pipeline
    log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
    video2world_pipe = Video2WorldPipeline.from_config(
        config=config_video2world,
        dit_path=dit_path_video2world,
        text_encoder_path=None,  # We'll use the existing text_encoder
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_prompt_refiner=True,
    )
    # This will ensure that text encoder is not re-initialized
    video2world_pipe.text_encoder = text_encoder

    return text2image_pipe, video2world_pipe


def generate_video(args: argparse.Namespace, pipelines: Tuple[Text2ImagePipeline, Video2WorldPipeline]) -> None:
    if args.benchmark:
        log.warning(
            "Running in benchmark mode. Each generation will be rerun a couple of times and the average generation time will be shown."
        )
    text2image_pipe, video2world_pipe = pipelines

    # Get the base path for temporary image (without file extension)
    if args.batch_input_json is None:
        # In single mode, derive temp image path from save_path
        temp_image_path = os.path.splitext(args.save_path)[0] + "_temp.jpg"
    else:
        # For batch mode, we'll generate temp paths per item
        temp_image_path = None

    # Text-to-image
    batch_items = []

    if args.batch_input_json is not None:
        # Process batch inputs from JSON file
        log.info(f"Loading batch inputs from JSON file: {args.batch_input_json}")
        with open(args.batch_input_json, "r") as f:
            batch_inputs = json.load(f)

        # Generate all the first frames first
        for idx, item in enumerate(tqdm(batch_inputs, desc="Generating first frames")):
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
                seed=args.seed,
                use_cuda_graphs=args.use_cuda_graphs,
                benchmark=args.benchmark,
            ):
                # Save the item for the second stage
                batch_items.append({"prompt": prompt, "output_video": output_video, "temp_image_path": temp_image_name})
    else:
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
            seed=args.seed,
            use_cuda_graphs=args.use_cuda_graphs,
            benchmark=args.benchmark,
        ):
            # Add single item to batch_items for consistent processing
            batch_items.append(
                {"prompt": args.prompt, "output_video": args.save_path, "temp_image_path": temp_image_path}
            )

    # Process all items (batch or single) in a consistent way
    for item in tqdm(batch_items, desc="Generating videos from first frames"):
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
            num_conditional_frames=1,  # Always use 1 frame for text2world
            guidance=args.guidance,
            seed=args.seed,
            benchmark=args.benchmark,
        )

        # Clean up the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            log.success(f"Cleaned up temporary image: {temp_image_path}")

    return


def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    try:
        pipelines = setup_pipeline(args)
        generate_video(args, pipelines)
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()
