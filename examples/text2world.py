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

import torch
from tqdm import tqdm

from cosmos_predict2.pipelines.text2image import Text2ImagePipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

# Import functionality from other example scripts
from examples.text2image import process_single_generation as process_single_image_generation
from examples.text2image import setup_pipeline as setup_text2image_pipeline
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT, cleanup_distributed
from examples.video2world import process_single_generation as process_single_video_generation
from examples.video2world import setup_pipeline as setup_video2world_pipeline
from imaginaire.utils import log

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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/generated_video.mp4",
        help="Path to save the generated video (include file extension)",
    )
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
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
        "--dit_path",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained models.",
    )
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value for video generation")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference in the video2world part",
    )
    parser.add_argument("--offload_guardrail", action="store_true", help="Offload guardrail to CPU to save GPU memory")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )
    parser.add_argument(
        "--offload_prompt_refiner", action="store_true", help="Offload prompt refiner to CPU to save GPU memory"
    )
    return parser.parse_args()


def generate_first_frames(text2image_pipe: Text2ImagePipeline, args: argparse.Namespace) -> list:
    """
    Generate first frames using the text2image pipeline.
    Returns a list of batch items containing prompt, output video path, and temp image path.
    """
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
            seed=args.seed,
            use_cuda_graphs=args.use_cuda_graphs,
            benchmark=args.benchmark,
        ):
            # Add single item to batch_items for consistent processing
            batch_items.append(
                {"prompt": args.prompt, "output_video": args.save_path, "temp_image_path": temp_image_path}
            )

    return batch_items


def generate_videos(video2world_pipe: Video2WorldPipeline, batch_items: list, args: argparse.Namespace) -> None:
    """
    Generate videos from first frames using the video2world pipeline.
    """
    # Process all items for video generation
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


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.benchmark:
            log.warning(
                "Running in benchmark mode. Each generation will be rerun a couple of times and the average generation time will be shown."
            )

        # Step 1: Initialize text2image pipeline and generate all first frames
        log.info("Step 1: Initializing text2image pipeline...")
        text2image_pipe = setup_text2image_pipeline(args)

        # Store text encoder for later use
        text_encoder = text2image_pipe.text_encoder

        # Generate first frames
        log.info("Step 1: Generating first frames...")
        batch_items = generate_first_frames(text2image_pipe, args)

        # Clean up text2image pipeline
        log.info("Step 1 complete. Cleaning up text2image pipeline to free memory...")
        del text2image_pipe
        torch.cuda.empty_cache()

        # Step 2: Initialize video2world pipeline and generate videos
        log.info("Step 2: Initializing video2world pipeline...")
        # Pass all video2world relevant arguments and the text encoder
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
