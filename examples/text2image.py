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
from tqdm import tqdm

from cosmos_predict2.configs.base.config_text2image import (
    PREDICT2_TEXT2IMAGE_PIPELINE_2B,
    PREDICT2_TEXT2IMAGE_PIPELINE_14B,
)
from cosmos_predict2.pipelines.text2image import Text2ImagePipeline
from imaginaire.utils import log, misc
from imaginaire.utils.io import save_image_or_video

_DEFAULT_POSITIVE_PROMPT = "A well-worn broom sweeps across a dusty wooden floor, its bristles gathering crumbs and flecks of debris in swift, rhythmic strokes. Dust motes dance in the sunbeams filtering through the window, glowing momentarily before settling. The quiet swish of straw brushing wood is interrupted only by the occasional creak of old floorboards. With each pass, the floor grows cleaner, restoring a sense of quiet order to the humble room."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text to Image Generation with Cosmos Predict2")
    parser.add_argument(
        "--model_size",
        choices=["2B", "14B"],
        default="2B",
        help="Size of the model to use for text-to-image generation",
    )
    parser.add_argument("--prompt", type=str, default=_DEFAULT_POSITIVE_PROMPT, help="Text prompt for image generation")
    parser.add_argument(
        "--batch_input_json",
        type=str,
        default=None,
        help="Path to JSON file containing batch inputs. Each entry should have 'prompt' and 'output_image' fields.",
    )
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative text prompt for image generation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/generated_image.jpg",
        help="Path to save the generated image (include file extension)",
    )
    parser.add_argument("--use_cuda_graphs", action="store_true", help="Use CUDA Graphs for the inference.")
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the generation in benchmark mode. It means that generation will be rerun a few times and the average generation time will be shown."
    )
    return parser.parse_args()


def setup_pipeline(args: argparse.Namespace) -> Text2ImagePipeline:
    log.info(f"Using model size: {args.model_size}")
    if args.model_size == "2B":
        config = PREDICT2_TEXT2IMAGE_PIPELINE_2B
        dit_path = "checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/model.pt"
    elif args.model_size == "14B":
        config = PREDICT2_TEXT2IMAGE_PIPELINE_14B
        dit_path = "checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/model.pt"
    else:
        raise ValueError("Invalid model size. Choose either '2B' or '14B'.")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load models
    log.info(f"Initializing Text2ImagePipeline with model size: {args.model_size}")
    pipe = Text2ImagePipeline.from_config(
        config=config,
        dit_path=dit_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )

    return pipe


def process_single_generation(pipe, prompt, output_path, negative_prompt, seed, use_cuda_graphs, benchmark):
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
            negative_prompt=negative_prompt,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
        )
        if benchmark and i > 0:
            torch.cuda.synchronize()
            time_sum += time.time() - start_time
    if benchmark:
        log.critical(f"The benchmarked generation time for Text2ImagePipeline is {time_sum / 3} seconds.")

    if image is not None:
        # save the generated image
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving generated image to: {output_path}")
        save_image_or_video(image, output_path)
        log.success(f"Successfully saved image to: {output_path}")
        return True
    return False


def generate_image(args: argparse.Namespace, pipe: Text2ImagePipeline) -> None:
    # Text-to-image
    if args.batch_input_json is not None:
        # Process batch inputs from JSON file
        log.info(f"Loading batch inputs from JSON file: {args.batch_input_json}")
        with open(args.batch_input_json, "r") as f:
            batch_inputs = json.load(f)

        for idx, item in enumerate(tqdm(batch_inputs)):
            prompt = item.get("prompt", "")
            output_image = item.get("output_image", f"output_{idx}.jpg")

            if not prompt:
                log.warning(f"Skipping item {idx}: Missing prompt")
                continue

            process_single_generation(
                pipe=pipe,
                prompt=prompt,
                output_path=output_image,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                use_cuda_graphs=args.use_cuda_graphs,
                benchmark=args.benchmark,
            )
    else:
        if args.use_cuda_graphs:
            log.warning(
                "Using CUDA Graphs for a single inference call may not be beneficial because of overhead of Graphs creation."
            )
        process_single_generation(
            pipe=pipe,
            prompt=args.prompt,
            output_path=args.save_path,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            use_cuda_graphs=args.use_cuda_graphs,
            benchmark=args.benchmark,
        )

    return


if __name__ == "__main__":
    args = parse_args()
    pipe = setup_pipeline(args)
    generate_image(args, pipe)
