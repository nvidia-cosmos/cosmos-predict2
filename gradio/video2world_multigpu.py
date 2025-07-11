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
import os
import subprocess
import typing

from loguru import logger
from datetime import datetime

# Ensure h11 is uninstalled for Gradio compatibility, because httpx conflicts with Gradio's requirements
try:
    import gradio as gr
except ImportError:
    subprocess.check_call(["pip", "uninstall", "h11", "-y"])
    subprocess.check_call(["pip", "install", "gradio", "--use-deprecated=legacy-resolver"])
    import gradio as gr

VIDEO_EXTENSION = typing.Literal[".mp4"]
IMAGE_EXTENSION = typing.Literal[".jpg", ".jpeg"]
IMAGE_VIDEO_EXTENSION = typing.Literal[VIDEO_EXTENSION, IMAGE_EXTENSION]
FILE_TYPE = typing.Literal["video", "image", "other"]
GRID_COLUMNS = 4
MAX_VIDEOS = 16  # Limit to avoid overflowing UI


class CosmosPredict2GradioApp:
    def __init__(self, checkpoint_dir="checkpoints", output_dir="/mnt/pvc/gradio/cosmos-predict2/output"):
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    def create_args_namespace(
        self,
        input_path,
        prompt,
        negative_prompt,
        model_size,
        resolution,
        fps,
        guidance,
        seed,
        num_conditional_frames,
        num_gpus=8,
        disable_guardrail=False,
        offload_guardrail=False,
        disable_prompt_refiner=False,
        offload_prompt_refiner=False,
        batch_input_json=None,
        dit_path=None
    ):
        """Create an argparse.Namespace object that mimics command line arguments in transfer.py"""
        args = argparse.Namespace()

        # Required arguments
        args.input_path = input_path
        args.model_size = model_size
        args.resolution = resolution
        args.fps = fps
        args.dit_path = dit_path

        # Video and prompt settings
        args.prompt = prompt
        args.negative_prompt = negative_prompt
        args.num_conditional_frames = num_conditional_frames
        args.batch_input_json = batch_input_json

        # Create unique output folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_dir, exist_ok=True)
        args.save_path = os.path.join(self.output_dir, f"generated_video_{timestamp}.mp4")

        # Generation parameters
        args.guidance = guidance
        args.seed = seed
        args.num_gpus = num_gpus

        # Memory optimization
        args.disable_guardrail = disable_guardrail
        args.offload_guardrail = offload_guardrail
        args.disable_prompt_refiner = disable_prompt_refiner
        args.offload_prompt_refiner = offload_prompt_refiner
        args.benchmark = False  # Disable benchmarking by default

        return args

    def namespace_to_cli_args(self, args: argparse.Namespace) -> list[str]:
        """Convert argparse.Namespace to list of CLI args like --key=value (no quotes)."""
        cli_args = []
        for k, v in vars(args).items():
            if isinstance(v, bool):
                if v:
                    cli_args.append(f"--{k}")
            elif isinstance(v, (list, tuple)):
                for item in v:
                    item_str = f'{item}' if isinstance(item, str) else str(item)
                    cli_args.extend([f"--{k}", item_str])
            elif v is not None:
                value_str = f'{v}' if isinstance(v, str) else str(v)
                cli_args.extend([f"--{k}", value_str])
        return cli_args

    def run_inference(self, args: argparse.Namespace):
        cli_args = self.namespace_to_cli_args(args)
        logger.info(f"Running inference with args: {cli_args}")
        cmd = [
            "torchrun",
            f"--nproc-per-node={args.num_gpus}",
            "./examples/video2world.py",
            *cli_args
        ]
        logger.info(f"Running inference with cmd: {cmd}")
        subprocess.run(cmd, check=True)

    def infer(
        self,
        input_path,
        prompt,
        negative_prompt,
        model_size,
        resolution,
        fps,
        guidance,
        seed,
        num_conditional_frames,
        num_gpus=1,
        disable_guardrail=False,
        offload_guardrail=False,
        disable_prompt_refiner=False,
        offload_prompt_refiner=False,
        batch_input_json=None,
    ):
        """Run inference with the provided parameters"""

        # Create args namespace
        args = self.create_args_namespace(
            input_path=input_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_size=model_size,
            resolution=resolution,
            fps=fps,
            guidance=guidance,
            seed=seed,
            num_conditional_frames=num_conditional_frames,
            num_gpus=num_gpus,
            disable_guardrail=disable_guardrail,
            offload_guardrail=offload_guardrail,
            disable_prompt_refiner=disable_prompt_refiner,
            offload_prompt_refiner=offload_prompt_refiner,
            batch_input_json=batch_input_json,
        )
        self.run_inference(args)

        # Return the generated video path and status message
        return args.save_path, f"Video generated successfully at {args.save_path}"


def setup_gradio_interface():
    _output_dir = "/mnt/pvc/gradio/cosmos-predict2/output"
    # Ensure output directories exist\
    os.makedirs(_output_dir, exist_ok=True)
    app = CosmosPredict2GradioApp(checkpoint_dir="checkpoints", output_dir=_output_dir)

    # Recursively collect video paths
    def _get_all_video_paths():
        video_paths = []
        for root, _, files in os.walk(app.output_dir):
            for file in files:
                if file.lower().endswith(VIDEO_EXTENSION.__args__):
                    full_path = os.path.join(root, file)
                    video_paths.append(full_path)

        # Sort by creation time (newest first)
        video_paths.sort(key=lambda x: os.path.getctime(x), reverse=True)
        return video_paths[:MAX_VIDEOS]  # Limit total for layout

    # Function to return list of file paths to update video components
    def _load_video_values():
        paths = _get_all_video_paths()
        values = [gr.update(value=path, label=os.path.basename(path), visible=True) for path in paths]
        # Fill remaining placeholders with None
        while len(values) < GRID_COLUMNS * (MAX_VIDEOS // GRID_COLUMNS):
            values.append(gr.update(value=None, label=""))
        return values

    def _clear_video_values():
        paths = _get_all_video_paths()
        for path in paths:
            os.remove(path)
        return [gr.update(value=None, label="", visible=False) for _ in range(GRID_COLUMNS * (MAX_VIDEOS // GRID_COLUMNS))]

    with gr.Blocks(title="Cosmos-Predict2 Video Generation", theme=gr.themes.Soft()) as interface:
        with gr.Tab("Main"):
            gr.Markdown("# Cosmos-Predict2: World Generation with Adaptive Multimodal Control")
            gr.Markdown("Upload a image or video and configure controls to generate a new video with the Cosmos-Predict2 model.")
            gr.Markdown(f"**Output Directory**: {app.output_dir}, Please look into Gallery tab to see the previous results.")

            with gr.Row():
                with gr.Column(scale=1):
                    # Input controls
                    input_file = gr.File(label="Upload Image or Video File", file_types=list(IMAGE_VIDEO_EXTENSION.__args__), height=300, visible=True, value=None)
                    input_preview1 = gr.Image(label="Input Preview", height=300, visible=False)
                    input_preview2 = gr.Video(label="Input Preview", height=300, visible=False)

                    def _update_preview(file):
                        if file:
                            ext = os.path.splitext(file.name)[1]
                            if ext in [".jpg", ".jpeg"]:
                                return gr.update(visible=True, value=file), gr.update(visible=False, value=None)
                            else:
                                return gr.update(visible=False, value=None), gr.update(visible=True, value=file)

                    input_file.upload(
                        fn=_update_preview,
                        inputs=[input_file],
                        outputs=[input_preview1, input_preview2],
                    )

                    logger.info(f"Input file controls initialized with input file {input_file}")

                    logger.info("Input file controls initialized.")
                    prompt = gr.Textbox(
                        label="Prompt",
                        value="A point-of-view video shot from inside a vehicle, capturing a quiet suburban street bathed in bright sunlight. The road is lined with parked cars on both sides, and buildings, likely residential or small businesses, are visible across the street. A STOP sign is prominently displayed near the center of the intersection. The sky is clear and blue, with the sun shining brightly overhead, casting long shadows on the pavement. On the left side of the street, several vehicles are parked, including a van with some text on its side. Across the street, a white van is parked near two trash bins, and a red SUV is parked further down. The buildings on either side have a mix of architectural styles, with some featuring flat roofs and others with sloped roofs. Overhead, numerous power lines stretch across the street, and a few trees are visible in the background, partially obscuring the view of the buildings. As the video progresses, a white car truck makes a right turn into the adjacent opposite lane. The ego vehicle slows down and comes to a stop, waiting until the car fully enters the opposite lane before proceeding. The pedestrian keeps walking on the street. The other vehicles remain stationary, parked along the curb. The scene remains static otherwise, with no significant changes in the environment or additional objects entering the frame. By the end of the video, the white car truck has moved out of the camera view, the rest of the scene remains largely unchanged, maintaining the same composition and lighting conditions as the beginning.",
                        lines=2,
                    )

                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.",
                        lines=1,
                    )

                    # Model and video settings
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            model_size = gr.Dropdown(choices=["2B", "14B"], label="Model Size", interactive=True, value="2B")
                        with gr.Column(scale=3):
                            resolution = gr.Dropdown(choices=["480", "720"], label="Resolution", interactive=True, value="720")
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            fps = gr.Dropdown(choices=[10, 16], label="FPS", interactive=True, value=16)
                        with gr.Column(scale=3):
                            num_conditional_frames = gr.Dropdown(choices=[1, 5], label="Conditional Frames", interactive=True, value=1)

                    # Advanced settings
                    with gr.Accordion("Advanced Settings", open=False):
                        guidance = gr.Slider(1, 15, value=7.0, step=0.5, label="Guidance")
                        seed = gr.Number(value=1, label="Seed", precision=0, interactive=True)
                        num_gpus = gr.Slider(1, 8, value=8, step=1, label="Number of GPUs")
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                disable_guardrail = gr.Checkbox(label="Disable Guardrail", value=True)
                            with gr.Column(scale=3):
                                offload_guardrail = gr.Checkbox(label="Offload Guardrail", value=True)
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1):
                                disable_prompt_refiner = gr.Checkbox(label="Disable Prompt Refiner", value=True)
                            with gr.Column(scale=3):
                                offload_prompt_refiner = gr.Checkbox(label="Offload Prompt Refiner", value=True)

                    generate_btn = gr.Button("Generate Video", variant="primary", size="lg")

                with gr.Column(scale=1):
                    # Output
                    output_video = gr.Video(label="Generated Video", height=400)
                    status_text = gr.Textbox(label="Status", lines=5, interactive=False)

            # Event handler
            def gradio_infer(
                input_file,
                prompt,
                negative_prompt,
                model_size,
                resolution,
                fps,
                guidance,
                seed,
                num_conditional_frames,
                num_gpus,
                disable_guardrail,
                offload_guardrail,
                disable_prompt_refiner,
                offload_prompt_refiner,
                batch_input_json=None,
            ):
                logger.info(f"Starting inference with input file: {input_file}, prompt: {prompt}, model size: {model_size}, resolution: {resolution}, fps: {fps}, guidance: {guidance}, seed: {seed}, num_conditional_frames: {num_conditional_frames}, num_gpus: {num_gpus}")
                return app.infer(
                    input_path=input_file,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    model_size=model_size,
                    resolution=resolution,
                    fps=fps,
                    guidance=guidance,
                    seed=seed,
                    num_conditional_frames=num_conditional_frames,
                    num_gpus=num_gpus,
                    disable_guardrail=disable_guardrail,
                    offload_guardrail=offload_guardrail,
                    disable_prompt_refiner=disable_prompt_refiner,
                    offload_prompt_refiner=offload_prompt_refiner,   
                    batch_input_json=batch_input_json,             
                )

            generate_btn.click(
                fn=gradio_infer,
                inputs=[
                    input_file,
                    prompt,
                    negative_prompt,
                    model_size,
                    resolution,
                    fps,
                    guidance,
                    seed,
                    num_conditional_frames,
                    num_gpus,
                    disable_guardrail,
                    offload_guardrail,
                    disable_prompt_refiner,
                    offload_prompt_refiner,
                    # batch_input_json,
                ],
                outputs=[output_video, status_text],
            )

            # Examples section
            gr.Markdown("## Tips for better results:")
            gr.Markdown(
                """
            - **Describe a single, captivating scene**: Focus on one scene to prevent unnecessary shot changes
            - **Use detailed prompts**: Rich descriptions lead to better quality outputs  
            - **Experiment with control weights**: Different combinations can yield different artistic effects
            - **Adjust sigma_max**: Lower values preserve more of the input video structure
            """
            )

            gr.Markdown("## File Storage:")
            gr.Markdown(
                f"""
            - **Input videos**: Temporarily stored in Gradio's cache, then copied to output folder
            - **Generated videos**: Saved to `{app.output_dir}/generation_YYYYMMDD_HHMMSS.mp4`
            - **Output structure**: Each generation gets its own timestamped folder with input copy, output video, and prompt
            """
            )
        with gr.Tab("Result Gallery"):
            gr.Markdown(f"# Video Results from {app.output_dir}")
            # Create placeholders
            video_outputs = []
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    load_btn = gr.Button("Refresh Results", variant="primary", size="lg")
                with gr.Column(scale=1):
                    clear_btn = gr.Button("Clear Results", variant="primary", size="lg")

            clear_btn.click(fn=_clear_video_values, outputs=video_outputs)
            load_btn.click(fn=_load_video_values, outputs=video_outputs)

            for _ in range(MAX_VIDEOS // GRID_COLUMNS):
                with gr.Row():
                    for _ in range(GRID_COLUMNS):
                        vid = gr.Video(interactive=False, visible=False)
                        video_outputs.append(vid)

    return interface


if __name__ == "__main__":
    # Check if checkpoints exist
    if not os.path.exists("checkpoints"):
        print("Error: checkpoints directory not found. Please download the model checkpoints first.")
        print("Run: python scripts/download_checkpoints.py --output_dir checkpoints/")
        exit(1)

    interface = setup_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        # Configure file upload limits
        # max_file_size="500MB",  # Adjust as needed
        allowed_paths=["/mnt/pvc/gradio"],  # Allow access to output directory
    )