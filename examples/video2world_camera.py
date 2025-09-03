import argparse
import os

from cosmos_predict2.configs.camera_conditioned.config import PREDICT2_VIDEO2WORLD_PIPELINE_2B_CAMERA_CONDITIONED
from cosmos_predict2.data.datasets import AGIBotDataset, CameraTrajectoryDataset
from cosmos_predict2.pipelines.video2world_camera import Video2WorldCameraConditionedPipeline
import torch
import torch.distributed as dist
from megatron.core import parallel_state

from imaginaire.utils import distributed, log
from imaginaire.visualize.video import save_img_or_video


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the Video2World inference script."""
    parser = argparse.ArgumentParser(description="Camera-conditioned video generation with Cosmos Predict2")

    # Mode parameters
    parser.add_argument(
        "--mode",
        type=str,
        choices=["camera_trajectory", "agi_bot"],
        required=True,
        help="Type of dataset to use for camera-conditioned video generation. Options: 'camera_trajectory', 'agi_bot'"
    )

    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to directory containing model, tokenizer, and text encoder checkpoint files",
    )
    parser.add_argument(
        "--load_ema",
        action="store_true",
        help="Use EMA weights for generation.",
    )

    # Input parameters
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input image or video for conditioning (include file extension)"
    )
    parser.add_argument(
        "--camera_path",
        type=str,
        required=True,
        help="Path to directory containing camera trajectory files (e.g., pan_right.txt, arc_left.txt, etc.)"
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        nargs=2,
        help="Required for camera_trajectory mode. List of camera trajectories to use for camera conditioned video generation (e.g., 'pan_right' 'pan_left'). Should be present in the camera_path directory."
    )
    parser.add_argument(
        "--num_latent_conditional_frames",
        type=int,
        default=1,
        help="Number of latent conditional frames (1 or 2). For images, both values work by duplicating frames. For videos, uses the first N frames.",
    )
    parser.add_argument(
        "--focal",
        type=int,
        help="Focal length of the camera"
    )

    # Generation parameters
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to guide the video generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Custom negative prompt for classifier-free guidance. If not specified, uses default embeddings from S3.",
    )
    parser.add_argument(
        "--num_video_frames", 
        type=int, 
        default=93, 
        help="Number of video frames to generate"
    )
    parser.add_argument(
        "--guidance", 
        type=int, 
        default=7, 
        help="Guidance scale for classifier-free guidance"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for text encoder"
    )

    # Output parameters
    parser.add_argument(
        "--save_path", 
        type=str, 
        required=True,
        help="Path to save generated video"
    )

    # System parameters
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallelism. For example, set to 8 for 8 GPUs",
    )

    return parser.parse_args()

def setup_pipeline(args: argparse.Namespace):
    if args.num_gpus > 1:
        # Initialize distributed environment
        distributed.init()

        # Initialize model parallel states
        parallel_state.initialize_model_parallel(
            context_parallel_size=args.num_gpus,
        )

    config = PREDICT2_VIDEO2WORLD_PIPELINE_2B_CAMERA_CONDITIONED

    pipeline = Video2WorldCameraConditionedPipeline.from_config(
        config, 
        model_path=args.model_path,
        load_ema_to_reg=args.load_ema,
        torch_dtype=torch.bfloat16,
        num_gpus=args.num_gpus,
        cache_dir=args.cache_dir,
    )

    if args.num_gpus > 1:
        cp_group = parallel_state.get_context_parallel_group()
        pipeline.dit.enable_context_parallel(cp_group)

    return pipeline


def get_pipeline_input(
    pipeline: Video2WorldCameraConditionedPipeline,
    video: torch.Tensor,
    prompt: str,
    camera: torch.Tensor,
    negative_prompt: str = None,
    use_neg_prompt: bool = True,
    batch_size: int = 1,
):
    """
    Prepares the input data batch for the diffusion model.

    Constructs a dictionary containing the video tensor, text embeddings,
    and other necessary metadata required by the model's forward pass.
    Optionally includes negative text embeddings.

    Args:
        video (torch.Tensor): The input video tensor (B, C, T, H, W).
        prompt (str): The text prompt for conditioning.
        camera: (torch.Tensor) Target camera extrinsics and intrinsics for the K output videos.
        num_conditional_frames (int): Number of conditional frames to use.
        negative_prompt (str, optional): Custom negative prompt. If None, uses default S3 embeddings.
        use_neg_prompt (bool, optional): Whether to include negative prompt embeddings. Defaults to True.

    Returns:
        dict: A dictionary containing the prepared data batch, moved to the correct device and dtype.
    """
    _, _, _, H, W = video.shape

    data_batch = {
        "dataset_name": "video_data",
        "video": video,
        "camera": camera,
        "fps": torch.full((batch_size,), 15.0),  # FPS value (might be used by model)
        "padding_mask": torch.zeros(batch_size, 1, H, W),  # Padding mask (assumed no padding here)
    }

    # Move tensors to GPU and convert to bfloat16 if they are floating point
    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
            data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

    # Handle negative prompts for classifier-free guidance
    if use_neg_prompt:
        assert negative_prompt is not None, "Negative prompt is required when use_neg_prompt is True"

    # Compute text embeddings
    data_batch["ai_caption"] = [prompt]
    data_batch["t5_text_embeddings"] = pipeline.text_encoder.compute_text_embeddings_online(
        data_batch={"ai_caption": [prompt], "images": None},
        input_caption_key="ai_caption",
    )
    if use_neg_prompt:
        data_batch["neg_t5_text_embeddings"] = pipeline.text_encoder.compute_text_embeddings_online(
            data_batch={"ai_caption": [negative_prompt], "images": None},
            input_caption_key="ai_caption",
        )

    # Move tensors to GPU and convert to bfloat16 if they are floating point
    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(data_batch[k]):
            data_batch[k] = v.cuda().to(dtype=torch.bfloat16)

    return data_batch

def cleanup(num_gpus):
    """Clean up distributed resources."""
    if num_gpus > 1:
        torch.distributed.barrier()
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def save_video(video, save_path):
    if distributed.get_rank() == 0:
        save_root = "/".join(save_path.split("/")[:-1])
        os.makedirs(save_root, exist_ok=True)
        save_path = save_path.replace(".mp4", "")
        save_img_or_video((1.0 + video[0]) / 2, save_path, fps=30)
        log.info(f"Saved video to {save_path}.mp4")

def main():
    args = parse_arguments()

    height, width = (704, 1280)

    log.info(f"Validating arguments...")

     # Validate mode and prepare test data (source video, target camera, target trajectory)
    if args.mode == "camera_trajectory":
        assert args.trajectories is not None, "trajectories is required for camera_trajectory mode"
        assert args.focal is not None, "focal is required for camera_trajectory mode"
        dataset = CameraTrajectoryDataset(
            trajectories=args.trajectories,
            input_path=args.input_path,
            camera_path=args.camera_path,
            prompt=args.prompt,
            focal=args.focal,
            height=height,
            width=width,
        )
    elif args.mode == "agi_bot":
        video_prefix = args.input_path.split("/")[-1].split(".")[0]
        dataset = AGIBotDataset(
            video_prefix=video_prefix,
            input_path=args.input_path,
            camera_path=args.camera_path,
            prompt=args.prompt,
            height=height,
            width=width,
        )

    log.info(f"Setting up pipeline...")
    
    # Set up pipeline
    pipeline = setup_pipeline(args)

    log.info(f"Loading dataset from {args.input_path}...")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_gpus,
    )

    use_neg_prompt = args.negative_prompt is not None
    
    log.info(f"Generating pipeline inputs...")

    # Generate pipeline inputs
    for batch in dataloader:
        data_batch = get_pipeline_input(
            pipeline=pipeline,
            video=batch[0]["video"],
            prompt=batch[0]["text"],
            camera=batch[0]["camera"],
            negative_prompt=args.negative_prompt if use_neg_prompt else None,
            use_neg_prompt=use_neg_prompt,
        )

    log.info(f"Generating video...")

    # Pass pipeline inputs to the pipeline to generate video
    video = pipeline(
        data_batch,
        n_sample=1,
        guidance=args.guidance,
        seed=args.seed,
        use_negative_prompt=use_neg_prompt,
        num_conditional_frames=args.num_latent_conditional_frames,
    )

    # Save video to given path
    save_video(video, args.save_path)

    log.info(f"Cleaning up distributed resources")
    # Clean up distributed resources
    cleanup(args.num_gpus)

    print(f"Done!")


if __name__ == "__main__":
    main()