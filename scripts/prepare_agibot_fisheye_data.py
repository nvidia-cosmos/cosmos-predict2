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

"""
Script to download, process, and split AgiBotWorld-Alpha dataset videos.
Dataset: https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha

Features:
1. Download specific tasks from the dataset
2. Extract tar files
3. Clean observations to keep only specified camera videos
4. Split videos into fixed-size windows

Usage examples:
    # Login to HuggingFace command line
    huggingface-cli login

    # Download, extract, and clean (default behavior)
    python prepare_agibot_fisheye_data.py

    # Only clean existing data
    python prepare_agibot_fisheye_data.py --delete-only

    # Only split videos into (5-second)windows
    python prepare_agibot_fisheye_data.py --split-only

"""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a subset of AgiBotWorld dataset")
    parser.add_argument(
        "--task_ids",
        nargs="+",
        type=int,
        default=[327],
        help="List of task IDs to download from AgiBotWorld",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/agibot",
        help="Directory to save the downloaded data (default: datasets/agibot)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="agibot-world/AgiBotWorld-Alpha",
        help="Hugging Face dataset repository ID",
    )
    parser.add_argument(
        "--keep_tar",
        action="store_true",
        help="Keep tar files after extraction (default: remove them)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="head_center_fisheye_color",
        help="Camera video to keep in observations (default: head_center_fisheye_color)",
    )
    parser.add_argument(
        "--delete-only",
        action="store_true",
        help="Only delete unwanted files, skip downloading (assumes data already exists)",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Only split videos into windows, skip downloading and cleaning",
    )
    parser.add_argument(
        "--val_episode_ids",
        type=str,
        nargs="+",
        default=[],
        help="List of episode IDs to use for validation (only used if --split-only is true)",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=5.0,
        help="Window size in seconds for video splitting (default: 5.0)",
    )
    parser.add_argument(
        "--min-last-window",
        type=float,
        default=7.5,
        help="Minimum size for last window; if remainder is larger, split into two windows (default: 7.5)",
    )
    return parser.parse_args()


def get_allow_patterns(task_ids) -> list:
    """Generate allow patterns for specific task IDs."""
    patterns = []

    # Add patterns for observations
    for task_id in task_ids:
        patterns.append(f"observations/{task_id}/*.tar")

    # Add patterns for task_info files
    for task_id in task_ids:
        patterns.append(f"task_info/task_{task_id}.json")

    return patterns


def extract_tar_files(data_dir, remove_tar=True) -> None:
    """Extract all tar files in the observations directory and optionally remove them."""
    observations_dir = Path(data_dir) / "observations"

    if not observations_dir.exists():
        print(f"Observations directory not found: {observations_dir}")
        return

    # Find all tar files recursively
    tar_files = list(observations_dir.rglob("*.tar"))
    tar_files.extend(list(observations_dir.rglob("*.tar.gz")))
    tar_files.extend(list(observations_dir.rglob("*.tgz")))

    if not tar_files:
        print("No tar files found in observations directory")
        return

    print(f"Found {len(tar_files)} tar file(s) to extract")

    for tar_path in tar_files:
        print(f"Extracting: {tar_path}")
        try:
            # Extract to the same directory as the tar file
            extract_dir = tar_path.parent

            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=extract_dir)

            print(f"  ✓ Extracted to: {extract_dir}")

            # Remove tar file if requested
            if remove_tar:
                tar_path.unlink()
                print(f"  ✓ Removed tar file: {tar_path}")

        except Exception as e:
            print(f"  ✗ Error processing {tar_path}: {e}")


def get_video_duration(video_path) -> float | None:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting duration for {video_path}: {e}")
        return None


def get_video_info(video_path) -> tuple[float, float] | tuple[None, None]:
    """Get video duration and frame rate using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)

        # Get duration
        duration = float(info["format"]["duration"])

        # Get frame rate (r_frame_rate is in the format "num/den")
        fps_str = info["streams"][0]["r_frame_rate"]
        fps_num, fps_den = map(int, fps_str.split("/"))
        fps = fps_num / fps_den

        return duration, fps
    except (subprocess.CalledProcessError, ValueError, KeyError) as e:
        print(f"Error getting video info for {video_path}: {e}")
        return None, None


def check_ffmpeg() -> bool:
    """Check if ffmpeg and ffprobe are available."""
    for cmd in ["ffmpeg", "ffprobe"]:
        try:
            subprocess.run([cmd, "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Error: {cmd} not found. Please install ffmpeg.")
            return False
    return True


def split_video_into_windows(video_path, output_dir, task_id, episode_id, window_size=5.0, min_last_window=7.5) -> list:
    """Split a video into fixed-size windows.

    Window splitting logic:
    - Create windows of size `window_size` seconds
    - For the last portion of the video:
      - If remainder < min_last_window: keep as single window
      - If remainder >= min_last_window: split into two windows

    Examples:
    - 27s video with 5s windows: [0-5], [5-10], [10-15], [15-20], [20-27] (5 windows)
    - 28s video with 5s windows: [0-5], [5-10], [10-15], [15-20], [20-25], [25-28] (6 windows)
    - 32s video with 5s windows: [0-5], [5-10], [10-15], [15-20], [20-25], [25-30], [30-32] (7 windows)
    """
    # Get video info (duration and fps)
    duration, fps = get_video_info(video_path)
    if duration is None or fps is None:
        return []

    # Calculate windows
    windows = []
    current_time = 0.0
    window_id = 0

    while current_time < duration:
        # Calculate remaining duration
        remaining = duration - current_time

        # If this is potentially the last window
        if remaining <= window_size + min_last_window:
            if remaining <= min_last_window:
                # Keep as single window
                windows.append((window_id, current_time, duration))
            else:
                # Split into two windows
                # First window is standard size
                windows.append((window_id, current_time, current_time + window_size))
                window_id += 1
                # Second window gets the remainder
                windows.append((window_id, current_time + window_size, duration))
            break
        else:
            # Standard window
            windows.append((window_id, current_time, current_time + window_size))
            current_time += window_size
            window_id += 1

    # Extract windows using ffmpeg
    extracted_files = []
    for window_id, start_time, end_time in windows:
        # Calculate frame range
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) - 1  # -1 because frame range is inclusive

        # Include frame range in filename
        output_file = (
            output_dir / f"task_{task_id}_episode_{episode_id}_window_{window_id}_frame_{start_frame}-{end_frame}.mp4"
        )

        # ffmpeg command to extract segment
        cmd = [
            "ffmpeg",
            "-ss",
            str(start_time),
            "-i",
            str(video_path),
            "-t",
            str(end_time - start_time),
            "-c",
            "libx264",
            "-avoid_negative_ts",
            "make_zero",
            "-y",  # Overwrite output
            str(output_file),
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            extracted_files.append(output_file)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting window {window_id}: {e}")

    return extracted_files


def split_videos(
    data_dir, camera_name, task_ids=None, window_size=5.0, min_last_window=7.5, val_episode_ids=[]
) -> None:
    """Split all videos for specified camera into windows."""
    # Check ffmpeg availability
    if not check_ffmpeg():
        return

    observations_dir = Path(data_dir) / "observations"

    if not observations_dir.exists():
        print(f"Observations directory not found: {observations_dir}")
        return

    # Create output directory
    output_base = Path(data_dir).parent / f"agibot_{camera_name}"
    output_dir_train = output_base / "train" / "videos"
    output_dir_train.mkdir(parents=True, exist_ok=True)

    output_dir_val = output_base / "val" / "videos"
    output_dir_val.mkdir(parents=True, exist_ok=True)

    print(f"\nSplitting videos into {window_size}s windows")
    print(f"Output directory: {output_dir_train}, {output_dir_val}")
    print(f"Camera: {camera_name}.mp4")
    print(f"Minimum last window size: {min_last_window}s")
    print("-" * 50)

    # If task_ids is specified, only process those tasks
    if task_ids:
        task_dirs = [
            observations_dir / str(task_id) for task_id in task_ids if (observations_dir / str(task_id)).exists()
        ]
    else:
        task_dirs = [d for d in observations_dir.iterdir() if d.is_dir()]

    total_videos = 0
    total_windows = 0

    print("val_episode_ids:", val_episode_ids)
    for task_dir in sorted(task_dirs):
        task_id = task_dir.name
        print(f"\nProcessing task {task_id}:")

        # Find all episodes with the target video
        video_files = list(task_dir.rglob(f"{camera_name}.mp4"))

        for video_path in sorted(video_files):
            # Extract episode_id from path
            episode_id = video_path.parent.parent.name
            # print(episode_id)

            # Split video into windows
            if episode_id in val_episode_ids:
                mode = "val"
                output_dir = output_dir_val
            else:
                mode = "train"
                output_dir = output_dir_train

            print(f"  Splitting episode {episode_id} and saving as {mode} data...", end="", flush=True)
            # Use training output directory
            windows = split_video_into_windows(
                video_path, output_dir, task_id, episode_id, window_size, min_last_window
            )

            total_videos += 1
            total_windows += len(windows)

            duration, fps = get_video_info(video_path)
            if duration:
                print(f" ✓ ({duration:.1f}s @ {fps:.1f}fps → {len(windows)} windows)")
            else:
                print(" ✗ (failed to get video info)")

    print("\n" + "-" * 50)
    print("Split summary:")
    print(f"  - Total videos processed: {total_videos}")
    print(f"  - Total windows created: {total_windows}")
    print(f"  - Average windows per video: {total_windows/total_videos:.1f}" if total_videos > 0 else "N/A")
    print(f"  - Output directory for training: {output_dir_train}")
    print(f"  - Output directory for validation: {output_dir_val}")
    print("-" * 50)


def clean_observations(data_dir, camera_name, task_ids=None) -> None:
    """Clean observations directory, keeping only specified camera videos."""
    observations_dir = Path(data_dir) / "observations"

    if not observations_dir.exists():
        print(f"Observations directory not found: {observations_dir}")
        return

    print(f"\nCleaning observations - keeping only '{camera_name}.mp4' videos")
    print("-" * 50)

    total_removed_files = 0
    total_removed_dirs = 0
    total_kept_files = 0

    # If task_ids is specified, only process those tasks
    if task_ids:
        task_dirs = [
            observations_dir / str(task_id) for task_id in task_ids if (observations_dir / str(task_id)).exists()
        ]
    else:
        task_dirs = [d for d in observations_dir.iterdir() if d.is_dir()]

    for task_dir in task_dirs:
        print(f"\nProcessing task: {task_dir.name}")

        # Process each episode in the task
        for episode_dir in [d for d in task_dir.iterdir() if d.is_dir()]:
            # Remove non-video directories (depth, tactile, etc.)
            for subdir in [d for d in episode_dir.iterdir() if d.is_dir() and d.name != "videos"]:
                try:
                    # Count files before removing
                    file_count = sum(1 for _ in subdir.rglob("*") if _.is_file())
                    total_removed_files += file_count

                    # Remove the directory
                    shutil.rmtree(subdir)
                    total_removed_dirs += 1
                    print(f"  ✓ Removed {subdir.name}/ ({file_count} files)")
                except Exception as e:
                    print(f"  ✗ Error removing {subdir}: {e}")

            # Process videos directory
            videos_dir = episode_dir / "videos"
            if videos_dir.exists():
                # Keep only the specified camera video
                target_video = f"{camera_name}.mp4"
                kept = False

                for video_file in videos_dir.glob("*.mp4"):
                    if video_file.name == target_video:
                        total_kept_files += 1
                        kept = True
                    else:
                        try:
                            video_file.unlink()
                            total_removed_files += 1
                        except Exception as e:
                            print(f"  ✗ Error removing {video_file}: {e}")

                if kept:
                    print(f"  ✓ Episode {episode_dir.name}: kept {target_video}")
                else:
                    print(f"  ⚠ Episode {episode_dir.name}: {target_video} not found")

    print("\n" + "-" * 50)
    print("Cleanup summary:")
    print(f"  - Kept files: {total_kept_files}")
    print(f"  - Removed files: {total_removed_files}")
    print(f"  - Removed directories: {total_removed_dirs}")
    print("-" * 50)


def extract_captions_from_jsonl(data_dir, camera_name, val_episode_ids) -> bool:
    """Extract captions from JSONL file and save as individual text files."""
    # Look for the JSONL file
    jsonl_path = Path(data_dir).parent / f"agibot_{camera_name}.jsonl"

    if not jsonl_path.exists():
        return False

    print(f"\nFound caption JSONL file: {jsonl_path}")
    print("Extracting captions to individual text files...")

    # Create metas directory
    output_base = Path(data_dir).parent / f"agibot_{camera_name}"
    metas_dir_train = output_base / "train" / "metas"
    metas_dir_train.mkdir(parents=True, exist_ok=True)
    metas_dir_val = output_base / "val" / "metas"
    metas_dir_val.mkdir(parents=True, exist_ok=True)

    caption_count_train = 0
    caption_count_val = 0

    # Read JSONL file and extract captions
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line)
                    video_clip = data.get("video_clip", "")
                    caption = data.get("caption", "")
                    episode_id = data.get("episode_id", "")

                    if episode_id in val_episode_ids:
                        # Save to validation metas directory
                        mode = "val"
                        metas_dir = metas_dir_val
                    else:
                        # Save to training metas directory
                        mode = "train"
                        metas_dir = metas_dir_train

                    if video_clip and caption:
                        # Create text file with video_clip name
                        txt_path = metas_dir / f"{video_clip}.txt"
                        with open(txt_path, "w") as txt_file:
                            txt_file.write(caption)
                        if mode == "train":
                            caption_count_train += 1
                        else:
                            caption_count_val += 1

        print(f"  ✓ Extracted {caption_count_train} captions to {metas_dir_train}")
        print(f"  ✓ Extracted {caption_count_val} captions to {metas_dir_val}")
        return True

    except Exception as e:
        print(f"  ✗ Error processing JSONL file: {e}")
        return False


def validate_video_caption_correspondence(data_dir, camera_name) -> None:
    """Check if all videos have captions and vice versa."""

    def _validate_video_caption_correspondence(videos_dir, metas_dir) -> None:
        if not videos_dir.exists() or not metas_dir.exists():
            return

        print("\nValidating video-caption correspondence...")
        print("  - Videos directory:", videos_dir)
        print("  - Captions directory:", metas_dir)

        # Get all video files (without extension)
        video_files = set()
        for video_path in videos_dir.glob("*.mp4"):
            video_files.add(video_path.stem)  # filename without .mp4

        # Get all caption files (without extension)
        caption_files = set()
        for caption_path in metas_dir.glob("*.txt"):
            caption_files.add(caption_path.stem)  # filename without .txt

        # Check for videos without captions
        videos_without_captions = video_files - caption_files
        captions_without_videos = caption_files - video_files

        print(f"  - Total videos: {len(video_files)}")
        print(f"  - Total captions: {len(caption_files)}")

        if videos_without_captions:
            print(f"  ⚠ Videos without captions: {len(videos_without_captions)}")
            for video in sorted(list(videos_without_captions))[:5]:  # Show first 5
                print(f"    • {video}")
            if len(videos_without_captions) > 5:
                print(f"    ... and {len(videos_without_captions) - 5} more")
        else:
            print("  ✓ All videos have corresponding captions")

        if captions_without_videos:
            print(f"  ⚠ Captions without videos: {len(captions_without_videos)}")
            for caption in sorted(list(captions_without_videos))[:5]:  # Show first 5
                print(f"    • {caption}")
            if len(captions_without_videos) > 5:
                print(f"    ... and {len(captions_without_videos) - 5} more")
        else:
            print("  ✓ All captions have corresponding videos")

        if not videos_without_captions and not captions_without_videos:
            print("  ✓ Perfect correspondence: all videos have captions and all captions have videos")

        print("-" * 50)

        return

    output_base = Path(data_dir).parent / f"agibot_{camera_name}"

    videos_dir_train = output_base / "train" / "videos"
    metas_dir_train = output_base / "train" / "metas"
    videos_dir_val = output_base / "val" / "videos"
    metas_dir_val = output_base / "val" / "metas"

    _validate_video_caption_correspondence(videos_dir_train, metas_dir_train)
    _validate_video_caption_correspondence(videos_dir_val, metas_dir_val)

    return


def main() -> None:
    args = parse_args()

    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)

    # If split-only mode, skip downloading and go straight to video splitting
    if args.split_only:
        print(f"Split-only mode: splitting existing videos in {args.data_dir}")
        print(f"Camera: {args.camera}")
        print(f"Task IDs: {args.task_ids if args.task_ids else 'all'}")
        print("-" * 50)

        split_videos(
            args.data_dir, args.camera, args.task_ids, args.window_size, args.min_last_window, args.val_episode_ids
        )

        # Extract captions from JSONL if available
        if extract_captions_from_jsonl(args.data_dir, args.camera, args.val_episode_ids):
            # Validate correspondence
            validate_video_caption_correspondence(args.data_dir, args.camera)

        return

    # If delete-only mode, skip downloading and go straight to cleaning
    if args.delete_only:
        print(f"Delete-only mode: cleaning existing data in {args.data_dir}")
        print(f"Camera to keep: {args.camera}")
        print(f"Task IDs: {args.task_ids if args.task_ids else 'all'}")
        print("-" * 50)

        clean_observations(args.data_dir, args.camera, args.task_ids)
        return

    # Generate allow patterns for the specified task IDs
    allow_patterns = get_allow_patterns(args.task_ids)

    print(f"Downloading data for task IDs: {args.task_ids}")
    print(f"Data directory: {args.data_dir}")
    print(f"Repository: {args.repo_id}")
    print(f"Camera to keep: {args.camera}")
    print("-" * 50)

    # Download the subset of the dataset
    try:
        print("Starting download...")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.data_dir,
            allow_patterns=allow_patterns,
        )
        print("✓ Download completed successfully")

    except Exception as e:
        print(f"✗ Error during download: {e}")
        return

    print("-" * 50)

    # Extract tar files and remove them
    print("Extracting tar files...")
    extract_tar_files(args.data_dir, remove_tar=not args.keep_tar)

    print("-" * 50)

    # Clean observations to keep only specified camera videos
    clean_observations(args.data_dir, args.camera, args.task_ids)

    print("-" * 50)
    print("✓ Data preparation completed!")

    # Print summary of downloaded data
    data_path = Path(args.data_dir)
    if data_path.exists():
        print("\nFinal structure:")

        # Check task_info files
        task_info_dir = data_path / "task_info"
        if task_info_dir.exists():
            task_files = list(task_info_dir.glob("task_*.json"))
            print(f"  - Task info files: {len(task_files)}")
            for tf in sorted(task_files):
                print(f"    • {tf.name}")

        # Check observations directories
        obs_dir = data_path / "observations"
        if obs_dir.exists():
            task_dirs = [d for d in obs_dir.iterdir() if d.is_dir()]
            print(f"  - Observation tasks: {len(task_dirs)}")
            for td in sorted(task_dirs):
                episode_count = len([d for d in td.iterdir() if d.is_dir()])
                # Count videos in each task
                video_count = sum(1 for _ in td.rglob(f"{args.camera}.mp4"))
                print(f"    • Task {td.name}: {episode_count} episodes, {video_count} {args.camera}.mp4 files")

    print("\n" + "=" * 50)
    print("NEXT STEP: To split videos into windows, run:")
    print(
        f"  python {Path(__file__).name} --split-only --data_dir {args.data_dir} --camera {args.camera} --task_ids {' '.join(map(str, args.task_ids))}"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
