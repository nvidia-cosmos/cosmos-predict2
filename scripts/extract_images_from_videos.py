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

import imageio

"""
Example command:
python -m scripts.extract_images_from_videos --input_dataset_dir datasets/cosmos_nemo_assets --output_dataset_dir datasets/cosmos_nemo_assets_images --stride 30
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from videos and save as images")
    parser.add_argument(
        "--input_dataset_dir", type=str, required=True, help="Path to the dataset directory containing videos"
    )
    parser.add_argument(
        "--output_dataset_dir", type=str, required=True, help="Path to the dataset directory containing videos"
    )
    parser.add_argument("--stride", type=int, default=30, help="Stride for frame extraction (default: 30)")
    return parser.parse_args()


def main(args) -> None:
    videos_dir = os.path.join(args.input_dataset_dir, "videos")
    # Ensure the dataset directory exists
    if not os.path.exists(videos_dir):
        raise FileNotFoundError(f"Videos directory {videos_dir} does not exist.")

    # Create output directory for images
    output_images_dir = os.path.join(args.output_dataset_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)

    # Get the list of video files in the dataset directory
    video_files = [filename for filename in os.listdir(videos_dir) if filename.endswith((".mp4"))]

    global_count = 0
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        print(f"Extracting frames from video: {video_path}")

        video_basename = os.path.splitext(video_file)[0]
        # Read the video frames
        reader = imageio.get_reader(video_path)

        count = 0  # count for the saved images
        for i, frame in enumerate(reader):
            if i % args.stride == 0:  # Apply stride
                # save a frame as an image
                output_image_path = os.path.join(output_images_dir, f"{video_basename}_{count:08d}.jpg")
                imageio.v3.imwrite(output_image_path, frame)

                count += 1

        global_count += count

    print(f"Total frames saved: {global_count}")

    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
