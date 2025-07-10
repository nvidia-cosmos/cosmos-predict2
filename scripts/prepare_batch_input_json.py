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
import glob
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    save_path = args.save_path
    output_path = args.output_path

    input_files = glob.glob(os.path.join(dataset_path, "*.jpg")) + glob.glob(os.path.join(dataset_path, "*.png"))
    output_json = []
    for input_file in input_files:
        print(input_file)
        prompt_file = input_file.replace(".jpg", ".txt").replace(".png", ".txt")
        if not os.path.exists(prompt_file):
            # The "dream_gen_benchmark/gr1_object/12_Use the right hand to pick up bok choy from top black shelf to paper bag.png" is missing a . before .png
            prompt_file = input_file.replace(".jpg", "..txt").replace(".png", "..txt")

        output_json.append(
            {
                "input_video": input_file,
                "prompt": open(prompt_file).read(),
                "output_video": os.path.join(
                    save_path, os.path.basename(input_file.replace(".jpg", ".mp4").replace(".png", ".mp4"))
                ),
            }
        )
    print(f"Saved {len(output_json)} items to {output_path}")
    with open(output_path, "w") as f:
        json.dump(output_json, f, indent=4)


if __name__ == "__main__":
    main()
