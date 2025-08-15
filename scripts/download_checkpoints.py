#!/usr/bin/env -S uv run --script
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

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface-hub",
# ]
# [tool.uv]
# exclude-newer = "2025-08-05T00:00:00Z"
# ///

import argparse
import os

import huggingface_hub
from huggingface_hub import snapshot_download

"""Download NVIDIA Cosmos Predict2 diffusion models from Hugging Face."""

MODEL_SIZE_MAPPING = {
    "0.6B": "Cosmos-Predict2-0.6B",
    "2B": "Cosmos-Predict2-2B",
    "14B": "Cosmos-Predict2-14B",
}
MODEL_TYPE_MAPPING = {
    "multiview": "Multiview",
    "sample_gr00t_dreams_droid": "Sample-GR00T-Dreams-DROID",
    "sample_gr00t_dreams_gr1": "Sample-GR00T-Dreams-GR1",
    "text2image": "Text2Image",
    "video2world": "Video2World",
}
REPO_ID_MAPPING = {
    "google-t5/t5-11b": "90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3",
    "meta-llama/Llama-Guard-3-8B": "7327bd9f6efbbe6101dc6cc4736302b3cbb6e425",
    "nvidia/Cosmos-Guardrail1": "d6d4bfa899a71454a700907664f3e88f503950cf",
    "nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID": "ebc29d30a3fab504bcc779a85bca073d14ad39f9",
    "nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1": "926f34dd83860c5406ab4b400e156c3fd6fe6c0d",
    "nvidia/Cosmos-Predict2-14B-Text2Image": "015332720f70dd7b497c1cff9fd0c936a77f160b",
    "nvidia/Cosmos-Predict2-14B-Video2World": "03b03a377fede782647afac998f674d9f358e319",
    "nvidia/Cosmos-Predict2-2B-Multiview": "52f3731663eecdc998d9608f9b21ac4dcbdea6f1",
    "nvidia/Cosmos-Predict2-2B-Text2Image": "acdb5fde992a73ef0355f287977d002cbfd127e0",
    "nvidia/Cosmos-Predict2-2B-Video2World": "f50c09f5d8ab133a90cac3f4886a6471e9ba3f18",
    "nvidia/Cosmos-Predict2-0.6B-Text2Image": "80e9f1599778499e9b5435179861871b5a3561d4",
    "nvidia/Cosmos-Reason1-7B": "8fe96c1fa10db9e666b6fa6a87fea57dd9635649",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_sizes",
        nargs="*",
        default=["2B", "14B"],
        choices=["0.6B", "2B", "14B"],
        help="Which model sizes to download.",
    )
    parser.add_argument(
        "--model_types",
        nargs="*",
        default=[
            "text2image",
            "video2world",
            "sample_action_conditioned",
            "sample_gr00t_dreams_gr1",
            "sample_gr00t_dreams_droid",
            "multiview",
        ],
        choices=[
            "text2image",
            "video2world",
            "sample_action_conditioned",
            "sample_gr00t_dreams_gr1",
            "sample_gr00t_dreams_droid",
            "multiview",
        ],
        help="Which model types to download.",
    )
    parser.add_argument(
        "--fps",
        nargs="*",
        default=["16"],
        choices=["10", "16"],
        help="Which fps to download. This is only for Video2World models and will be ignored for other model_types",
    )
    parser.add_argument(
        "--resolution",
        nargs="*",
        default=["720"],
        choices=["480", "720"],
        help="Which resolution to download. This is only for Video2World models and will be ignored for other model_types",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save the downloaded checkpoints."
    )
    parser.add_argument(
        "--natten",
        action="store_true",
        default=False,
        help="Download Video2World + NATTEN (sparse attention) checkpoints.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run the script and print the download commands without actually downloading the files.",
    )
    args = parser.parse_args()
    return args


def main(args):
    # Create local checkpoints folder
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    def download(repo_id: str, **download_kwargs):
        local_dir = os.path.join(args.checkpoint_dir, repo_id)
        print(f"Downloading {repo_id} to {local_dir}...")
        if repo_id in REPO_ID_MAPPING:
            revision = REPO_ID_MAPPING[repo_id]
        else:
            revision = huggingface_hub.HfApi().repo_info(repo_id=repo_id).sha
            print(f"Revision: {revision}")
        if not args.dry_run:
            try:
                snapshot_download(repo_id=repo_id, local_dir=local_dir, revision=revision, **download_kwargs)
            except Exception as e:
                print(f"\033[91mError downloading {repo_id}: {e}\033[0m")
        print("-" * 20)

    if "text2image" in args.model_types:
        for size in args.model_sizes:
            download(f"nvidia/{MODEL_SIZE_MAPPING[size]}-{MODEL_TYPE_MAPPING['text2image']}")

    if "video2world" in args.model_types:
        for size in args.model_sizes:
            for fps in args.fps:
                for res in args.resolution:
                    allow_patterns = [f"model-{res}p-{fps}fps.pt"]
                    if args.natten and res == "720":
                        allow_patterns.append(f"model-{res}p-{fps}fps-natten.pt")
                    download(
                        f"nvidia/{MODEL_SIZE_MAPPING[size]}-{MODEL_TYPE_MAPPING['video2world']}",
                        allow_patterns=allow_patterns,
                    )

            download(
                f"nvidia/{MODEL_SIZE_MAPPING[size]}-{MODEL_TYPE_MAPPING['video2world']}", allow_patterns="tokenizer/*"
            )
        download("nvidia/Cosmos-Reason1-7B")

    if "multiview" in args.model_types:
        download("nvidia/Cosmos-Predict2-2B-Multiview", allow_patterns="*.pt")

    if "sample_action_conditioned" in args.model_types:
        if "2B" in args.model_sizes and "480" in args.resolution and "4" in args.fps:
            download("nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned")
        else:
            print("Sample Action Conditioned model is only available for 2B model size, 480P and 4FPS. Skipping...")

    # Download the GR00T models
    if "sample_gr00t_dreams_gr1" in args.model_types:
        download(f"nvidia/{MODEL_SIZE_MAPPING['14B']}-{MODEL_TYPE_MAPPING['sample_gr00t_dreams_gr1']}")
    if "sample_gr00t_dreams_droid" in args.model_types:
        download(f"nvidia/{MODEL_SIZE_MAPPING['14B']}-{MODEL_TYPE_MAPPING['sample_gr00t_dreams_droid']}")

    # Download T5 model
    download("google-t5/t5-11b", ignore_patterns=["tf_model.h5"])

    # Download the guardrail models
    download("nvidia/Cosmos-Guardrail1")
    download("meta-llama/Llama-Guard-3-8B", ignore_patterns=["original/*"])

    print("Checkpoint downloading done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
