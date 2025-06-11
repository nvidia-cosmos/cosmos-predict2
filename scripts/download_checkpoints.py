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
import hashlib
import os

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download NVIDIA Cosmos Predict2 diffusion models from Hugging Face")
    parser.add_argument(
        "--model_sizes",
        nargs="*",
        default=["2B", "14B"],
        choices=["2B", "14B"],
        help="Which model sizes to download. Possible values: 7B, 14B",
    )
    parser.add_argument(
        "--model_types",
        nargs="*",
        default=["text2image", "video2world"],
        choices=["text2image", "video2world"],
        help="Which model types to download. Possible values: text2image, video2world",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save the downloaded checkpoints."
    )
    parser.add_argument(
        "--verify_md5", action="store_true", default=False, help="Verify MD5 checksums of existing files."
    )
    args = parser.parse_args()
    return args


MD5_CHECKSUM_LOOKUP = {
    # Cosmos-Predict2 models
    "nvidia/Cosmos-Predict2-2B-Text2Image/model.pt": "0336b218dffe32848d075ba7606c522b",
    "nvidia/Cosmos-Predict2-14B-Text2Image/model.pt": "3bc68c3384b4985120b13f964e9d6c03",
    "nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt": "66157e5aece452a5a121cbb6fe0580ac",
    "nvidia/Cosmos-Predict2-14B-Video2World/model-720p-16fps.pt": "79c86aa3c91897d9d402e70a3525ed96",
    "nvidia/Cosmos-Predict2-2B-Text2Image/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth": "854fcb755005951fa5b329799af6199f",
    "nvidia/Cosmos-Predict2-14B-Text2Image/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth": "854fcb755005951fa5b329799af6199f",
    "nvidia/Cosmos-Predict2-2B-Video2World/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth": "854fcb755005951fa5b329799af6199f",
    "nvidia/Cosmos-Predict2-14B-Video2World/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth": "854fcb755005951fa5b329799af6199f",
    # Cosmos-Reason1-7B
    "nvidia/Cosmos-Reason1-7B/model-00001-of-00004.safetensors": "40e531a49383b8ea73273d2e1fe59a4f",
    "nvidia/Cosmos-Reason1-7B/model-00002-of-00004.safetensors": "4d5684fca6b056f09f825fe1e436c3ab",
    "nvidia/Cosmos-Reason1-7B/model-00003-of-00004.safetensors": "639e5a2041a4332aefff57a7d7595245",
    "nvidia/Cosmos-Reason1-7B/model-00004-of-00004.safetensors": "63f9e7855dcc6d382d43c2e2411991f1",
    # T5 text encoder
    "google-t5/t5-11b/pytorch_model.bin": "f890878d8a162e0045a25196e27089a3",
    # Cosmos-Guardrail1
    "nvidia/Cosmos-Guardrail1/face_blur_filter/Resnet50_Final.pth": "bce939bc22d8cec91229716dd932e56e",
    "nvidia/Cosmos-Guardrail1/video_content_safety_filter/safety_filter.pt": "b46dc2ad821fc3b0d946549d7ade19cf",
    "nvidia/Cosmos-Guardrail1/video_content_safety_filter/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3/model.safetensors": "f4c887e55e159f96453e18a1d6ca984f",
    # Meta Llama guard
    "meta-llama/Llama-Guard-3-8B/model-00001-of-00004.safetensors": "5748060ae47b335dc19263060c921a54",
    "meta-llama/Llama-Guard-3-8B/model-00002-of-00004.safetensors": "89e7dce10959cab81c0d09468a873f33",
    "meta-llama/Llama-Guard-3-8B/model-00003-of-00004.safetensors": "e7e5f50ecdb02a946d071373a52c01b8",
    "meta-llama/Llama-Guard-3-8B/model-00004-of-00004.safetensors": "a94c830cafe5e1d8d54ea2f83378b234",
}


def validate_files(checkpoints_dir, model_name, verify_md5=False):
    for key, value in MD5_CHECKSUM_LOOKUP.items():
        if key.startswith(model_name + "/"):
            file_path = os.path.join(checkpoints_dir, key)
            # File must exist
            if not os.path.exists(file_path):
                print(f"\033[93mCheckpoint {key} does not exist.\033[0m")
                return False
            # Verify MD5 checksum if requested
            if verify_md5:
                print(f"Verifying MD5 checksum of checkpoint {key}...")
                with open(file_path, "rb") as f:
                    file_md5 = hashlib.md5(f.read()).hexdigest()
                if file_md5 != value:
                    print(f"\033[93mMD5 checksum of checkpoint {key} does not match.\033[0m")
                    return False
    if verify_md5:
        print(f"\033[92mModel checkpoints for {model_name} exist with matched MD5 checksums.\033[0m")
    else:
        print(f"\033[92mFiles for {model_name} already exist\033[0m \033[93m(MD5 not verified).\033[0m")
    return True


def download_model(checkpoint_dir, repo_id, verify_md5=False, **download_kwargs):
    local_dir = os.path.join(checkpoint_dir, repo_id)
    try:
        # Check if files exist and optionally verify checksums
        if not validate_files(checkpoint_dir, repo_id, verify_md5):
            print(f"Downloading {repo_id} to {local_dir}...")
            snapshot_download(repo_id=repo_id, local_dir=local_dir, force_download=True, **download_kwargs)
            print(f"\033[92mSuccessfully downloaded {repo_id}\033[0m")
    except Exception as e:
        print(f"\033[91mError downloading {repo_id}: {e}\033[0m")
    print("---------------------")


def main(args):
    # Create local checkpoints folder
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Download the Cosmos-Predict2 models
    model_size_mapping = {"2B": "Cosmos-Predict2-2B", "14B": "Cosmos-Predict2-14B"}
    model_type_mapping = {"text2image": "Text2Image", "video2world": "Video2World"}
    for size in args.model_sizes:
        for type in args.model_types:
            repo_id = f"nvidia/{model_size_mapping[size]}-{model_type_mapping[type]}"
            download_model(args.checkpoint_dir, repo_id, verify_md5=args.verify_md5)
    if "video2world" in args.model_types:
        download_model(args.checkpoint_dir, "nvidia/Cosmos-Reason1-7B", verify_md5=args.verify_md5)

    # Download T5 model
    download_model(args.checkpoint_dir, "google-t5/t5-11b", verify_md5=args.verify_md5, ignore_patterns="tf_model.h5")

    # Download the guardrail models
    download_model(args.checkpoint_dir, "nvidia/Cosmos-Guardrail1", verify_md5=args.verify_md5)
    download_model(
        args.checkpoint_dir, "meta-llama/Llama-Guard-3-8B", verify_md5=args.verify_md5, ignore_patterns="original/*"
    )

    print("Checkpoint downloading done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
