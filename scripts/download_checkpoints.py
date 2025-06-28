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
import fnmatch
import hashlib
import os

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(description="Download NVIDIA Cosmos Predict2 diffusion models from Hugging Face")
    parser.add_argument(
        "--model_sizes",
        nargs="*",
        default=["2B", "14B"],
        choices=["0.6B", "2B", "14B"],
        help="Which model sizes to download. Possible values: 2B, 14B",
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
        help="Which model types to download. Possible values: text2image, video2world, sample_action_conditioned",
    )
    parser.add_argument(
        "--fps",
        nargs="*",
        default=["16"],
        choices=["16", "10"],
        help="Which fps to download. Possible values: 16, 10. This is only for Video2World models and will be ignored for other model_types",
    )
    parser.add_argument(
        "--resolution",
        nargs="*",
        default=["720"],
        choices=["480", "720"],
        help="Which resolution to download. Possible values: 480, 720. This is only for Video2World models and will be ignored for other model_types",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save the downloaded checkpoints."
    )
    parser.add_argument(
        "--verify_md5", action="store_true", default=False, help="Verify MD5 checksums of existing files."
    )
    parser.add_argument(
        "--natten",
        action="store_true",
        default=False,
        help="Download Video2World + NATTEN (sparse attention) checkpoints.",
    )
    args = parser.parse_args()
    return args


MD5_CHECKSUM_LOOKUP = {
    # Cosmos-Predict2 models
    "nvidia/Cosmos-Predict2-0.6B-Text2Image/model.pt": "f93f9655fce91840723cdb28ebbfa1fe",
    "nvidia/Cosmos-Predict2-2B-Text2Image/model.pt": "0336b218dffe32848d075ba7606c522b",
    "nvidia/Cosmos-Predict2-14B-Text2Image/model.pt": "3bc68c3384b4985120b13f964e9d6c03",
    # 8 variants of Video2World models
    "nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt": "66157e5aece452a5a121cbb6fe0580ac",
    "nvidia/Cosmos-Predict2-2B-Video2World/model-720p-10fps.pt": "1884e792fe3a57c3384c68ff1c0ef0d3",
    "nvidia/Cosmos-Predict2-2B-Video2World/model-480p-10fps.pt": "af1e352c5a8fb52ee1de19e307731b6b",
    "nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt": "dad6861b72d595f6fae8d91c08a58e9e",
    "nvidia/Cosmos-Predict2-14B-Video2World/model-720p-16fps.pt": "79c86aa3c91897d9d402e70a3525ed96",
    "nvidia/Cosmos-Predict2-14B-Video2World/model-720p-10fps.pt": "34730e3d5e65c4c590f3a88ca3fd4e74",
    "nvidia/Cosmos-Predict2-14B-Video2World/model-480p-10fps.pt": "b1dcd8adbe82e69496532d1e237c7022",
    "nvidia/Cosmos-Predict2-14B-Video2World/model-480p-16fps.pt": "53a04f51880272d9f4a5c4460b82966d",
    "nvidia/Cosmos-Predict2-0.6B-Text2Image/tokenizer/tokenizer.pth": "6eddc5eaf9e9e527803184f6daa65d25",
    "nvidia/Cosmos-Predict2-2B-Text2Image/tokenizer/tokenizer.pth": "854fcb755005951fa5b329799af6199f",
    "nvidia/Cosmos-Predict2-14B-Text2Image/tokenizer/tokenizer.pth": "854fcb755005951fa5b329799af6199f",
    "nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth": "854fcb755005951fa5b329799af6199f",
    "nvidia/Cosmos-Predict2-14B-Video2World/tokenizer/tokenizer.pth": "854fcb755005951fa5b329799af6199f",
    # Video2World Sparse Variants
    "nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps-natten.pt": "91355cc9979f47aeb7ee991193e88305",
    "nvidia/Cosmos-Predict2-2B-Video2World/model-720p-10fps-natten.pt": "4ef2b03da1ca0888e3a4054dc8b4b2f0",
    "nvidia/Cosmos-Predict2-14B-Video2World/model-720p-16fps-natten.pt": "09167672edf4bcd456318d8498cb6f36",
    "nvidia/Cosmos-Predict2-14B-Video2World/model-720p-10fps-natten.pt": "86d5e27ac75021798ab594f257600e50",
    # Multiview models
    "nvidia/Cosmos-Predict2-2B-Multiview/model-720p-10fps-7views-29frames.pt": "0336b218dffe32848d075ba7606c522b",
    # Cosmos-Reason1-7B
    "nvidia/Cosmos-Reason1-7B/model-00001-of-00004.safetensors": "90198d3b3dab5a00b7b9288cecffa5e9",
    "nvidia/Cosmos-Reason1-7B/model-00002-of-00004.safetensors": "6bde197d212f2a83ae19585b87de500e",
    "nvidia/Cosmos-Reason1-7B/model-00003-of-00004.safetensors": "c999ec0bc79fccf2f2cdba598d4e951f",
    "nvidia/Cosmos-Reason1-7B/model-00004-of-00004.safetensors": "232e93dfc82361ea8b0678fffc8660ef",
    # Cosmos-Predict2-2B-Sample-Action-Conditioned
    "nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned/model-480p-4fps.pt": "b4db0f266cc487f1242dc09a082c6dd5",
    # Cosmos-Predict2 Post-training models
    "nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID/model-720p-16fps.pt": "af799ec678f6f18e3b3cfe3c1d9c591b",
    "nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1/model-720p-16fps.pt": "b7f92ff4d0943ab7477ad873fb17015558ee597897782032bdfeb1db2aee0796",
    "nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-DROID/tokenizer/tokenizer.pth": "854fcb755005951fa5b329799af6199f",
    "nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1/tokenizer/tokenizer.pth": "854fcb755005951fa5b329799af6199f",
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


def validate_files(checkpoints_dir, model_name, verify_md5=False, **download_kwargs):
    for key, value in MD5_CHECKSUM_LOOKUP.items():
        if key.startswith(model_name + "/"):
            if "allow_patterns" in download_kwargs:
                # only check if the key matches the allow_patterns
                relative_path = key[len(model_name + "/") :]
                if not fnmatch.fnmatch(relative_path, download_kwargs["allow_patterns"]):
                    continue
            if "ignore_patterns" in download_kwargs:
                # only check if the key does not match the ignore_patterns
                relative_path = key[len(model_name + "/") :]
                if any(fnmatch.fnmatch(relative_path, pattern) for pattern in download_kwargs["ignore_patterns"]):
                    continue
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
        if not validate_files(checkpoint_dir, repo_id, verify_md5, **download_kwargs):
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
    model_size_mapping = {"0.6B": "Cosmos-Predict2-0.6B", "2B": "Cosmos-Predict2-2B", "14B": "Cosmos-Predict2-14B"}
    model_type_mapping = {
        "text2image": "Text2Image",
        "video2world": "Video2World",
        "sample_gr00t_dreams_gr1": "Sample-GR00T-Dreams-GR1",
        "sample_gr00t_dreams_droid": "Sample-GR00T-Dreams-DROID",
        "multiview": "Multiview",
    }
    if "text2image" in args.model_types:
        for size in args.model_sizes:
            repo_id = f"nvidia/{model_size_mapping[size]}-{model_type_mapping['text2image']}"
            download_model(args.checkpoint_dir, repo_id, verify_md5=args.verify_md5)

    if "video2world" in args.model_types:
        for size in args.model_sizes:
            for fps in args.fps:
                for res in args.resolution:
                    repo_id = f"nvidia/{model_size_mapping[size]}-{model_type_mapping['video2world']}"
                    allow_patterns = f"model-{res}p-{fps}fps.pt"
                    download_model(
                        args.checkpoint_dir, repo_id, verify_md5=args.verify_md5, allow_patterns=allow_patterns
                    )
                    # Sparse variant (if any)
                    if args.natten and res == "720":
                        download_model(
                            args.checkpoint_dir,
                            repo_id,
                            verify_md5=args.verify_md5,
                            allow_patterns=f"model-{res}p-{fps}fps-natten.pt",
                        )

            # donwload the remaining
            repo_id = f"nvidia/{model_size_mapping[size]}-{model_type_mapping['video2world']}"
            download_model(args.checkpoint_dir, repo_id, verify_md5=args.verify_md5, allow_patterns="tokenizer/*")
        download_model(args.checkpoint_dir, "nvidia/Cosmos-Reason1-7B", verify_md5=args.verify_md5)
    
    if "multiview" in args.model_types:
        repo_id = f"nvidia/Cosmos-Predict2-2B-Multiview"
        download_model(args.checkpoint_dir, repo_id, verify_md5=args.verify_md5, allow_patterns="*.pt")

    if "sample_action_conditioned" in args.model_types:
        print("NOTE: Sample Action Conditioned model is only available for 2B model size, 480P and 4FPS")
        repo_id = "nvidia/Cosmos-Predict2-2B-Sample-Action-Conditioned"
        download_model(args.checkpoint_dir, repo_id, verify_md5=args.verify_md5)

    # Download the GR00T models
    if "sample_gr00t_dreams_gr1" in args.model_types:
        repo_id = f"nvidia/{model_size_mapping['14B']}-{model_type_mapping['sample_gr00t_dreams_gr1']}"
        download_model(args.checkpoint_dir, repo_id, verify_md5=args.verify_md5)

    if "sample_gr00t_dreams_droid" in args.model_types:
        repo_id = f"nvidia/{model_size_mapping['14B']}-{model_type_mapping['sample_gr00t_dreams_droid']}"
        download_model(args.checkpoint_dir, repo_id, verify_md5=args.verify_md5)

    # Download T5 model
    download_model(args.checkpoint_dir, "google-t5/t5-11b", verify_md5=args.verify_md5, ignore_patterns=["tf_model.h5"])

    # Download the guardrail models
    download_model(args.checkpoint_dir, "nvidia/Cosmos-Guardrail1", verify_md5=args.verify_md5)
    download_model(
        args.checkpoint_dir, "meta-llama/Llama-Guard-3-8B", verify_md5=args.verify_md5, ignore_patterns=["original/*"]
    )

    print("Checkpoint downloading done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
