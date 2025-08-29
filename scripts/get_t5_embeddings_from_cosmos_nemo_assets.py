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
import pickle

import numpy as np

from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig
from imaginaire.constants import T5_MODEL_DIR

"""example command
python -m scripts.get_t5_embeddings_from_cosmos_nemo_assets --dataset_path datasets/cosmos_nemo_assets
"""


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute T5 embeddings for text prompts")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/cosmos_nemo_assets",
        help="Root path to the dataset",
    )
    parser.add_argument("--max_length", type=int, help="Maximum length of the text embedding")
    parser.add_argument("--prompt", type=str, default="A video of sks teal robot.", help="Text prompt for the dataset")
    parser.add_argument("--cache_dir", type=str, default=T5_MODEL_DIR, help="Directory to cache the T5 model")
    parser.add_argument("--is_image", action="store_true", help="Set if the dataset is image-based")
    return parser.parse_args()


def main(args) -> None:
    images_dir = os.path.join(args.dataset_path, "images")
    videos_dir = os.path.join(args.dataset_path, "videos")

    # Cosmos-NeMo-Assets come with videos only. A prompt is provided as an argument.
    metas_dir = os.path.join(args.dataset_path, "metas")
    os.makedirs(metas_dir, exist_ok=True)
    if args.is_image:
        metas_list = [
            os.path.join(metas_dir, filename.replace(".jpg", ".txt"))
            for filename in sorted(os.listdir(images_dir))
            if filename.endswith(".jpg")
        ]
    else:
        metas_list = [
            os.path.join(metas_dir, filename.replace(".mp4", ".txt"))
            for filename in sorted(os.listdir(videos_dir))
            if filename.endswith(".mp4")
        ]

    # Write txt files to match other dataset formats.
    for meta_filename in metas_list:
        if not os.path.exists(meta_filename):
            with open(meta_filename, "w") as fp:
                fp.write(args.prompt)

    t5_xxl_dir = os.path.join(args.dataset_path, "t5_xxl")
    os.makedirs(t5_xxl_dir, exist_ok=True)

    # Initialize T5
    encoder_config = CosmosT5TextEncoderConfig(ckpt_path=args.cache_dir)
    encoder = CosmosT5TextEncoder(config=encoder_config)

    # Compute T5 embeddings
    print(f"Computing T5 embeddings for the prompt: {args.prompt}")
    encoded_text, mask_bool = encoder.encode_prompts(
        args.prompt, max_length=args.max_length, return_mask=True
    )  # list of np.ndarray in (len, 1024)
    attn_mask = mask_bool.long()
    lengths = attn_mask.sum(dim=1).cpu()

    encoded_text = encoded_text.cpu().numpy().astype(np.float16)

    # trim zeros to save space
    encoded_text = [encoded_text[batch_id][: lengths[batch_id]] for batch_id in range(encoded_text.shape[0])]

    print(f"Saving T5 embeddings to {t5_xxl_dir}")
    for meta_filename in metas_list:
        t5_xxl_filename = os.path.join(t5_xxl_dir, os.path.basename(meta_filename).replace(".txt", ".pickle"))
        if os.path.exists(t5_xxl_filename):
            # Skip if the file already exists
            continue

        # Save T5 embeddings as pickle file
        with open(t5_xxl_filename, "wb") as fp:
            pickle.dump(encoded_text, fp)


if __name__ == "__main__":
    args = parse_args()
    main(args)
