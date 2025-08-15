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

from imaginaire.auxiliary.text_encoder import CosmosReason1TextEncoder, CosmosReason1TextEncoderConfig

"""example command
python -m scripts.get_cr1_embeddings --dataset_path datasets/hdvila
"""


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute CR1 embeddings for text prompts")
    parser.add_argument("--dataset_path", type=str, default="datasets/hdvila", help="Root path to the dataset")
    return parser.parse_args()


def main(args) -> None:
    metas_dir = os.path.join(args.dataset_path, "metas")
    metas_list = [
        os.path.join(metas_dir, filename) for filename in sorted(os.listdir(metas_dir)) if filename.endswith(".txt")
    ]

    cr1_dir = os.path.join(args.dataset_path, "cr1")
    os.makedirs(cr1_dir, exist_ok=True)

    # Initialize CR1
    encoder_config = CosmosReason1TextEncoderConfig()
    encoder = CosmosReason1TextEncoder(config=encoder_config)

    for meta_filename in metas_list:
        cr1_filename = os.path.join(cr1_dir, os.path.basename(meta_filename).replace(".txt", ".pickle"))
        if os.path.exists(cr1_filename):
            # Skip if the file already exists
            continue

        with open(meta_filename) as fp:
            prompt = fp.read().strip()

        # Compute CR1 embeddings
        max_length = args.max_length
        encoded_text, mask_bool = encoder.encode_prompts(
            prompt, max_length=max_length, return_mask=True
        )  # list of np.ndarray in (len, 1024)
        attn_mask = mask_bool.long()
        lengths = attn_mask.sum(dim=1).cpu()

        encoded_text = encoded_text.cpu().numpy().astype(np.float16)

        # trim zeros to save space
        encoded_text = [encoded_text[batch_id][: lengths[batch_id]] for batch_id in range(encoded_text.shape[0])]

        # Save CR1 embeddings as pickle file
        with open(cr1_filename, "wb") as fp:
            pickle.dump(encoded_text, fp)


if __name__ == "__main__":
    args = parse_args()
    main(args)
