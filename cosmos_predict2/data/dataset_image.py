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

import os
import pickle
import traceback
import warnings

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2.data.dataset_utils import Resize_Preprocess, ToTensorImage
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig
from imaginaire.utils import log

"""
Test the dataset with the following command:
python -m cosmos_predict2.data.dataset_image
"""


class ImageDataset(Dataset):
    def __init__(self, dataset_dir: str, image_size: list):
        """Dataset class for loading text-to-image generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            image_size (list): Target size [H,W] for video frames

        Returns dict with:
            - image: RGB frames tensor [C,H,W]
        """

        super().__init__()
        self.dataset_dir = dataset_dir

        image_dir = os.path.join(self.dataset_dir, "images")
        self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl")

        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.image_paths = sorted(self.image_paths)

        # remove video paths that does not have t5_embedding
        self.image_paths = [
            path
            for path in self.image_paths
            if os.path.exists(os.path.join(self.t5_dir, os.path.basename(path).replace(".jpg", ".pickle")))
        ]
        log.info(f"{len(self.image_paths)} images in total")

        self.wrong_number = 0
        self.preprocess = T.Compose([ToTensorImage(), Resize_Preprocess(tuple(image_size))])

    def __str__(self) -> str:
        return f"{len(self.image_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.image_paths)

    def _get_image(self, image_path) -> torch.Tensor:
        image = imageio.v3.imread(image_path)
        image = torch.from_numpy(image).permute(2, 0, 1)  # (c, h, w)
        image = self.preprocess(image)
        image = torch.clamp(image, 0.0, 1.0).sub_(0.5).div_(0.5)  # Normalize to [-1, 1]

        return image

    def __getitem__(self, index):
        try:
            data = dict()
            image = self._get_image(self.image_paths[index])
            image_path = self.image_paths[index]
            t5_embedding_path = os.path.join(
                self.t5_dir,
                os.path.basename(image_path).replace(".jpg", ".pickle"),
            )

            _, h, w = image.shape

            data["images"] = image
            with open(t5_embedding_path, "rb") as f:
                t5_embedding = pickle.load(f)[0]  # [n_tokens, CosmosTextEncoderConfig.EMBED_DIM]
            n_tokens = t5_embedding.shape[0]
            if n_tokens < CosmosTextEncoderConfig.NUM_TOKENS:
                t5_embedding = np.concatenate(
                    [
                        t5_embedding,
                        np.zeros(
                            (CosmosTextEncoderConfig.NUM_TOKENS - n_tokens, CosmosTextEncoderConfig.EMBED_DIM),
                            dtype=np.float32,
                        ),
                    ],
                    axis=0,
                )
            t5_text_mask = torch.zeros(CosmosTextEncoderConfig.NUM_TOKENS, dtype=torch.int64)
            t5_text_mask[:n_tokens] = 1

            data["t5_text_embeddings"] = torch.from_numpy(t5_embedding)
            data["t5_text_mask"] = t5_text_mask
            data["fps"] = torch.ones(1, dtype=torch.float) * 16  # Dummy FPS for images
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = torch.ones(1)
            data["padding_mask"] = torch.zeros(1, h, w)

            return data
        except Exception:
            warnings.warn(  # noqa: B028
                f"Invalid data encountered: {self.image_paths[index]}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")  # noqa: B028
            warnings.warn(traceback.format_exc())  # noqa: B028
            self.wrong_number += 1
            log.info(self.wrong_number, rank0_only=False)
            return self[np.random.randint(len(self.samples))]


if __name__ == "__main__":
    dataset = ImageDataset(
        dataset_dir="datasets/cosmos_nemo_assets_images",
        image_size=[480, 832],
    )

    indices = [0, 13, -1]
    for idx in indices:
        data = dataset[idx]
        log.info(f"{idx=} {data['images'].sum()=}\n{data['images'].shape=}\n{data['t5_text_embeddings'].shape=}\n---")
