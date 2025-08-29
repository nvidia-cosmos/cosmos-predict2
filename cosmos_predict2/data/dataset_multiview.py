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
Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict1/diffusion/training/datasets/dataset_multiview.py

Adapted from:
https://github.com/bytedance/IRASim/blob/main/dataset/dataset_3D.py
"""

import os
import pickle
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from cosmos_predict2.data.dataset_utils import (
    Resize_Preprocess,
    ToTensorVideo,
)
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig


class MultiviewDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        sequence_interval,
        num_frames,
        camera_keys,
        video_size,
        start_frame_interval=1,
        state_t=8,
        camera_to_view_id=None,
        front_camera_key=None,
        front_view_caption_only=False,
        is_train=True,
    ):
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            sequence_interval (int): Interval between sampled frames in a sequence
            num_frames (int): Number of frames to load per sequence
            video_size (list): Target size [H,W] for video frames
            front_camera_key (str): Key of the front camera
            front_view_caption_only (bool): Whether to use only the front view for captioning
            camera_to_view_id (dict): Dictionary mapping camera keys to view indices
        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        self.dataset_dir = dataset_dir
        self.start_frame_interval = start_frame_interval
        self.sequence_interval = sequence_interval
        self.sequence_length = num_frames
        self.camera_keys = camera_keys
        self.state_t = state_t
        self.H, self.W = video_size
        self.front_view_caption_only = front_view_caption_only
        video_dir = os.path.join(self.dataset_dir, "videos")
        self.video_paths = [
            os.path.join(video_dir, camera_keys[0], f) for f in os.listdir(os.path.join(video_dir, camera_keys[0]))
        ]
        self.video_paths = sorted(self.video_paths)
        # Use 10% of the videos for validation rounded up to the nearest 1
        cutoff = int(len(self.video_paths) * 0.1) + 1
        if is_train:
            self.video_paths = self.video_paths[:-cutoff]
        else:
            self.video_paths = self.video_paths[-cutoff:]
        print(f"{len(self.video_paths)} videos in total")

        self.t5_dir = os.path.join(self.dataset_dir, "t5_xxl")
        self.samples = self._init_samples(self.video_paths)
        self.samples = sorted(self.samples, key=lambda x: (x["video_path"], x["frame_ids"][0]))
        print(f"{len(self.samples)} samples in total")
        self.wrong_number = 0
        self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

        self.camera_to_view_id = camera_to_view_id
        self.front_camera_key = front_camera_key

    def __str__(self):
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def _init_samples(self, video_paths):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_video_path = {
                executor.submit(self._load_and_process_video_path, video_path): video_path for video_path in video_paths
            }
            for future in tqdm(as_completed(future_to_video_path), total=len(video_paths)):
                samples.extend(future.result())
        return samples

    def _load_and_process_video_path(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        n_frames = len(vr)

        samples = []
        for frame_i in range(0, n_frames, self.start_frame_interval):
            sample = dict()
            sample["video_path"] = video_path
            sample["t5_embedding_path"] = os.path.join(
                self.t5_dir,
                os.path.basename(os.path.dirname(video_path)),
                os.path.basename(video_path).replace(".mp4", ".pkl"),
            )
            sample["frame_ids"] = []
            curr_frame_i = frame_i
            while True:
                if curr_frame_i > (n_frames - 1):
                    break
                sample["frame_ids"].append(curr_frame_i)
                if len(sample["frame_ids"]) == self.sequence_length:
                    break
                curr_frame_i += self.sequence_interval
            # make sure there are sequence_length number of frames
            if len(sample["frame_ids"]) == self.sequence_length:
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path, frame_ids):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS
            fps = 24
        return frame_data, fps

    def _get_frames(self, video_path, frame_ids):
        frames, fps = self._load_video(video_path, frame_ids)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps

    def __getitem__(self, index):
        try:
            sample = self.samples[index]
            video_path = sample["video_path"]
            frame_ids = sample["frame_ids"]
            t5_embedding_path = sample["t5_embedding_path"]

            data = dict()

            videos = []
            t5_embeddings = []

            t5_masks = []
            camera_keys_selection = self.camera_keys
            view_indices_selection = [self.camera_to_view_id[camera_key] for camera_key in camera_keys_selection]
            view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.sequence_length)
            latent_view_indices_t = torch.tensor(view_indices_selection).repeat_interleave(self.state_t)

            for camera_key in camera_keys_selection:
                video, fps = self._get_frames(
                    os.path.join(
                        os.path.dirname(os.path.dirname(video_path)), camera_key, os.path.basename(video_path)
                    ),
                    frame_ids,
                )
                video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
                videos.append(video)

                if camera_key == self.front_camera_key or not self.front_view_caption_only:
                    t5_embedding_path = os.path.join(
                        os.path.dirname(os.path.dirname(t5_embedding_path)),
                        camera_key,
                        os.path.basename(t5_embedding_path),
                    )
                    with open(t5_embedding_path, "rb") as f:
                        t5_embedding = torch.from_numpy(pickle.load(f)[0])
                else:
                    t5_embedding = torch.zeros(CosmosTextEncoderConfig.NUM_TOKENS, CosmosTextEncoderConfig.EMBED_DIM)

                t5_mask = torch.ones(t5_embedding.shape[0], dtype=torch.int64)
                if t5_embedding.shape[0] < CosmosTextEncoderConfig.NUM_TOKENS:
                    t5_embedding = torch.cat(
                        [
                            t5_embedding,
                            torch.zeros(
                                CosmosTextEncoderConfig.NUM_TOKENS - t5_embedding.shape[0],
                                CosmosTextEncoderConfig.EMBED_DIM,
                            ),
                        ],
                        dim=0,
                    )
                    t5_mask = torch.cat(
                        [t5_mask, torch.zeros(CosmosTextEncoderConfig.NUM_TOKENS - t5_mask.shape[0])], dim=0
                    )
                else:
                    t5_embedding = t5_embedding[: CosmosTextEncoderConfig.NUM_TOKENS]
                    t5_mask = t5_mask[: CosmosTextEncoderConfig.NUM_TOKENS]
                t5_embeddings.append(t5_embedding)
                t5_masks.append(t5_mask)
            video = torch.cat(videos, dim=1)
            t5_embedding = torch.cat(t5_embeddings, dim=0)

            data["video"] = video
            data["video_name"] = {
                "video_path": video_path,
                "t5_embedding_path": t5_embedding_path,
                "start_frame_id": str(frame_ids[0]),
            }
            data["t5_text_embeddings"] = t5_embedding
            data["t5_text_mask"] = torch.cat(t5_masks)
            data["fps"] = fps
            data["image_size"] = torch.tensor([self.H, self.W, self.H, self.W])
            data["num_frames"] = self.sequence_length
            data["sample_n_views"] = len(camera_keys_selection)
            data["padding_mask"] = torch.zeros(1, self.H, self.W)
            data["view_indices"] = view_indices_t.contiguous()
            data["latent_view_indices_B_T"] = latent_view_indices_t.contiguous()
            data["ref_cam_view_idx_sample_position"] = torch.ones(1, dtype=torch.int64) * (-1)
            return data
        except Exception:
            warnings.warn(  # noqa: B028
                f"Invalid data encountered: {self.samples[index]['video_path']}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")  # noqa: B028
            warnings.warn(traceback.format_exc())  # noqa: B028
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]


if __name__ == "__main__":
    dataset = MultiviewDataset(
        dataset_dir="datasets/waymo/",
        sequence_interval=1,
        num_frames=29,
        camera_keys=[
            "pinhole_front_left",
            "pinhole_front",
            "pinhole_front_right",
            "pinhole_side_left",
            "pinhole_side_right",
        ],
        video_size=[480, 848],
        front_camera_key="pinhole_front",
        front_view_caption_only=True,
        camera_to_view_id={
            "pinhole_front": 0,
            "pinhole_front_left": 5,
            "pinhole_front_right": 1,
            "pinhole_side_left": 4,
            "pinhole_side_right": 2,
        },
        state_t=8,
        is_train=False,
    )

    indices = [0, -1]
    for idx in indices:
        data = dataset[idx]
        print(
            f"{idx=} "
            f"{data['video'].sum()=}\n"
            f"{data['video'].shape=}\n"
            f"{data['video_name']=}\n"
            f"{data['t5_text_embeddings'].shape=}\n"
            f"{data['latent_view_indices_B_T'].shape=}\n"
            "---"
        )
