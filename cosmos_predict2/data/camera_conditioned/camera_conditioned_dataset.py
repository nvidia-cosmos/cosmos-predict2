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

from abc import ABC, abstractmethod
import os
import torch
import torchvision
import torchvision.transforms as transforms

import imageio
from einops import rearrange
from PIL import Image
import numpy as np

from cosmos_predict2.data.camera_conditioned.dataset_utils import ray_condition

class TextVideoCameraDataset(torch.utils.data.Dataset, ABC):
    def __init__(
        self,
        input_path,
        camera_path,
        prompt,
        max_num_frames=93,
        frame_interval=1,
        num_frames=93,
        patch_spatial=16,
        height=432,
        width=768,
    ):
        self.input_path = [input_path]
        self.text = [prompt]
        self.camera_path = camera_path

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.latent_frames = num_frames // 4 + 1
        self.patch_spatial = patch_spatial
        self.height = height
        self.width = width

        self.frame_process = transforms.v2.Compose(
            [
                transforms.v2.CenterCrop(size=(height, width)),
                transforms.v2.Resize(size=(height, width), antialias=True),
                transforms.v2.ToImage(),
                transforms.v2.ToDtype(torch.float32, scale=True),
                transforms.v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return image

    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):

        reader = imageio.get_reader(file_path)
        if (
            reader.count_frames() < max_num_frames
            or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval
        ):
            reader.close()
            return None

        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        return frames

    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(
            file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process
        )
        return frames
    
    @abstractmethod
    def load_trajectories(self):
        raise NotImplementedError("load_trajectories is not implemented")

    def __getitem__(self, data_id):

        text = self.text[data_id]
        path = self.input_path[data_id]
        video = self.load_video(path)
        if video is None:
            raise ValueError(f"{path} is not a valid video.")

        data = {"text": text, "video": video, "path": path}

        data["camera"] = self.load_trajectories()

        # cond, out1, out2 = torch.chunk(self.load_trajectories(), 3, dim=1)
        # data["camera"] = torch.cat((out1, cond, out2), dim=1)

        return [data]

    def __len__(self):
        return len(self.input_path)

class AGIBotDataset(TextVideoCameraDataset):
    def __init__(self, video_prefix, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_prefix = video_prefix
        self.trajectories = ["camera_tgt_0", "camera_tgt_1"]
        self.intrinsic_data_lists = ["intrinsic_head", "intrinsic_hand_0", "intrinsic_hand_1"]

    def load_trajectories(self):
        
        extrinsics_list = []
        for cam_type in self.trajectories:
            extrinsics_tgt = torch.tensor(
                np.loadtxt(
                    os.path.join(self.camera_path, f"{self.video_prefix}_{cam_type}.txt")
                )
            ).to(torch.bfloat16)
            extrinsics_tgt = torch.cat(
                (
                    extrinsics_tgt,
                    torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.bfloat16)
                    .unsqueeze(0)
                    .expand(self.latent_frames, -1),
                ),
                dim=1,
            ).reshape(-1, 4, 4)
            extrinsics_list.append(extrinsics_tgt)
        extrinsics = torch.cat(extrinsics_list, dim=0)
        # assert input video has static cameras (head-view)
        extrinsics = torch.cat(
            (torch.eye(4).unsqueeze(0).expand(self.latent_frames, -1, -1).to(extrinsics), extrinsics), dim=0
        )
        intrinsics_list = []
        for intrinsic_type in self.intrinsic_data_lists:
            intrinsics_tgt = torch.tensor(
                np.loadtxt(os.path.join(self.camera_path, f"{self.video_prefix}_{intrinsic_type}.txt"))
            ).to(torch.bfloat16)
            intrinsics_list.append(intrinsics_tgt)
        intrinsics = torch.cat(intrinsics_list, dim=0)
        plucker_rays = ray_condition(
            intrinsics.unsqueeze(0), extrinsics.unsqueeze(0), self.height, self.width, extrinsics.device
        )[0]
        return rearrange(
            plucker_rays,
            "T (H p1) (W p2) C -> T H W (p1 p2 C)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
        )

class CameraTrajectoryDataset(TextVideoCameraDataset):
    def __init__(self, trajectories, focal, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectories = trajectories
        self.focal = focal

    def load_trajectories(self):
        extrinsics_list = []
        for cam_type in self.trajectories:
            extrinsics_tgt = torch.tensor(
                np.loadtxt(os.path.join(self.camera_path, cam_type + ".txt"))
            ).to(torch.bfloat16)
            extrinsics_tgt = torch.cat(
                (
                    extrinsics_tgt,
                    torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.bfloat16)
                    .unsqueeze(0)
                    .expand(self.latent_frames, -1),
                ),
                dim=1,
            ).reshape(-1, 4, 4)
            extrinsics_list.append(extrinsics_tgt)
        extrinsics = torch.cat(extrinsics_list, dim=0)
        # assert input video has static cameras
        extrinsics = torch.cat(
            (torch.eye(4).unsqueeze(0).expand(self.latent_frames, -1, -1).to(extrinsics), extrinsics), dim=0
        )

        intrinsics = torch.tensor(np.loadtxt(os.path.join(self.camera_path, f"intrinsics_focal{self.focal}.txt"))).to(
            torch.bfloat16
        )
        intrinsics = intrinsics.unsqueeze(0).expand(extrinsics.shape[0], -1)
        plucker_rays = ray_condition(
            intrinsics.unsqueeze(0), extrinsics.unsqueeze(0), self.height, self.width, extrinsics.device
        )[0]
        return rearrange(
            plucker_rays,
            "T (H p1) (W p2) C -> T H W (p1 p2 C)",
            p1=self.patch_spatial,
            p2=self.patch_spatial,
        )