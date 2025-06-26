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

import re
from typing import List, Tuple

import torch
import torchvision.transforms.functional as F


class Resize_Preprocess:
    def __init__(self, size):
        """
        Initialize the preprocessing class with the target size.
        Args:
        size (tuple): The target height and width as a tuple (height, width).
        """
        self.size = size

    def __call__(self, video_frames):
        """
        Apply the transformation to each frame in the video.
        Args:
        video_frames (torch.Tensor): A tensor representing a batch of video frames.
        Returns:
        torch.Tensor: The transformed video frames.
        """
        # Resize each frame in the video
        resized_frames = torch.stack([F.resize(frame, self.size, antialias=True) for frame in video_frames])
        return resized_frames


class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


IMAGE_RES_SIZE_INFO: dict[str, tuple[int, int]] = {
    "1080": {
        "1,1": (1024, 1024),
        "4,3": (1440, 1056),
        "3,4": (1056, 1440),
        "16,9": (1920, 1056),
        "9,16": (1056, 1920),
    },
    # "1024": {"1,1": (1024, 1024), "4,3": (1280, 1024), "3,4": (1024, 1280), "16,9": (1280, 768), "9,16": (768, 1280)},
    "1024": {"1,1": (1024, 1024), "4,3": (1168, 880), "3,4": (880, 1168), "16,9": (1360, 768), "9,16": (768, 1360)},
    "720": {"1,1": (960, 960), "4,3": (960, 704), "3,4": (704, 960), "16,9": (1280, 704), "9,16": (704, 1280)},
    "512": {"1,1": (512, 512), "4,3": (640, 512), "3,4": (512, 640), "16,9": (640, 384), "9,16": (384, 640)},
    "480": {"1,1": (480, 480), "4,3": (640, 480), "3,4": (480, 640), "16,9": (768, 432), "9,16": (432, 768)},
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
}


VIDEO_RES_SIZE_INFO: dict[str, tuple[int, int]] = {
    "1080": {
        "1,1": (1024, 1024),
        "4,3": (1440, 1056),
        "3,4": (1056, 1440),
        "16,9": (1920, 1056),
        "9,16": (1056, 1920),
    },
    "1024": {"1,1": (1024, 1024), "4,3": (1280, 1024), "3,4": (1024, 1280), "16,9": (1280, 768), "9,16": (768, 1280)},
    "720": {"1,1": (960, 960), "4,3": (960, 704), "3,4": (704, 960), "16,9": (1280, 704), "9,16": (704, 1280)},
    "512": {"1,1": (512, 512), "4,3": (640, 512), "3,4": (512, 640), "16,9": (640, 384), "9,16": (384, 640)},
    "480": {"1,1": (480, 480), "4,3": (640, 480), "3,4": (480, 640), "16,9": (768, 432), "9,16": (432, 768)},
    "720p": {"1,1": (960, 960), "4,3": (960, 720), "3,4": (720, 960), "16,9": (1280, 720), "9,16": (720, 1280)},
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
}


def get_aspect_ratios_from_wdinfos(wdinfos: list[str]) -> list[str]:
    aspect_ratios = []
    for wdinfo in wdinfos:
        aspect_ratio_match = re.search(r"aspect_ratio_(\d+_\d+)", wdinfo)
        aspect_ratios.append(aspect_ratio_match.group(1))

    return aspect_ratios


def get_wdinfos_w_aspect_ratio(wdinfos: list[str]) -> List[Tuple[str, str]]:
    aspect_ratios = get_aspect_ratios_from_wdinfos(wdinfos)

    # return a list of (wdinfo_path, aspect_ratio) pairs
    return [(wdinfo, aspect_ratio.replace("_", ",")) for wdinfo, aspect_ratio in zip(wdinfos, aspect_ratios)]
