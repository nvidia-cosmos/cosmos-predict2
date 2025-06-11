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

from hydra.core.config_store import ConfigStore

from cosmos_predict2.datasets.cached_replay_dataloader import get_cached_replay_dataloader
from cosmos_predict2.datasets.data_sources.mock_data import get_image_dataset, get_video_dataset
from cosmos_predict2.datasets.joint_dataloader import IterativeJointDataLoader
from cosmos_predict2.data.dataset_video import Dataset
from imaginaire.lazy_config import LazyCall as L
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler


mock_image_dataloader = L(get_cached_replay_dataloader)(
    dataset=L(get_image_dataset)(
        resolution="512",
    ),
    batch_size=2,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    webdataset=False,
    cache_replay_name="image_dataloader",
)

mock_video_dataloader = L(get_cached_replay_dataloader)(
    dataset=L(get_video_dataset)(
        resolution="512",
        num_video_frames=136,  # number of pixel frames, the number needs to agree with tokenizer encoder since tokenizer can not handle arbitrary length
    ),
    batch_size=1,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    webdataset=False,
    cache_replay_name="video_dataloader",
)

mock_interleaved_dataloader = L(IterativeJointDataLoader)(
    dataloaders={
        "image_data": {
            "dataloader": mock_image_dataloader,
            "ratio": 1,
        },
        "video_data": {
            "dataloader": mock_video_dataloader,
            "ratio": 1,
        },
    }
)


# Cosmos-NeMo-Assets example
def get_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )

example_video_dataset_cosmos_nemo_assets = L(Dataset)(
    dataset_dir="datasets/benchmark_train/cosmos_nemo_assets",
    num_frames=77,
    video_size=(720, 1280),
)

dataloader_train_cosmos_nemo_assets = L(DataLoader)(
    dataset=example_video_dataset_cosmos_nemo_assets,
    sampler=L(get_sampler)(dataset=example_video_dataset_cosmos_nemo_assets),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)

def register_training_and_val_data():
    cs = ConfigStore()
    cs.store(group="data_train", package="dataloader_train", name="cosmos_nemo_assets", node=dataloader_train_cosmos_nemo_assets)
    cs.store(group="data_train", package="dataloader_train", name="mock", node=mock_interleaved_dataloader)
    cs.store(group="data_train", package="dataloader_train", name="mock_image", node=mock_image_dataloader)
    cs.store(group="data_train", package="dataloader_train", name="mock_video", node=mock_video_dataloader)
    cs.store(group="data_val", package="dataloader_val", name="mock", node=mock_interleaved_dataloader)

