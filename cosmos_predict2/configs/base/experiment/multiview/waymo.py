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
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_multiview import MultiviewDataset
from cosmos_predict2.conditioner import ConditionLocation
from imaginaire.lazy_config import LazyCall as L
from cosmos_predict2.callbacks.every_n_draw_sample_multiviewvideo import EveryNDrawSampleMultiviewVideo


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


cs = ConfigStore.instance()

camera_keys = ["pinhole_side_left", "pinhole_front_left", "pinhole_front", "pinhole_front_right",  "pinhole_side_right"]
camera_to_view_id = {
    "pinhole_front": 0,
    "pinhole_front_left": 5,
    "pinhole_front_right": 1,
    "pinhole_side_left": 4,
    "pinhole_side_right": 2
}
example_video_dataset_waymo_train = L(MultiviewDataset)(
    dataset_dir="datasets/waymo",
    state_t=8,
    num_frames=29,
    sequence_interval=1,
    camera_keys=camera_keys,
    video_size=(480, 848),
    front_camera_key="pinhole_front",
    camera_to_view_id=camera_to_view_id,
    front_view_caption_only=True,
    is_train=True,
)

example_video_dataset_waymo_val = L(MultiviewDataset)(
    dataset_dir="datasets/waymo",
    state_t=8,
    num_frames=29,
    sequence_interval=1,
    camera_keys=camera_keys,
    video_size=(480, 848),
    front_camera_key="pinhole_front",
    camera_to_view_id=camera_to_view_id,
    front_view_caption_only=True,
    is_train=False,
)


dataloader_train_waymo = L(DataLoader)(
    dataset=example_video_dataset_waymo_train,
    sampler=L(get_sampler)(dataset=example_video_dataset_waymo_train),
    batch_size=1,
    drop_last=True,
    num_workers=6,
    pin_memory=True,
)

dataloader_val_waymo = L(DataLoader)(
    dataset=example_video_dataset_waymo_val,
    sampler=L(get_sampler)(dataset=example_video_dataset_waymo_val),
    batch_size=1,
    drop_last=True,
    num_workers=6,
    pin_memory=True,
)

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_multiview_training_2b_waymo
predict2_multiview_training_2b_waymo_480p_10fps = dict(
    defaults=[
        {"override /model": "predict2_multiview_fsdp_2b_480p_29frames_10fps"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="multiview",
        name="2b_waymo_480p_10fps",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                state_t = 8,
                condition_locations=[ConditionLocation.FIRST_RANDOM_N],
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
                min_num_conditional_frames_per_view=0,
                max_num_conditional_frames_per_view=2,
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                net=dict(
                    rope_h_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=8.0 / 24.0,
                    rope_w_extrapolation_ratio=3.0,
                    sac_config=dict(
                        mode="none",
                    ),
                ),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=8,
    ),
    dataloader_train=dataloader_train_waymo,
    dataloader_val=dataloader_val_waymo,
    trainer=dict(
        logging_iter=10,
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
            device_monitor=dict(every_n=100),
            every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                every_n=500,
                is_x0=False,
                is_ema=False,
                guidance=[0, 2, 7],
                num_sampling_step=35,
                fps=10,
                sample_n_views=len(camera_keys),
                dataset_name=None,
            ),
            every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                every_n=500,
                is_x0=False,
                is_ema=True,
                guidance=[0, 2, 7],
                num_sampling_step=35,
                fps=10,
                sample_n_views=len(camera_keys),
                dataset_name=None,
            )
            
        ),
        max_iter=100_000,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.1,
    ),
    scheduler=dict(
        f_max=[0.25],
        f_min=[0.1],
        warm_up_steps=[2_000],
        cycle_lengths=[40_000],
    )
)

# torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_multiview_lora_training_2b_waymo_480p_10fps
predict2_multiview_lora_training_2b_waymo_480p_10fps = dict(
    defaults=[
        {"override /model": "predict2_multiview_fsdp_2b_480p_29frames_10fps"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="multiview_lora",
        name="2b_waymo_480p_10fps",
    ),
    model=dict(
        config=dict(
            train_architecture="lora",
            lora_rank=16,
            lora_alpha=16,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
            pipe_config=dict(
                state_t = 8,
                condition_locations=[ConditionLocation.FIRST_RANDOM_N],
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
                min_num_conditional_frames_per_view=0,
                max_num_conditional_frames_per_view=2,
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                net=dict(
                    rope_h_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=8.0 / 24.0,
                    rope_w_extrapolation_ratio=3.0,
                    sac_config=dict(
                        mode="none",
                    ),
                ),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=8,
    ),
    dataloader_train=dataloader_train_waymo,
    dataloader_val=dataloader_val_waymo,
    trainer=dict(
        logging_iter=10,
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),
            device_monitor=dict(every_n=100),
            every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                every_n=500,
                is_x0=False,
                is_ema=False,
                guidance=[0, 2, 7],
                num_sampling_step=35,
                fps=10,
                sample_n_views=len(camera_keys),
                dataset_name=None,
            ),
            every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                every_n=500,
                is_x0=False,
                is_ema=True,
                guidance=[0, 2, 7],
                num_sampling_step=35,
                fps=10,
                sample_n_views=len(camera_keys),
                dataset_name=None,
            )
            
        ),
        max_iter=100_000,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),
        weight_decay=0.1,
    ),
    scheduler=dict(
        f_max=[0.25],
        f_min=[0.1],
        warm_up_steps=[2_000],
        cycle_lengths=[40_000],
    )
)


for _item in [
    predict2_multiview_training_2b_waymo_480p_10fps, 
    predict2_multiview_lora_training_2b_waymo_480p_10fps,
]:
    # Get the experiment name from the global variable.
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )