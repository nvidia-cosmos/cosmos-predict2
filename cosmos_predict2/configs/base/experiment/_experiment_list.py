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

from cosmos_predict2.configs.base.experiment.utils import Experiment

EXPERIMENTS = {}


def add_example_group():
    EXPERIMENTS["wan_example"] = Experiment(
        job_group="debug",
        job_exp="predict2_video2world_2b_training",
        nnode=2,
        command_args=[
            "trainer.max_iter=50",
        ],
    )


add_example_group()
