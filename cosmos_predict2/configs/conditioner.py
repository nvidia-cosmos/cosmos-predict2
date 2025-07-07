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

from typing import Optional

from hydra.core.config_store import ConfigStore

from cosmos_predict2.conditioner import BooleanFlag, ReMapkey, TextAttr, VideoConditioner
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict

_SHARED_CONFIG = dict(
    fps=L(ReMapkey)(
        input_key="fps",
        output_key="fps",
        dropout_rate=0.0,
        dtype=None,
    ),
    padding_mask=L(ReMapkey)(
        input_key="padding_mask",
        output_key="padding_mask",
        dropout_rate=0.0,
        dtype=None,
    ),
    text=L(TextAttr)(
        input_key=["t5_text_embeddings"],
        dropout_rate=0.2,
    ),
    use_video_condition=L(BooleanFlag)(
        input_key="fps",
        output_key="use_video_condition",
        dropout_rate=0.2,
    ),
)

VideoPredictionConditioner: LazyDict = L(VideoConditioner)(
    **_SHARED_CONFIG,
)


def register_conditioner():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="video_prediction_conditioner",
        node=VideoPredictionConditioner,
    )
