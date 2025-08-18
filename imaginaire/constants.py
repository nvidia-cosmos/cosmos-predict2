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

"""Cosmos Predict2 package configuration."""

import argparse
import enum
import os
import shlex
from typing import Literal

from imaginaire.utils import log


class TextEncoderClass(str, enum.Enum):
    COSMOS_REASON1 = "cosmos_reason1"
    T5 = "t5"


_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--checkpoints", default="checkpoints", help="Path to the checkpoints directory")
_parser.add_argument(
    "--text_encoder",
    default=TextEncoderClass.T5,
    type=TextEncoderClass,
    choices=list(TextEncoderClass),
    help="Text encoder to use",
)
_args = shlex.split(os.environ.get("COSMOS_PREDICT2_ARGS", ""))
_args = _parser.parse_args(_args)
log.debug(f"Cosmos Predict2 args: {_args}")


# Feature flags
TEXT_ENCODER_CLASS: TextEncoderClass = _args.text_encoder

# Checkpoints
CHECKPOINTS_DIR = _args.checkpoints

T5_MODEL_DIR = f"{CHECKPOINTS_DIR}/google-t5/t5-11b"

LLAMA_GUARD3_MODEL_DIR = f"{CHECKPOINTS_DIR}/meta-llama/Llama-Guard-3-8B"

COSMOS_GUARDRAIL1_MODEL_DIR = f"{CHECKPOINTS_DIR}/nvidia/Cosmos-Guardrail1"

COSMOS_REASON1_MODEL_DIR = f"{CHECKPOINTS_DIR}/nvidia/Cosmos-Reason1-7B"
_COSMOS_REASON1_PRIVATE_MODEL_DIR = f"{CHECKPOINTS_DIR}/nvidia/Cosmos-Reason1-Private"
COSMOS_REASON1_PRIVATE_TOKENIZER = f"{_COSMOS_REASON1_PRIVATE_MODEL_DIR}/tokenizer"
COSMOS_REASON1_PRIVATE_CHECKPOINT = f"{_COSMOS_REASON1_PRIVATE_MODEL_DIR}/reason1_internal_real.pt"


CosmosPredict2Text2ImageModelSize = Literal["0.6B", "2B", "14B"]
CosmosPredict2Text2ImageModelType = Literal["Text2Image"]


def _get_cosmos_predict2_text2image_model_dir(
    *, model_size: CosmosPredict2Text2ImageModelSize, model_type: CosmosPredict2Text2ImageModelType = "Text2Image"
) -> str:
    return f"{CHECKPOINTS_DIR}/nvidia/Cosmos-Predict2-{model_size}-{model_type}"


def get_cosmos_predict2_text2image_tokenizer(
    *,
    model_size: CosmosPredict2Text2ImageModelSize,
    model_type: CosmosPredict2Text2ImageModelType = "Text2Image",
    fast_tokenizer: bool = False,
) -> str:
    model_dir = _get_cosmos_predict2_text2image_model_dir(model_size=model_size, model_type=model_type)
    suffix = ""
    if fast_tokenizer:
        suffix += "_fast"
    return f"{model_dir}/tokenizer{suffix}/tokenizer.pth"


def get_cosmos_predict2_text2image_checkpoint(
    *,
    model_size: CosmosPredict2Text2ImageModelSize,
    model_type: CosmosPredict2Text2ImageModelType = "Text2Image",
    fast_tokenizer: bool = False,
) -> str:
    model_dir = _get_cosmos_predict2_text2image_model_dir(model_size=model_size, model_type=model_type)
    suffix = ""
    if fast_tokenizer:
        suffix += "_fast_tokenizer"
    return f"{model_dir}/model{suffix}.pt"


CosmosPredict2Video2WorldModelSize = Literal["0.6B", "2B", "14B"]
CosmosPredict2Video2WorldResolution = Literal["480", "720"]
CosmosPredict2Video2WorldFPS = Literal[10, 16]
CosmosPredict2Video2WorldAspectRatio = Literal["1:1", "4:3", "3:4", "16:9", "9:16"]
CosmosPredict2Video2WorldModelType = Literal["Text2Image", "Video2World", "Multiview"]


def _get_cosmos_predict2_video2world_model_dir(
    *,
    model_size: CosmosPredict2Video2WorldModelSize,
    model_type: CosmosPredict2Video2WorldModelType = "Video2World",
) -> str:
    return f"{CHECKPOINTS_DIR}/nvidia/Cosmos-Predict2-{model_size}-{model_type}"


def get_cosmos_predict2_video2world_tokenizer(
    *,
    model_size: CosmosPredict2Video2WorldModelSize,
    model_type: CosmosPredict2Video2WorldModelType = "Video2World",
) -> str:
    model_dir = _get_cosmos_predict2_video2world_model_dir(model_size=model_size, model_type=model_type)
    return f"{model_dir}/tokenizer/tokenizer.pth"


def get_cosmos_predict2_video2world_checkpoint(
    *,
    model_size: CosmosPredict2Video2WorldModelSize,
    model_type: CosmosPredict2Video2WorldModelType = "Video2World",
    resolution: CosmosPredict2Video2WorldResolution = "720",
    fps: CosmosPredict2Video2WorldFPS = 16,
    aspect_ratio: CosmosPredict2Video2WorldAspectRatio = "16:9",
    natten: bool = False,
) -> str:
    model_dir = _get_cosmos_predict2_video2world_model_dir(model_size=model_size, model_type=model_type)
    suffix = ""
    if natten:
        if aspect_ratio != "16:9":
            raise NotImplementedError("Cosmos-Predict2 + NATTEN only supports 16:9 aspect ratio at the moment.")
        suffix += "-natten"
    return f"{model_dir}/model-{resolution}p-{fps}fps{suffix}.pt"


CosmosPredict2MultiviewModelSize = Literal["2B"]
CosmosPredict2MultiviewResolution = Literal["720"]
CosmosPredict2MultiviewFPS = Literal[10]
CosmosPredict2MultiviewViews = Literal[7]
CosmosPredict2MultiviewFrames = Literal[29]


def get_cosmos_predict2_multiview_checkpoint(
    *,
    model_size: CosmosPredict2MultiviewModelSize,
    views: int = CosmosPredict2MultiviewViews,
    frames: int = CosmosPredict2MultiviewFrames,
    resolution: CosmosPredict2MultiviewResolution = "720",
    fps: CosmosPredict2MultiviewFPS = 16,
) -> str:
    model_dir = _get_cosmos_predict2_video2world_model_dir(model_size=model_size, model_type="Multiview")
    return f"{model_dir}/model-{resolution}p-{fps}fps-{views}views-{frames}frames.pt"


CosmosPredict2ActionConditionedModelSize = Literal["2B"]
CosmosPredict2ActionConditionedResolution = Literal["720"]
CosmosPredict2ActionConditionedFPS = Literal[16]


def get_cosmos_predict2_action_conditioned_checkpoint(
    *,
    model_size: CosmosPredict2ActionConditionedModelSize,
    resolution: CosmosPredict2ActionConditionedResolution,
    fps: CosmosPredict2ActionConditionedFPS,
) -> str:
    return get_cosmos_predict2_video2world_checkpoint(
        model_size=model_size,
        model_type="Sample-Action-Conditioned",
        resolution=resolution,
        fps=fps,
    )


CosmosPredict2Gr00tModelSize = Literal["14B"]
CosmosPredict2Gr00tResolution = Literal["480"]
CosmosPredict2Gr00tFPS = Literal[16]
CosmosPredict2Gr00tAspectRatio = CosmosPredict2Video2WorldAspectRatio
CosmosPredict2Gr00tVariant = Literal["gr1", "droid"]
CosmosPredict2Gr00tModelType = Literal["Sample-GR00T-Dreams-GR1", "Sample-GR00T-Dreams-DROID"]

_GR00T_MODEL_TYPE_MAPPING: dict[CosmosPredict2Gr00tVariant, CosmosPredict2Gr00tModelType] = {
    "gr1": "Sample-GR00T-Dreams-GR1",
    "droid": "Sample-GR00T-Dreams-DROID",
}


def get_cosmos_predict2_gr00t_checkpoint(
    *,
    gr00t_variant: CosmosPredict2Gr00tVariant,
    model_size: CosmosPredict2Gr00tModelSize,
    resolution: CosmosPredict2Video2WorldResolution,
    fps: CosmosPredict2Video2WorldFPS,
    aspect_ratio: CosmosPredict2Gr00tAspectRatio,
) -> str:
    return get_cosmos_predict2_video2world_checkpoint(
        model_size=model_size,
        model_type=_GR00T_MODEL_TYPE_MAPPING[gr00t_variant],
        resolution=resolution,
        fps=fps,
        aspect_ratio=aspect_ratio,
    )
