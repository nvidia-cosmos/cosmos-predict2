#!/usr/bin/env -S uv run --script
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

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "torch",
# ]
# [tool.uv.sources]
# torch = [{ index = "pytorch" }]
# [[tool.uv.index]]
# name = "pytorch"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///

"""Convert .distcp checkpoint to .pt format.

Usage:

```python
./scripts/convert_distcp.py checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps
```
"""

import argparse
import os

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def convert_checkpoint(
    save_path: str,
) -> None:
    dcp_checkpoint_dir = os.path.join(save_path, "model")
    if not os.path.exists(dcp_checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {dcp_checkpoint_dir} does not exist")

    print(f"Converting checkpoints in {save_path}...")
    torch_save_path_ema_reg = os.path.join(save_path, "model_ema_reg.pt")
    torch_save_path_ema_only_fp32 = os.path.join(save_path, "model_ema_fp32.pt")
    torch_save_path_ema_only_bf16 = f"{save_path}.pt"

    # 1. Convert distributed checkpoint to torch single checkpoint
    if os.path.exists(torch_save_path_ema_reg):
        print(f"{torch_save_path_ema_reg} already exists, skipping.")
    else:
        dcp_to_torch_save(dcp_checkpoint_dir, torch_save_path_ema_reg)
        print(f"Converted {dcp_checkpoint_dir} to {torch_save_path_ema_reg}")

    # 2. Drop Reg keys and save EMA weights only in fp32 precision
    if os.path.exists(torch_save_path_ema_only_fp32):
        print(f"{torch_save_path_ema_only_fp32} already exists, skipping.")
    else:
        state_dict_ema_reg = torch.load(torch_save_path_ema_reg, map_location="cpu", weights_only=False)
        state_dict_ema_only_fp32 = dict()  # ema only
        for key in state_dict_ema_reg:
            if key.startswith("net_ema."):
                key_new = key.replace("net_ema.", "net.")
                state_dict_ema_only_fp32[key_new] = state_dict_ema_reg[key]

        torch.save(state_dict_ema_only_fp32, torch_save_path_ema_only_fp32)
        print(f"Saved EMA fp32 weights from {torch_save_path_ema_reg} to {torch_save_path_ema_only_fp32}")

    # 3. Save EMA weights only in bf16 precision
    if "state_dict_ema_only_fp32" not in locals():
        state_dict_ema_only_fp32 = torch.load(torch_save_path_ema_only_fp32, map_location="cpu", weights_only=False)

    state_dict_ema_only_bf16 = dict()  # ema only
    for key in state_dict_ema_only_fp32:
        if (
            isinstance(state_dict_ema_only_fp32[key], torch.Tensor)
            and state_dict_ema_only_fp32[key].dtype == torch.float32
        ):
            state_dict_ema_only_bf16[key] = state_dict_ema_only_fp32[key].bfloat16()
        else:
            state_dict_ema_only_bf16[key] = state_dict_ema_only_fp32[key]

    torch.save(state_dict_ema_only_bf16, torch_save_path_ema_only_bf16)
    print(f"fp32 -> bf16: {torch_save_path_ema_only_fp32} to {torch_save_path_ema_only_bf16}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "save_path",
        type=str,
        help="Local path to the checkpoint",
    )
    args = parser.parse_args()

    if os.path.exists(f"{args.save_path}.pt"):
        os.remove(f"{args.save_path}.pt")

    convert_checkpoint(args.save_path)
    print("All checkpoints converted successfully.")


if __name__ == "__main__":
    main()
