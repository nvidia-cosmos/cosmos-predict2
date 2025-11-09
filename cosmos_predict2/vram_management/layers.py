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

import copy

import torch

from cosmos_predict2.models.utils import init_weights_on_device


def cast_to(weight, dtype, device):
    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight)
    return r


class AutoWrappedModule(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        offload_dtype,
        offload_device,
        onload_dtype,
        onload_device,
        computation_dtype,
        computation_device,
    ):
        super().__init__()
        self.module = module.to(dtype=offload_dtype, device=offload_device)
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.module.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            module = self.module
        else:
            module = copy.deepcopy(self.module).to(dtype=self.computation_dtype, device=self.computation_device)
        return module(*args, **kwargs)


class AutoWrappedLinear(torch.nn.Linear):
    def __init__(
        self,
        module: torch.nn.Linear,
        offload_dtype,
        offload_device,
        onload_dtype,
        onload_device,
        computation_dtype,
        computation_device,
    ):
        with init_weights_on_device(device=torch.device("meta")):
            super().__init__(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                dtype=offload_dtype,
                device=offload_device,
            )
        self.weight = module.weight
        self.bias = module.bias
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.state = 0

    def offload(self):
        if self.state == 1 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self):
        if self.state == 0 and (self.offload_dtype != self.onload_dtype or self.offload_device != self.onload_device):
            self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def forward(self, x, *args, **kwargs):
        if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
            weight, bias = self.weight, self.bias
        else:
            weight = cast_to(self.weight, self.computation_dtype, self.computation_device)
            bias = None if self.bias is None else cast_to(self.bias, self.computation_dtype, self.computation_device)
        return torch.nn.functional.linear(x, weight, bias)


def enable_vram_management_recursively(
    model: torch.nn.Module,
    module_map: dict,
    module_config: dict,
    max_num_param=None,
    overflow_module_config: dict = None,  # noqa: RUF013
    total_num_param=0,
    diffsynth_opts=None
):
    for name, module in model.named_children():
        if diffsynth_opts != None and diffsynth_opts["cache"]:
            # In tokenizer.py, in forward functions of these classes, clear cache
            from cosmos_predict2.tokenizers.tokenizer import ResidualBlock, Encoder3d
            if isinstance(module, ResidualBlock) or isinstance(module, Encoder3d):
                module.cuda_empty_cache = True
        if diffsynth_opts != None and diffsynth_opts["dit_opt"]:
            # These classes have certain attributess and methods that Diffsynth's offload/onload can't handle properly
            # So, simply keep them on the GPU the entire time
            from cosmos_predict2.models.text2image_dit import Attention, PatchEmbed, VideoRopePosition3DEmb
            if isinstance(module, Attention) or isinstance(module, PatchEmbed) or isinstance(module, VideoRopePosition3DEmb):
                module.to(device='cuda')
        for source_module, target_module in module_map.items():
            if isinstance(module, source_module):
                num_param = sum(p.numel() for p in module.parameters())
                if max_num_param is not None and total_num_param + num_param > max_num_param:
                    module_config_ = overflow_module_config
                else:
                    module_config_ = module_config
                
                # This import allows looking for CausalConv3d
                # which allows setting the useTiledConv3d attribute
                # (an addition to the class). If useTiledConv3d=True
                # then patched CausalConv3d is run.
                if diffsynth_opts != None and diffsynth_opts["patchconv3d"]:
                    from cosmos_predict2.tokenizers.tokenizer import CausalConv3d
                    if isinstance(module, CausalConv3d):
                        module.useTiledConv3d = True
                module_ = target_module(module, **module_config_)
                # Some computations in the Guardrail model access the weight attribute
                # This will allow it to be accessed using the Wrapper class AutowrappedModule.weight
                # module_ is the wrapper class, module_.module is the original class
                if diffsynth_opts != None and diffsynth_opts["video_gd_opt"]:
                    if not isinstance(module_, AutoWrappedLinear) and hasattr(module_.module, 'weight'):
                        module_.weight = module_.module.weight.to('cuda')
                setattr(model, name, module_)
                getattr(model, name).to('cpu')
                total_num_param += num_param
                break
        else:
            total_num_param = enable_vram_management_recursively(
                module, module_map, module_config, max_num_param, overflow_module_config, total_num_param, diffsynth_opts=diffsynth_opts
            )
    return total_num_param


def enable_vram_management(
    model: torch.nn.Module,
    module_map: dict,
    module_config: dict,
    max_num_param=None,
    overflow_module_config: dict = None,  # noqa: RUF013
    diffsynth_opts = None,
):
    
    enable_vram_management_recursively(
        model, module_map, module_config, max_num_param, overflow_module_config, total_num_param=0, diffsynth_opts=diffsynth_opts
    )
    model.vram_management_enabled = True
