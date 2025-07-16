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

#
# This file holds neighborhood attention [1] parameters for the 2B and 14B Video2World
# models. These parameters were selected from a small pool of varying attention patterns,
# some simply local (strided) sliding window attention [1,4], and some sparse-global dilated sliding
# window attention [2]. All patterns in the pool were derived using NATTENSim [4] and confirmed to
# be highly or fully block-sparse.
# The selection process was a simple profiling with layer granularity. Attention head granularity
# (similar to Hydra-NA [6]) is left as future work.
#
# These variants can bring 1.7X to 2.6X speedup to 720p Video2World inference depending on
# frame-rate, model size, and GPU architecture. The baselines are Flash Attention 3 [5] and cuDNN
# Attention, which are the state of the art for the Hopper and Blackwell architectures respectively.
# NATTEN's Hopper and Blackwell FNA [3] kernels are built on top of CUTLASS's FMHA kernels, which
# achieve similar performance to the baselines.
#
# For more information refer to Generalized Neighborhood Attention [4], and the NATTEN project (natten.org).
#
#  [1] Neighborhood Attention Transformer: https://arxiv.org/abs/2204.07143
#  [2] Dilated Neighborhood Attention Transformer: https://arxiv.org/abs/2209.15001
#  [3] Faster Neighborhood Attention: https://arxiv.org/abs/2403.04690
#  [4] Generalized Neighborhood Attention: https://arxiv.org/abs/2504.16922
#  [5] Flash Attention 3: https://arxiv.org/abs/2407.08608
#  [6] Efficient Image Generation with Variadic Attention Heads: https://arxiv.org/abs/2211.05770
#

# 2B Configuration:
# Layers 0, 1: 98% sparsity, sparse global (max dilation) along H and W
# Layers 2, 5, 10: 95% sparsity, sparse global (max dilation) along W
# Layers 3, 4, 6, 7, 8, 9, 27: 92% sparsity, local
# Layers 11-22, 24-26: 77% sparsity, local
# Layer 23: 55% sparsity
# No global self attention layers.
#
# Expected 1.9 – 2.6X End-to-End speedup depending on FPS, device arch.
#
PREDICT2_VIDEO2WORLD_NET_2B_NATTEN_PARAMETERS = [
    {"window_size": (-1, 4, 16), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # layer 0
    {"window_size": (-1, 4, 16), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # layer 1
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # layer 2
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 3
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 4
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # layer 5
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 6
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 7
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 8
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 9
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # layer 10
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 11
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 12
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 13
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 14
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 15
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 16
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 17
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 18
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 19
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 20
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 21
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 22
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 23
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 24
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 25
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 26
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 27
]



# 14B Configuration:
# Layers 0-10: 98% sparsity, sparse global (max dilation) along H and W
# Layers 11, 12: 95% sparsity, sparse global (max dilation) along W
# Layers 13-22: 92% sparsity, local
# Layers 23, 25, 29, 30, 33, 35: 77% sparsity, local
# Layers 24, 26, 28, 31, 32, 34: 55% sparsity
# Layer 27: 0% sparsity (self attn)
# 1 global self attention layer.
#
# Expected 1.7 – 2.1X End-to-End speedup depending on FPS, device arch.
#
PREDICT2_VIDEO2WORLD_NET_14B_NATTEN_PARAMETERS = [
    {"window_size": (-1, 4, 16), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # layer 0
    {"window_size": (-1, 4, 16), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # layer 1
    {"window_size": (-1, 4, 16), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # layer 2
    {"window_size": (-1, 4, 16), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # layer 3
    {"window_size": (-1, 4, 16), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # layer 4
    {"window_size": (-1, 4, 16), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # layer 5
    {"window_size": (-1, 4, 16), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # layer 6
    {"window_size": (-1, 4, 16), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # layer 7
    {"window_size": (-1, 4, 16), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # layer 8
    {"window_size": (-1, 4, 16), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # layer 9
    {"window_size": (-1, 4, 16), "dilation": (1, 11, 5), "base_size": (-1, 44, 80)},  # layer 10
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # layer 11
    {"window_size": (-1, 12, 16), "stride": (1, 4, 1), "dilation": (1, 1, 5), "base_size": (-1, 44, 80)},  # layer 12
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 13
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 14
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 15
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 16
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 17
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 18
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 19
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 20
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 21
    {"window_size": (-1, 12, 24), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 22
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 23
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 24
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 25
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 26
    None,  # layer 27
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 28
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 29
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 30
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 31
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 32
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 33
    {"window_size": (-1, 28, 56), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 34
    {"window_size": (-1, 20, 40), "stride": (1, 4, 8), "base_size": (-1, 44, 80)},  # layer 35
]
