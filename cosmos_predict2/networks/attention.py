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

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from flash_attn_3.flash_attn_interface import flash_attn_varlen_func

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    "attention",
]


def get_device_cc(device) -> int:
    if torch.cuda.is_available() and torch.version.cuda and device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        return major * 10 + minor

    return 0


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    deterministic=False,
    dtype=torch.bfloat16,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    compute_cap = get_device_cc(q.device)
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    assert FLASH_ATTN_3_AVAILABLE and compute_cap == 90

    # Note: dropout_p, window_size are not supported in FA3 now.
    x = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True),
        cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True),
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=lq,
        max_seqlen_k=lk,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )[0].unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    deterministic=False,
    dtype=torch.bfloat16,
):
    compute_cap = get_device_cc(q.device)

    if q_scale is not None:
        q = q * q_scale

    if compute_cap == 90 and FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            deterministic=deterministic,
            dtype=dtype,
        )
    else:
        if compute_cap in [90, 100]:
            SDPA_BACKENDS = [
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                ]
            BEST_SDPA_BACKEND = SDPBackend.CUDNN_ATTENTION
        else:
            SDPA_BACKENDS = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]
            BEST_SDPA_BACKEND = SDPBackend.FLASH_ATTENTION if compute_cap >= 80 else SDPBackend.EFFICIENT_ATTENTION

        if q_lens is not None or k_lens is not None:
            warnings.warn(
                "Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance."
            )

        if deterministic:
            raise NotImplementedError(
                "Deterministic mode in attention is only supported when Flash Attention 3 is available."
            )

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        # Torch 2.6 and later allows priorities for backends, but for older versions
        # we can only run with a specific backend. As long as we pick ones we're certain
        # will work on that device, it should be fine.
        try:
            sdpa_kernel(backends=SDPA_BACKENDS, set_priority_order=True)
            sdpa_kernel_ = partial(sdpa_kernel, set_priority_order=True)
        except TypeError:
            sdpa_kernel_ = sdpa_kernel
            SDPA_BACKENDS = [BEST_SDPA_BACKEND]

        with sdpa_kernel_(backends=SDPA_BACKENDS):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=causal,
                dropout_p=dropout_p,
                scale=softmax_scale,
            )

        out = out.transpose(1, 2).contiguous()
        return out
