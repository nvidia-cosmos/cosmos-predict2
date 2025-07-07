from typing import Optional
import torch
import importlib
from . import sageattn_quant

_qattn = {}

def _qattn_module(sm_version: int = 89):
    if sm_version not in _qattn.keys():
        if sm_version in [90]:
            _qattn[sm_version] = importlib.import_module("sageattention._qattn_sm90")
        elif sm_version in [89, 120]:
            _qattn[sm_version] = importlib.import_module("sageattention._qattn_sm89")
        else:
            raise NotImplementedError(f"Attention kernel not implemented for SM version {sm_version}")
    return _qattn[sm_version]

@torch.compiler.disable
def sageattn_fp8_quantize_all(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    alloc_kv: Optional[sageattn_quant.BaseAllocator] = None,
    sm_version: int = 89,
) -> torch.Tensor:

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
    if alloc_kv is None:
        alloc_kv = sageattn_quant.BaseAllocator()

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2

    km_shape = list(k.shape)
    km_shape[seq_dim] = 1
    km = alloc_kv(km_shape, dtype=k.dtype, device=k.device)
    @torch.jit.script
    def _calc_km(km, k, seq_dim:int):
        km.copy_(k.mean(dim=seq_dim, keepdim=True))
    _calc_km(km, k, seq_dim)

    if sm_version in [90]:
        qk_thrshape_args = { "BLKQ": 64, "WARPQ": 16, "BLKK": 128, "WARPK": 128 }
    elif sm_version in [89, 120]:
        qk_thrshape_args = { "BLKQ": 128, "WARPQ": 32, "BLKK": 64, "WARPK": 64 }
    else:
        raise NotImplementedError(f"Configuration not implemented for SM version {sm_version}")

    if qk_quant_gran == "per_warp":
        raise NotImplementedError("Per-warp quantization with allocator control is not implemented.")
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = sageattn_quant.per_thread_int8(
            q, k, km, tensor_layout=tensor_layout, alloc_k=alloc_kv, **qk_thrshape_args
        )

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    if sm_version in [90]:
        # pad v to multiple of 128
        # TODO: modify per_channel_fp8 kernel to handle this
        kv_len = k.size(seq_dim)
        v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
        if v_pad_len > 0:
            if tensor_layout == "HND":
                v = torch.cat([v, torch.zeros(v.size(0), v.size(1), v_pad_len, v.size(3), dtype=v.dtype, device=v.device)], dim=2)
            else:
                v = torch.cat([v, torch.zeros(v.size(0), v_pad_len, v.size(2), v.size(3), dtype=v.dtype, device=v.device)], dim=1)

    v_fp8, v_scale, _ = sageattn_quant.per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False, alloc_out=alloc_kv)

    return (
        o,
        {
            'q_int8': q_int8,
            'q_scale': q_scale,
            'km': km,
            'k_int8': k_int8,
            'k_scale': k_scale,
            'v_fp8': v_fp8,
            'v_scale': v_scale,
        }
    )


@torch.compiler.disable
def sageattn_fp8_from_quantized(
    q: torch.Tensor,
    q_int8: torch.Tensor,
    q_scale: torch.Tensor,
    km: torch.Tensor,
    k_int8: torch.Tensor,
    k_scale: torch.Tensor,
    v_fp8: torch.Tensor,
    v_scale: torch.Tensor,
    o: torch.Tensor,
    head_dim_og: int,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    sm_version: int = 89,
) -> torch.Tensor:

    dtype = o.dtype
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    if tensor_layout == "NHD":
        lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)

    qattn = _qattn_module(sm_version)

    if pv_accum_dtype == "fp32":
        if sm_version in [90]:
            raise NotImplementedError("Please use pv_accum_dtype='fp32+fp32' for sm90.")
        lse = qattn.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp32":
        lse = qattn.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)

    o = o[..., :head_dim_og]

    return (
        o,
        lse / 1.44269504 + lse_correction * sm_scale,
    )