from typing import Optional
from functools import reduce
import torch
from sageattention.triton import quant_per_thread as per_thread_triton
from sageattention import _fused


class BaseAllocator:
    def __call__(self, shape, dtype : torch.dtype = torch.float32, device=None):
        return torch.empty(shape, dtype=dtype, device=device)

class FromPoolAllocator(BaseAllocator):
    def __init__(self, pool: torch.Tensor):
        self.pool = pool.view(torch.uint8).flatten()
        self.alloc = 0

    def __call__(self, shape, dtype : torch.dtype = torch.float32, device=None):
        device_req = torch.device(device)
        device_pool = torch.device(self.pool.device)
        assert device_req == device_pool, "Requested device is different from pool device"

        nbytes = reduce(lambda x, y: x * y, tuple(shape)) * dtype.itemsize
        alloc = self.alloc
        self.alloc = alloc + nbytes
        return self.pool[alloc:self.alloc].view(dtype).view(shape)

    @staticmethod
    def cdiv(a: int, b: int):
        return (a + b - 1) // b

    def allocated_buffer(self, alignment: int = 1):
        return self.pool[:FromPoolAllocator.cdiv(self.alloc, alignment) * alignment]

def per_thread_int8(q, k, km=None, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64, sm_scale=None, tensor_layout="HND",
                    alloc_q : Optional[BaseAllocator] = None,
                    alloc_k : Optional[BaseAllocator] = None):
    if alloc_q is None:
        alloc_q = BaseAllocator()
    if alloc_k is None:
        alloc_k = BaseAllocator()
    q_int8 = alloc_q(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = alloc_k(k.shape, dtype=torch.int8, device=k.device)

    if km is not None:
        k = k - km

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_qo, stride_h_qo, stride_seq_qo = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = k_int8.stride(0), k_int8.stride(2), k_int8.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    q_scale = alloc_q((b, h_qo, (qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8), device=q.device, dtype=torch.float32)
    k_scale = alloc_k((b, h_kv, (kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    grid = ((qo_len + BLKQ - 1) // BLKQ * (BLKQ // WARPQ) * 8, h_qo, b)
    per_thread_triton.quant_query_per_thread_int8_kernel[grid](
        q, q_int8, q_scale, qo_len,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_qo, stride_h_qo, stride_seq_qo,
        q_scale.stride(0), q_scale.stride(1),
        C=head_dim, BLK=WARPQ
    )

    grid = ((kv_len + BLKK - 1) // BLKK * (BLKK // WARPK) * 4, h_kv, b)
    per_thread_triton.quant_key_per_thread_int8_kernel[grid](
        k, k_int8, k_scale, kv_len,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_ko, stride_h_ko, stride_seq_ko,
        k_scale.stride(0), k_scale.stride(1),
        C=head_dim, BLK=WARPK
    )

    return q_int8, q_scale, k_int8, k_scale

def per_channel_fp8(
    v: torch.Tensor,
    tensor_layout: str ="HND",
    scale_max: float = 448.0,
    smooth_v: bool = True,
    alloc_out: Optional[BaseAllocator] = None,
):
    if alloc_out is None:
        alloc_out = BaseAllocator()
    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    if tensor_layout == "HND":
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)

    elif tensor_layout == "NHD":
        b, kv_len, h_kv, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = torch.empty((b, head_dim, h_kv, padded_len), dtype=v.dtype, device=v.device)
    
    _fused.transpose_pad_permute_cuda(v, v_transposed_permutted, _tensor_layout)

    v_fp8 = alloc_out(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
    v_scale = alloc_out((b, h_kv, head_dim), dtype=torch.float32, device=v.device)

    if smooth_v:
        vm = alloc_out((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        _fused.mean_scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, vm, v_scale, kv_len, scale_max, _tensor_layout)
        return v_fp8, v_scale, vm
    else:
        _fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, scale_max, _tensor_layout)
        return v_fp8, v_scale, None
