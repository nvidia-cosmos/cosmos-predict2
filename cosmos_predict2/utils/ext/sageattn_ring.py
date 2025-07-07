from typing import Optional, List
import threading

import torch
import torch.distributed as dist
import torch.nn.functional as F
from .sageattn_fp8 import sageattn_fp8_quantize_all, sageattn_fp8_from_quantized
from .sageattn_quant import FromPoolAllocator

_sageattn_context = threading.local()
_sageattn_context.sm_version = {}
_sageattn_context.cp_procgrp = {}
_sageattn_context.cp_scomm = torch.cuda.Stream(priority=-1)

def set_cp_procgrp(cp_group_list: List[int], procgrp: dist.ProcessGroup) -> None:
    _sageattn_context.cp_procgrp[tuple(cp_group_list)] = procgrp

def _get_cp_procgrp(cp_group_list: List[int]) -> dist.ProcessGroup:
    cp_group = tuple(cp_group_list)
    if cp_group not in _sageattn_context.cp_procgrp:
        raise RuntimeError(f"CP group {cp_group} not found. Forbidding creation for safety. Only comment out this error when you know what you are doing.")
        _sageattn_context.cp_procgrp[cp_group] = dist.new_group(ranks=cp_group)
    return _sageattn_context.cp_procgrp[cp_group]

def _get_sm_version(rank: int) -> int:
    if rank not in _sageattn_context.sm_version:
        gpu_props = torch.cuda.get_device_properties(rank)
        sm_version = gpu_props.major * 10 + gpu_props.minor
        _sageattn_context.sm_version[rank] = sm_version
    return _sageattn_context.sm_version[rank]


# Util function to perform weighted sum according to LSE
# LSE tensors will have shape (1, H, S)
# Outputs will always be in float32
@torch.jit.script
def _reaccum_o_according_to_lse(out1, out1_lse, out2, out2_lse):
    log_sumall_exp = out2_lse + F.softplus(out1_lse - out2_lse)
    out1_qse = torch.exp(out1_lse - log_sumall_exp)
    out2_qse = torch.exp(out2_lse - log_sumall_exp)
    out1 = out1.to(torch.float32) * out1_qse.transpose(1, 2).unsqueeze(-1)
    out2 = out2.to(torch.float32) * out2_qse.transpose(1, 2).unsqueeze(-1)

    return out1 + out2, log_sumall_exp


# (B S H D), (B S H D), (B S H D) -> (B S (H D))
def ring_sage_attention_exec(
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    ext_stream : torch.cuda.Stream = torch.cuda.current_stream(),
    cp_rank: int = 0,
    cp_size: int = 1,
    cp_group: Optional[List[int]] = None,
):
    # Size check
    assert k_t.shape[0] == 1, "Batching unsupported yet."

    # Obtain PyTorch metadata
    sm_version = _get_sm_version(k_t.device.index)
    scomm = _sageattn_context.cp_scomm
    cp_active = cp_size > 1 and cp_group is not None
    procgrp = _get_cp_procgrp(cp_group) if cp_active else None

    # Determine the neighbouring nodes to send / receive tensors
    recv_from = (cp_rank + 1) % cp_size
    send_to = (cp_rank - 1 + cp_size) % cp_size

    # Estimate memory usage
    kv_size_byte = k_t.numel() // k_t.shape[1] * k_t.dtype.itemsize # km
    kv_size_byte += k_t.numel() * torch.int8.itemsize * 2 # k_int8 + (rough)k_scale
    kv_size_byte += v_t.numel() * torch.float8_e4m3fn.itemsize * 2 # v_fp8 + (rough)v_scale
    alloc_kv = FromPoolAllocator(torch.empty(kv_size_byte, dtype=torch.uint8, device=q_t.device))

    with torch.cuda.stream(ext_stream), torch.cuda.nvtx.range('Ring Attention local chunks'):
        # Compute local components
        # out will have shape (1, S, H, D)
        out1_base, params = sageattn_fp8_quantize_all(q_t, k_t, v_t, tensor_layout="NHD", alloc_kv=alloc_kv, sm_version=sm_version)

        # Kick off communications after quantizing
        scomm.wait_stream(ext_stream)

        # Execute local part
        out1, out1_lse = sageattn_fp8_from_quantized(
            q_t,
            params['q_int8'],
            params['q_scale'],
            params['km'],
            params['k_int8'],
            params['k_scale'],
            params['v_fp8'],
            params['v_scale'],
            out1_base,
            q_t.size(-1),
            tensor_layout="NHD",
            sm_version=sm_version,
        )

    if cp_size > 1 and procgrp is not None:
        with torch.cuda.stream(scomm), torch.cuda.nvtx.range('Prepare KV block exchange'):
            # Params ordered by allocation
            keys_to_exchange = [ 'km', 'k_int8', 'k_scale', 'v_fp8', 'v_scale', ]
            params_byte = { k: params[k].flatten().view(torch.uint8) for k in keys_to_exchange }

            # Contiguous memory to tranceive
            buffer_t = alloc_kv.allocated_buffer(alignment=4096)

            # Prepare the receiving buffer
            # Receiving buffer will have its first dimension as (world_size-1).
            buffer_pool_t = torch.empty((cp_size, *buffer_t.shape), dtype=torch.uint8, device=q_t.device)

        for ipeer in range(cp_size-1):
            with torch.cuda.stream(scomm), torch.cuda.nvtx.range('Exchange peer KV blocks'):
                # Send to / receive from peer procs
                if cp_rank % 2 == 0:
                    _ = dist.isend(buffer_t, send_to, group=procgrp)
                    req_r = dist.irecv(buffer_pool_t[ipeer], recv_from, group=procgrp)
                else:
                    req_r = dist.irecv(buffer_pool_t[ipeer], recv_from, group=procgrp)
                    _ = dist.isend(buffer_t, send_to, group=procgrp)
                req_r.wait()

            with torch.cuda.stream(ext_stream), torch.cuda.nvtx.range('Ring Attention remote chunks'):
                # Wait for peer KV chunks
                ext_stream.wait_stream(scomm)

                # Received buffer will be sent in the next iteration
                buffer_t = buffer_pool_t[ipeer]

                # Update params with the buffers just received
                bit_pos = 0
                for k in keys_to_exchange:
                    param_dtype = params[k].dtype
                    param_shape = params[k].shape
                    bit_pos_next = bit_pos + params_byte[k].numel()
                    params[k] = buffer_t[bit_pos:bit_pos_next].view(param_dtype).view(param_shape)
                    bit_pos = bit_pos_next

                # Compute contributions from peer KV chunks
                # out2 will have shape (1, S, H, D)
                out2_base = torch.empty_like(out1_base)
                out2, out2_lse = sageattn_fp8_from_quantized(
                    q_t,
                    params['q_int8'],
                    params['q_scale'],
                    params['km'],
                    params['k_int8'],
                    params['k_scale'],
                    params['v_fp8'],
                    params['v_scale'],
                    out2_base,
                    q_t.size(-1),
                    tensor_layout="NHD",
                    sm_version=sm_version,
                )

                # Scale & accumulate contributions from local & peer KV chunks
                out1, out1_lse = _reaccum_o_according_to_lse(out1, out1_lse, out2, out2_lse)

    with torch.cuda.stream(ext_stream):
        return out1.to(q_t.dtype).flatten(-2)
