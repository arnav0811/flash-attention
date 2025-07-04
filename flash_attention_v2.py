import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def flash_attention_v2_kernel(
    Q, K, V, O,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    seq_len, 
    scaling_factor,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Outer loop over rows - paRllelization
    program_id_batch_head = tl.program_id(0)
    program_id_m = tl.program_id(1)
    start_m = program_id_m * BLOCK_SIZE_M

    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, HEAD_DIM)
    # Mask for sequecnes not dividible by BLOCK_SIZE_M - ut of bounds 
    q_mask = offsets_m < seq_len

    Q_ptr = Q + program_id_batch_head * stride_qb + offsets_m[:, None] * stride_qn + offsets_k[None, :]
    Q_block = tl.load(Q_ptr, mask = q_mask[:, None], other = 0.0)
    Q_block = Q_block.to(tl.float32)

    accumulator = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype = tl.float32)
    max_i = tl.full([BLOCK_SIZE_M], -float('inf'), dtype = tl.float32)
    exp_sum_i = tl.zeros([BLOCK_SIZE_M], dtype = tl.float32)

    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        kv_mask = offsets_n < seq_len

        K_ptr = K + program_id_batch_head * stride_kb + offsets_n[:, None] * stride_kn + offsets_k[None, :]
        K_block = tl.load(K_ptr, mask = kv_mask[:, None], other = 0.0)

        V_ptr = V + program_id_batch_head * stride_vb + offsets_n[:, None] * stride_vn + offsets_k[None, :]
        V_block = tl.load(V_ptr, mask = kv_mask[:, None], other = 0.0)

        K_block = K_block.to(tl.float32)
        V_block = V_block.to(tl.float32)

        # S = QK^T
        S = tl.dot(Q_block, tl.trans(K_block)) * scaling_factor
        # Online Softmax
        max_ij = tl.max(S, axis = 1)
        max_new = tl.maximum(max_i, max_ij)
        accumulator *= tl.exp(max_i - max_new)[:, None]
        exp_sum_i *= tl.exp(max_i - max_new)

        exp_S = tl.exp(S - max_new[:, None])
        l_ij = tl.sum(exp_S, axis = 1)

        accumulator += tl.dot(exp_S, V_block)
        exp_sum_i += l_ij
        max_i = max_new

    accumulator /= exp_sum_i[:, None]

    O_ptr = O + program_id_batch_head * stride_ob + offsets_m[:, None] * stride_on + offsets_k[None, :]
    tl.store(O_ptr, accumulator.to(O.dtype.element_ty), mask = q_mask[:, None])

def flash_attention_v2(Q, K, V):
    batch_size, heads, seq_len, dim = Q.shape
    Q_reshaped = Q.reshape(-1, seq_len, dim).contiguous()
    K_reshaped = K.reshape(-1, seq_len, dim).contiguous()
    V_reshaped = V.reshape(-1, seq_len, dim).contiguous()
    O = torch.empty_like(Q_reshaped)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 64

    scaling_factor = 1.0 / math.sqrt(dim)

    grid = (batch_size * heads, triton.cdiv(seq_len, BLOCK_SIZE_M))
    flash_attention_v2_kernel[grid](
        Q_reshaped, K_reshaped, V_reshaped, O,
        Q_reshaped.stride(0), Q_reshaped.stride(1), Q_reshaped.stride(2),
        K_reshaped.stride(0), K_reshaped.stride(1), K_reshaped.stride(2),
        V_reshaped.stride(0), V_reshaped.stride(1), V_reshaped.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        seq_len,
        scaling_factor,
        dim,
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_N = BLOCK_SIZE_N)

    return O.view(batch_size, heads, seq_len, dim)
