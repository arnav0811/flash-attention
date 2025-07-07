import math
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_v2_kernel(
    Q, K, V, O,
    seq_len, scale,
    BLOCK_SIZE_M: tl.constexpr,   
    BLOCK_SIZE_N: tl.constexpr,  
    HEAD_DIM: tl.constexpr,   
):
    # Outer loop over rows - paRllelization
    # which sequence block this CTA owns
    pid_m = tl.program_id(0) 
    # batch * head index
    pid_batch_head = tl.program_id(1)

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    offsets_d = tl.arange(0, HEAD_DIM)

    q_ptr = Q + pid_batch_head * seq_len * HEAD_DIM + offsets_m[:, None] * HEAD_DIM + offsets_d[None, :]
    k_base = K + pid_batch_head * seq_len * HEAD_DIM
    v_base = V + pid_batch_head * seq_len * HEAD_DIM
    o_ptr = O + pid_batch_head * seq_len * HEAD_DIM + offsets_m[:, None] * HEAD_DIM + offsets_d[None, :]

    # load Q once into SRAM
    Q_block = tl.load(q_ptr, mask=offsets_m[:, None] < seq_len, other=0.0)

    accumulator = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.full([BLOCK_SIZE_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        offs_n_curr = start_n + offsets_n
        k_ptr = k_base + offs_n_curr[:, None] * HEAD_DIM + offsets_d[None, :]
        v_ptr = v_base + offs_n_curr[:, None] * HEAD_DIM + offsets_d[None, :]

        K_block = tl.load(k_ptr, mask = offs_n_curr[:, None] < seq_len, other = 0.0)
        V_block = tl.load(v_ptr, mask = offs_n_curr[:, None] < seq_len, other = 0.0)

        S = tl.dot(Q_block, tl.trans(K_block)) * scale

        # online softmax
        m_ij = tl.max(S, 1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        l_i *= alpha
        accumulator *= alpha[:, None]

        exp_S = tl.exp(S - m_new[:, None])
        l_i += tl.sum(exp_S, 1)
        accumulator += tl.dot(exp_S.to(tl.float16), V_block, out_dtype = tl.float32)
        m_i = m_new

    accumulator /= l_i[:, None]
    tl.store(o_ptr, accumulator.to(Q.dtype.element_ty),
             mask = offsets_m[:, None] < seq_len)

def flash_attention_v2(Q, K, V, BLOCK_SIZE_M: int = 128, BLOCK_SIZE_N: int = 64):
    assert Q.dtype == torch.float16, "kernel expects fp16 inputs"
    batch_size, heads, seq_len, dim = Q.shape
    scale = 1.0 / math.sqrt(dim)

    # flatten batch & head
    Q_flat = Q.reshape(-1, seq_len, dim).contiguous()
    K_flat = K.reshape(-1, seq_len, dim).contiguous()
    V_flat = V.reshape(-1, seq_len, dim).contiguous()
    O_flat = torch.empty_like(Q_flat)

    grid = (triton.cdiv(seq_len, BLOCK_SIZE_M), Q_flat.shape[0])

    flash_attention_v2_kernel[grid](
        Q_flat, K_flat, V_flat, O_flat,
        seq_len = seq_len, scale = scale,
        BLOCK_SIZE_M = BLOCK_SIZE_M,
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        HEAD_DIM=dim,
        num_warps=4,         
        num_stages=2,
    )

    return O_flat.view(batch_size, heads, seq_len, dim)
