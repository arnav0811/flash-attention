import torch
import triton
import triton.language as tl
import math

@triton.jit
def flash_attention_kernel(
    Q, K, V, O, 
    seq_len,
    scaling_factor,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    HEAD_DIM: tl.constexpr
):
    # Calculate which sequence block and which batch*head
    batch_head_id = tl.program_id(0)
    seq_block_id  = tl.program_id(1)
    
    start_m = seq_block_id * BLOCK_SIZE_M
    
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, HEAD_DIM)
    
    Q_base = Q + batch_head_id * seq_len * HEAD_DIM
    K_base = K + batch_head_id * seq_len * HEAD_DIM  
    V_base = V + batch_head_id * seq_len * HEAD_DIM
    O_base = O + batch_head_id * seq_len * HEAD_DIM

    Q_block = tl.load(Q_base + offsets_m[:, None] * HEAD_DIM + offsets_k[None, :], 
                      mask=offsets_m[:, None] < seq_len, other=0.0)
    
    accumulator = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype=tl.float32)
    max_i = tl.full([BLOCK_SIZE_M], -float('inf'), dtype=tl.float32) 
    exp_sum_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Loop over K,V blocks
    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        
        K_block = tl.load(K_base + offsets_n[:, None] * HEAD_DIM + offsets_k[None, :], mask = offsets_n[:, None] < seq_len, other = 0.0)
        V_block = tl.load(V_base + offsets_n[:, None] * HEAD_DIM + offsets_k[None, :], mask = offsets_n[:, None] < seq_len, other=0.0)
        
        S = tl.dot(Q_block, tl.trans(K_block))
        S = S * scaling_factor
        
        # Online softmax update
        # log sum exp trick

        # Prevent underflow/overflow Softmax
        max_ij = tl.max(S, axis=1)
        max_new = tl.maximum(max_i, max_ij)
        alpha = tl.exp(max_i - max_new)
        accumulator = accumulator * alpha[:, None]
        exp_sum_i = exp_sum_i * alpha
        
        # f(x)'s in softmax eq
        S_new = S - max_new[:, None]
        exp_S = tl.exp(S_new)
        l_ij = tl.sum(exp_S, axis=1)

        accumulator += tl.dot(exp_S.to(tl.float16), V_block, out_dtype=tl.float32)
        exp_sum_i += l_ij
        max_i = max_new
    
    # Normalization
    accumulator = accumulator / exp_sum_i[:, None]
    
    # Output Block woith Memory addr calc
    tl.store(O_base + offsets_m[:, None] * HEAD_DIM + offsets_k[None, :], accumulator.to(Q.dtype.element_ty), mask=offsets_m[:, None] < seq_len)

def flash_attention(Q, K, V, BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 32):
    batch_size, heads, seq_len, dim = Q.shape

    Q = Q.view(-1, seq_len, dim).contiguous()
    K = K.view(-1, seq_len, dim).contiguous()
    V = V.view(-1, seq_len, dim).contiguous()
    O = torch.empty_like(Q)
    
    scaling_factor = 1.0 / math.sqrt(dim)
    
    # Grid (batch*heads, seq_blocks)
    total_seq_blocks = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid = (batch_size * heads, total_seq_blocks)
    
    flash_attention_kernel[grid](
        Q, K, V, O,
        seq_len = seq_len,
        scaling_factor = scaling_factor,
        BLOCK_SIZE_M = BLOCK_SIZE_M,  
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        HEAD_DIM = dim
    )
    
    return O.view(batch_size, heads, seq_len, dim)
