import torch
import triton
import triton.language as tl
import math

@triton.jit
def flash_attention_kernel(
  Q, K, V, O, 
  seq_len,
  scaling_factor,
  BLOCK_SIZE_M: tl.constexpr, # Query Block Size
  BLOCK_SIZE_N: tl.constexpr, #K/V Block Size
  HEAD_DIM: tl.constexpr
):
  # Q, K, V, O are pointers
  # Index of program
  program_id_m = tl.program_id(0) #1-D launch grid, axis = 0
  # Program will process inputs that are offset from the initital data. 
  #[O:Block Size, Block Size : 2xBlock Size,...]
  start_m = program_id_m * BLOCK_SIZE_M

  # Offsets for Query Matrix
  # rows
  offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
  # columns
  offsets_k = tl.arange(0, HEAD_DIM)

  # [Block_size_m, head_dim]
  Q_block = tl.load(Q + offsets_m[:, None] * HEAD_DIM + offsets_k[None, :], mask = offsets_m[:, None] < seq_len, other=0.0)
  Q_block = Q_block.to(tl.float32)
  # O = PV
  accumulator = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype = tl.float32)
  
  # log sum exp trick
  # maximum current raw attention score for each query
  max_i = tl.full([BLOCK_SIZE_M], -float('inf'), dtype=tl.float32) 
  # sum of exp 
  exp_sum_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)

  for start_n in range(0, seq_len, BLOCK_SIZE_N):
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    K_block = tl.load(K + offsets_n[:, None] * HEAD_DIM + offsets_k[None, :], mask = offsets_n[:, None] < seq_len, other=0.0)
    V_block = tl.load(V + offsets_n[:, None] * HEAD_DIM + offsets_k[None, :], mask = offsets_n[:, None] < seq_len, other=0.0)

    K_block = K_block.to(tl.float32)
    V_block = V_block.to(tl.float32)
    
    # S = QK^T
    S = tl.dot(Q_block, tl.trans(K_block))
    S = S * scaling_factor

    # Prevent underflow/overflow Softmax
    max_ij = tl.max(S, axis = 1)
    max_new = tl.maximum(max_i, max_ij)

    alpha = tl.exp(max_i - max_new)
    accumulator = accumulator * alpha[:, None]
    exp_sum_i = exp_sum_i * alpha
    
    # f(x)'s in softmax eq
    S_new = S - max_new[:, None]

    exp_S = tl.exp(S_new)
    l_ij = tl.sum(exp_S, axis = 1)

    # Update accumulator
    accumulator += tl.dot(exp_S, V_block)
    exp_sum_i += l_ij
    max_i = max_new

  # Normalization
  accumulator = accumulator / exp_sum_i[:, None]
  # Output Block woith Memory addr calc
  tl.store(O + offsets_m[:, None] * HEAD_DIM + offsets_k[None, :], accumulator, mask = offsets_m[:, None] < seq_len)


def flash_attention(Q, K, V, BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64):
  batch_size, heads, seq_len, dim = Q.shape
  head_dim = dim
  Q = Q.view(-1, seq_len, dim).contiguous()
  K = K.view(-1, seq_len, dim).contiguous()
  V = V.view(-1, seq_len, dim).contiguous()
  O = torch.empty_like(Q)

  scaling_factor = 1.0 / math.sqrt(head_dim)

  # [(N // 64, )] - grid dim, n // 64 tells number of programs
  grid = (triton.cdiv(seq_len, BLOCK_SIZE_M) * batch_size * heads,)
  flash_attention_kernel[grid](
    Q, K, V, O,
    seq_len = seq_len,
    scaling_factor = scaling_factor,
    BLOCK_SIZE_M = BLOCK_SIZE_M,  
    BLOCK_SIZE_N = BLOCK_SIZE_N,
    HEAD_DIM = head_dim
  )

  return O.view(batch_size, heads, seq_len, dim)
  
  
