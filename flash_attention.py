import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_kernel(
  Q, K, V, O, 
  seq_len,
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
  Q_block = tl.load(Q + offsets_m[:, None] * HEAD_DIM + offsets_k[None, :])

  # O = PV
  accumulator = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype = tl.float32)
  
  # log sum exp trick
  # maximum current raw attention score for each query
  max_i = tl.full([BLOCK_SIZE_M], float('-inf'), dtype=tl.float32) 
  # sum of exp 
  exp_sum_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)

  for start_n in range(0, seq_len, BLOCK_SIZE_N):
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    K_block = tl.load(K + offsets_n[:, None] * HEAD_DIM + offsets_k[None, :])
    V_block = tl.load(V + offsets_n[:, None] * HEAD_DIM + offsets_k[None, :])

    # S = QK^T
    S = tl.dot(Q_block, tl.trans(K_block))

    # Prevent underflow/overflow Softmax
    max_ij = tl.max(S, axis = 1)
    # f(x)'s in softmax eq
    S_new = S - max_ij[:, None]

    exp_S = tl.exp(S_new)
    l_ij = tl.sum(exp_S, axis = 1)
    alpha = tl.exp(max_i - max_ij)
    accumulator *= alpha[:, None]
    accumulator += tl.dot(exp_S, V_block)

    # Update for softmax
    max_i = tl.maximum(max_i, max_ij)
    exp_sum_i += l_ij

  # Output Block woith Memory addr calc
  tl.store(O + offsets_m[:, None] * HEAD_DIM + offsets_k[None, :], accumulator)


def flash_attention(Q, K, V, BLOCK_SIZE_M: int = 64, BLOCK_SIZE_N: int = 64):
  batch_size, heads, seq_len, dim = Q.shape
  head_dim = dim
  Q = Q.view(-1, seq_len, dim).contiguous()
  K = K.view(-1, seq_len, dim).contiguous()
  V = V.view(-1, seq_len, dim).contiguous()
  O = torch.empty_like(Q)

  # [(N // 64, )] - grid dim, n // 64 tells number of programs
  grid = (Q.shape[0],)
  flash_attention_kernel[grid](
    Q, K, V, O,
    seq_len = seq_len,
    BLOCK_SIZE_M = BLOCK_SIZE_M,  
    BLOCK_SIZE_N = BLOCK_SIZE_N,
    HEAD_DIM = head_dim
  )

  return O.view(batch_size, heads, seq_len, dim)
  
  
