import math
import time
import torch
from attention import scaled_dot_product_attention
from multi_headed_attention import MultiHeadedAttention
from flash_attention import flash_attention

torch.manual_seed(0)
DEVICE = "cuda"

batch_size = 1
heads = 1
seq_len = 128
dim = 64

Q = torch.randn(batch_size, heads, seq_len, dim, device = DEVICE)
K = torch.randn(batch_size, heads, seq_len, dim, device = DEVICE)
V = torch.randn(batch_size, heads, seq_len, dim, device = DEVICE)

def naive_attention(Q, K, V):
  scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim)
  weights = torch.softmax(scores, dim = -1)
  return torch.matmul(weights, V)

mha = MultiHeadedAttention(embedding_dim = dim, num_heads = 1).to(DEVICE).eval()
out_reference = naive_attention(Q, K, V)
out_mha, _ = mha(Q.view(batch_size, seq_len, dim))
out_flash = flash_attention(Q, K, V)

print("MHA and reference:  ", torch.allclose(out_mha.view(out_reference.size), out_reference, atol = 1e-3))
print("Flash and reference:  ", torch.allclose(out_flash.view(out_reference.size), out_reference, atol = 1e-3))

def benchmark(function, *args, iterations = 100):
  torch.cuda.synchronize(); t0 = time.time()
  for i in range(iterations):
    function(*args)
  torch.cuda.synchronize()
  return (time.time() - t0) / iterations

print(f"Naive Attention : {benchmark(naive_attention, Q, K, V)*1e-3:.3f} ms")
print(f"Multi Headed Attention 1 head : {benchmark(mha, Q.view(batch_size, seq_len, dim))*1e-3:.3f} ms")
print(f"Flash Attention : {benchmark(flash_attention, Q, K, V)*1e-3:.3f} ms")
