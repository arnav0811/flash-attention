import torch
import torch.nn as nn
from attention import scaled_dot_product_attention

class MultiHeadedAttention(nn.Module):
  def __init__(self, embedding_dim, num_heads):
    super().__init__()
    assert embedding_dim % num_heads == 0

    self.num_heads = num_heads
    self.head_dim = embedding_dim // num_heads
    self.W_query = nn.Linear(embedding_dim, embedding_dim, dtype=torch.float16)
    self.W_key = nn.Linear(embedding_dim, embedding_dim, dtype=torch.float16)
    self.W_value = nn.Linear(embedding_dim, embedding_dim, dtype=torch.float16)
    self.out_projection = nn.Linear(embedding_dim, embedding_dim, dtype=torch.float16)

  def forward(self, x):
    batch_size, seq_len, _ = x.shape
    Q = self.W_query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    K = self.W_key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    V = self.W_value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    out, attention = scaled_dot_product_attention(Q, K, V)
    out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    return self.out_projection(out), attention
