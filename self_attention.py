import torch
import torch.nn as nn
from attention import scaled_dot_product_attention

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.key_dim = embedding_dim
        self.W_query = nn.Linear(embedding_dim, self.key_dim, bias = False)
        self.W_key = nn.Linear(embedding_dim, self.key_dim, bias = False)
        self.W_value = nn.Linear(embedding_dim, embedding_dim, bias = False)

    def forward(self, x):
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        return scaled_dot_product_attention(Q, K, V)
