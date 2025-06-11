import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.key_dim = embedding_dim
        self.W_query = nn.Linear(embedding_dim, self.key_dim, bias = False)
        self.W_key = nn.Linear(embedding_dim, self.key_dim, bias = False)
        self.W_value = nn.Linear(embedding_dim, embedding_dim, bias = False)

    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.key_dim)
        attention_weights = F.softmax(attention_scores, dim = -1)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights
