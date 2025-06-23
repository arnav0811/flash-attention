import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask = None):
  d_k = Q.size(-1)
  scores = torch.matmul(Q, K.transpose(-2, -1))
  if mask is not None:
    scores = scores.masked_fill(mask == 0, float('-inf'))
  attention_weights = F.softmax(scores, dim = -1)
  output = torch.matmul(attention_weights, V)
  return output, attention_weights
