# flash-attention
Implementation of [Flash Attention](https://arxiv.org/pdf/2205.14135) and [Flash Attention v2](https://arxiv.org/pdf/2307.08691) in Triton and benchmarking against a naive PyTorch implementation of Scaled Dot Product Attention and Multi Headed Attention.

# Overview
Scaled Dot Product Attention is **O(n²)** in memory & time. Flash Attention uses online softmax and streams tiles through on-chip SRAM, slashing memory from O(n²) to O(n) and makes long sequence inference dramatically faster.

# Benchmarking
Benchmarked the 4 implementations on the following configs 
```
self.configs = [
            {"batch":1,"seq_len":1024,"n_heads":8,"head_dim":64},
            {"batch":1,"seq_len":4096,"n_heads":8,"head_dim":64},
            {"batch":1,"seq_len":8192,"n_heads":8,"head_dim":64},
            {"batch":1,"seq_len":16384,"n_heads":8,"head_dim":64},
]
```

```
implementations = {
            "naive_attention": self.naive_attention_wrapper,
            "multi_head"     : self.multi_head_wrapper,
            "flash_v1"       : lambda q,k,v: flash_attention(q,k,v, BLOCK_SIZE_M=64, BLOCK_SIZE_N=32),
            "flash_v2"       : lambda q,k,v: flash_attention_v2(q,k,v, BLOCK_SIZE_M=128, BLOCK_SIZE_N=64),
}
```

# Results
```
Batch: 1, Seq: 1024, Heads: 8, Head Dim: 64
  naive_attention: 0.88ms 
  multi_head     : 1.07ms
  flash_v1       : 0.37ms 
  flash_v2       : 0.40ms

Batch: 1, Seq: 4096, Heads: 8, Head Dim: 64
  naive_attention: 13.71ms
  multi_head     : 10.81ms
  flash_v1       : 1.70ms 
  flash_v2       : 1.97ms 

Batch: 1, Seq: 8192, Heads: 8, Head Dim: 64
  naive_attention: 32.78ms 
  multi_head     : 34.90ms
  flash_v1       : 8.50ms
  flash_v2       : 6.71ms

Batch: 1, Seq: 16384, Heads: 8, Head Dim: 64
  naive_attention: 147.47ms
  multi_head     : 157.54ms
  flash_v1       : 33.98ms
  flash_v2       : 23.35ms

Fastest Implementation: flash_v2

Speedup vs Naive Attention:
  flash_v2       : 6.01x faster
  flash_v1       : 4.37x faster
  multi_head     : 0.95x faster
```
