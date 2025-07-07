import torch
import time
import numpy as np
import json
import math

from attention import scaled_dot_product_attention
from flash_attention_v1_fixed import flash_attention
from flash_attention_v2_fixed import flash_attention_v2
from multi_headed_attention import MultiHeadedAttention

class FlashAttentionBenchmark:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # Test configurations - optimized for performance differences
        self.configs = [
            {"batch":1,"seq_len":1024,"n_heads":8,"head_dim":64},
            {"batch":1,"seq_len":4096,"n_heads":8,"head_dim":64},
            {"batch":1,"seq_len":8192,"n_heads":8,"head_dim":64},
            {"batch":1,"seq_len":16384,"n_heads":8,"head_dim":64},
        ]
        
        self.results = []
        self._mha_cache = {}
    
    def create_tensors(self, batch_size, seq_len, n_heads, head_dim):
        dtype = torch.float16
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=self.device, dtype=dtype)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=self.device, dtype=dtype)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=self.device, dtype=dtype)
        return q, k, v
    
    def naive_attention_wrapper(self, q, k, v):
        batch_size, n_heads, seq_len, head_dim = q.shape
        
        # Reshape to combine batch and heads
        q_reshaped = q.view(batch_size * n_heads, seq_len, head_dim)
        k_reshaped = k.view(batch_size * n_heads, seq_len, head_dim)
        v_reshaped = v.view(batch_size * n_heads, seq_len, head_dim)
        
        output, _ = scaled_dot_product_attention(q_reshaped, k_reshaped, v_reshaped)
        output = output.view(batch_size, n_heads, seq_len, head_dim)
        return output
    
    def multi_head_wrapper(self, q, k, v):
        batch_size, n_heads, seq_len, head_dim = q.shape
        embedding_dim = n_heads * head_dim
        
        # Convert to input format expected by MHA
        x = q.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim).to(torch.float16)
        
        # Cache MHA to avoid overhead
        cache_key = (embedding_dim, n_heads)
        if cache_key not in self._mha_cache:
            self._mha_cache[cache_key] = MultiHeadedAttention(embedding_dim, n_heads).to(self.device)
        
        mha = self._mha_cache[cache_key]
        output, _ = mha(x)
        output = output.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        return output
    
    def benchmark_function(self, func, name, q, k, v, warmup_iters=3, bench_iters=10):
        try:
            for _ in range(warmup_iters):
                with torch.no_grad():
                    _ = func(q, k, v)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        except Exception as e:
            print(f"  {name:15}: ERROR during warmup - {str(e)}")
            return {"name": name, "error": str(e), "time_ms": float('inf')}
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Benchmarking
        times = []
        for _ in range(bench_iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            try:
                with torch.no_grad():
                    output = func(q, k, v)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
            except Exception as e:
                return {"name": name, "error": str(e), "time_ms": float('inf')}
        
        return {
            "name": name,
            "time_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
        }
    
    def run_benchmark(self):   
        implementations = {
            "naive_attention": self.naive_attention_wrapper,
            "multi_head"     : self.multi_head_wrapper,
            "flash_v1"       : lambda q,k,v: flash_attention(q,k,v, BLOCK_SIZE_M=64, BLOCK_SIZE_N=32),
            "flash_v2"       : lambda q,k,v: flash_attention_v2(q,k,v, BLOCK_SIZE_M=128, BLOCK_SIZE_N=64),
        }
        
        for i, config in enumerate(self.configs):
            print(f"\n Test {i+1}/{len(self.configs)}")
            print(f"Batch: {config['batch']}, Seq: {config['seq_len']}, "
                  f"Heads: {config['n_heads']}, Head Dim: {config['head_dim']}")
      
            q, k, v = self.create_tensors(
                config["batch"], config["seq_len"], 
                config["n_heads"], config["head_dim"]
            )
            config_results = []

            for impl_name, impl_func in implementations.items():
                result = self.benchmark_function(impl_func, impl_name, q, k, v)
                config_results.append(result)
                
                if "error" in result:
                    print(f"  {impl_name:15}: ERROR - {result['error']}")
                else:
                    print(f"  {impl_name:15}: {result['time_ms']:.2f}ms (±{result['std_ms']:.2f}ms)")
            
            self.results.append({
                "config": config,
                "results": config_results
            })
            
            # Clear cache between configs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def print_summary(self):
        # Calculate averages across all configs
        impl_times = {}
        for test in self.results:
            for result in test["results"]:
                name = result["name"]
                if "error" not in result:
                    if name not in impl_times:
                        impl_times[name] = []
                    impl_times[name].append(result["time_ms"])
        
        sorted_impls = sorted(impl_times.items(), key=lambda x: np.mean(x[1]))
        for name, times in sorted_impls:
            avg_time = np.mean(times)
            std_time = np.std(times)
        
        if len(sorted_impls) >= 4:
            fastest_name = sorted_impls[0][0]
            print(f"\nFastest Implementation: {fastest_name}")
            
        # Calculate speedups vs naive attention
        if "naive_attention" in impl_times and len(impl_times) > 1:
            baseline = np.mean(impl_times["naive_attention"])
            print(f"\nSpeedup vs Naive Attention:")
            for name, times in sorted_impls:
                if name != "naive_attention":
                    speedup = baseline / np.mean(times)
                    print(f"  {name:15}: {speedup:.2f}x faster")
    
    def save_results(self, filename="benchmark_results.json"):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")

def main():
    print("Flash Attention Benchmark")
    
    benchmark = FlashAttentionBenchmark()
    benchmark.run_benchmark()
    benchmark.print_summary()
    benchmark.save_results()

if __name__ == "__main__":
    main()
