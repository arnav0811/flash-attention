import torch
import time
import numpy as np
import json
import math

from attention import scaled_dot_product_attention
from flash_attention import flash_attention
from flash_attention_v2 import flash_attention_v2
from multi_headed_attention import MultiHeadedAttention

class FlashAttentionBenchmark:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        # Test configurations
        self.configs = [
            {"batch": 1, "seq_len": 1024, "n_heads": 8, "head_dim": 64},
            {"batch": 1, "seq_len": 2048, "n_heads": 8, "head_dim": 64},
            {"batch": 1, "seq_len": 4096, "n_heads": 8, "head_dim": 64},
            {"batch": 1, "seq_len": 8192, "n_heads": 8, "head_dim": 64}, 
        ]
        
        self.results = []
    
    def create_tensors(self, batch_size, seq_len, n_heads, head_dim):
        dtype = torch.float32 if seq_len > 4096 else torch.float16
        q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=self.device, dtype=dtype)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=self.device, dtype=dtype)
        v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=self.device, dtype=dtype)
        return q, k, v
    
    def scaled_dot_product_wrapper(self, q, k, v):
        output, _ = scaled_dot_product_attention(q, k, v)
        return output
    
    def multi_head_wrapper(self, q, k, v):
        batch_size, n_heads, seq_len, head_dim = q.shape
        embedding_dim = n_heads * head_dim
        
        # Convert from (batch, heads, seq, head_dim) to (batch, seq, embedding_dim)
        x = q.transpose(1, 2).contiguous().view(batch_size, seq_len, embedding_dim)
        cache_key = (embedding_dim, n_heads)
        if not hasattr(self, '_mha_caches'):
            self._mha_caches = {}
        
        if cache_key not in self._mha_caches:
            self._mha_caches[cache_key] = MultiHeadedAttention(embedding_dim, n_heads).to(self.device).half()
        
        mha = self._mha_caches[cache_key]
        output, _ = mha(x)
        # Convert back to (batch, heads, seq, head_dim)
        output = output.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        return output
    
    def benchmark_function(self, func, name, q, k, v):
        try:
            for _ in range(3):
                with torch.no_grad():
                    _ = func(q, k, v)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        except Exception as e:
            return {"name": name, "error": str(e), "time_ms": float('inf')}
        
        # Benchmarking
        times = []
        for _ in range(10):
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
            "scaled_dot_product": self.scaled_dot_product_wrapper,
            "multi_head": self.multi_head_wrapper,
            "flash_v1": flash_attention,
            "flash_v2": flash_attention_v2,
        }
        
        print(f"\n Testing {len(implementations)} implementations...")
        
        for i, config in enumerate(self.configs):
            print(f"\n=== Test {i+1}/{len(self.configs)} ===")
            print(f"Batch: {config['batch']}, Seq: {config['seq_len']}, "
                  f"Heads: {config['n_heads']}, Head Dim: {config['head_dim']}")
            
            # Create test data
            q, k, v = self.create_tensors(
                config["batch"], config["seq_len"], 
                config["n_heads"], config["head_dim"]
            )
            
            config_results = []
            
            # Test each implementation
            for impl_name, impl_func in implementations.items():
                result = self.benchmark_function(impl_func, impl_name, q, k, v)
                config_results.append(result)
                
                if "error" in result:
                    print(f"  {impl_name:15}: ERROR - {result['error']}")
                else:
                    print(f"  {impl_name:15}: {result['time_ms']:.2f}ms (±{result['std_ms']:.2f}ms)")
            
            # Store results
            self.results.append({
                "config": config,
                "results": config_results
            })
    
    def print_summary(self):
        print("\n" + "="*5)
        print("Performance")
        
        # Calculate averages across all configs
        impl_times = {}
        for test in self.results:
            for result in test["results"]:
                name = result["name"]
                if "error" not in result:
                    if name not in impl_times:
                        impl_times[name] = []
                    impl_times[name].append(result["time_ms"])
        
        # Print averages
        print("\n Average Performance:")
        for name, times in impl_times.items():
            avg_time = np.mean(times)
            print(f"  {name:15}: {avg_time:.2f}ms")
        
        # Find fastest
        if impl_times:
            fastest = min(impl_times.keys(), key=lambda x: np.mean(impl_times[x]))
            print(f"\n Fastest: {fastest}")
        
        # Calculate speedups vs scaled dot product
        if "scaled_dot_product" in impl_times and len(impl_times) > 1:
            baseline = np.mean(impl_times["scaled_dot_product"])
            print(f"\n Speedup vs Scaled Dot Product:")
            for name, times in impl_times.items():
                if name != "scaled_dot_product":
                    speedup = baseline / np.mean(times)
                    print(f"  {name:15}: {speedup:.2f}x")
    
    def save_results(self, filename="benchmark_results.json"):
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n Results saved to {filename}")

def main():
    benchmark = FlashAttentionBenchmark()
    benchmark.run_benchmark()
    benchmark.print_summary()
    benchmark.save_results()

if __name__ == "__main__":
    main()
