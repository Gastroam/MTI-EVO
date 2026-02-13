"""
Memory-Mapped Persistence Benchmark
====================================
Compares mmap vs JSON storage performance.
"""
import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from mti_evo.persistence import MMapNeuronStore, JSONNeuronStore


def benchmark_store(store_class, path, num_neurons=10000, dim=64):
    """Benchmark a storage backend."""
    print(f"\n{'='*50}")
    print(f"Benchmarking: {store_class.__name__}")
    print(f"Neurons: {num_neurons}, Dimension: {dim}")
    print('='*50)
    
    # Clean up
    if os.path.exists(path):
        os.remove(path)
    
    # Create store
    store = store_class(path, dim=dim, capacity=num_neurons * 2)
    
    # Generate random neuron data
    np.random.seed(42)
    neurons = []
    for i in range(num_neurons):
        # Use positive seed (matches MTI translator truncation)
        seed = abs(hash(f"concept_{i}")) % (2**31)
        weights = np.random.randn(dim).astype('float32')
        velocity = np.zeros(dim, dtype='float32')
        neurons.append((seed, weights, velocity, 0.0, 20.0, 0, time.time()))
    
    # Benchmark WRITE
    start = time.time()
    for seed, weights, velocity, bias, gravity, age, last_accessed in neurons:
        store.put(seed, weights, velocity, bias, gravity, age, last_accessed)
    write_time = (time.time() - start) * 1000
    print(f"✏️  Write {num_neurons} neurons: {write_time:.2f} ms")
    
    # Benchmark FLUSH
    start = time.time()
    store.flush()
    flush_time = (time.time() - start) * 1000
    print(f"💾 Flush to disk: {flush_time:.2f} ms")
    
    # Close and reopen (simulate reload)
    store.close()
    
    # Benchmark LOAD (reopen)
    start = time.time()
    store = store_class(path, dim=dim, capacity=num_neurons * 2)
    load_time = (time.time() - start) * 1000
    print(f"📂 Reopen file: {load_time:.2f} ms")
    
    # Benchmark READ (random access)
    sample_seeds = [neurons[i][0] for i in np.random.choice(len(neurons), 1000, replace=False)]
    start = time.time()
    for seed in sample_seeds:
        data = store.get(seed)
    read_time = (time.time() - start) * 1000
    print(f"👁️  Read 1000 neurons: {read_time:.2f} ms")
    
    # Verify integrity
    test_seed, test_weights, *_ = neurons[0]
    retrieved = store.get(test_seed)
    if retrieved is not None:
        integrity = np.allclose(retrieved['weights'], test_weights)
        print(f"✅ Integrity check: {'PASS' if integrity else 'FAIL'}")
    else:
        print(f"❌ Integrity check: FAIL (neuron not found)")
    
    store.close()
    
    # File size
    file_size = os.path.getsize(path) / (1024 * 1024)
    print(f"📦 File size: {file_size:.2f} MB")
    
    return {
        'write_ms': write_time,
        'flush_ms': flush_time,
        'load_ms': load_time,
        'read_1k_ms': read_time,
        'file_mb': file_size
    }


def main():
    print("🧪 MTI-EVO Persistence Benchmark")
    print("="*60)
    
    num_neurons = 10000
    dim = 64
    
    # Benchmark mmap
    mmap_results = benchmark_store(
        MMapNeuronStore, 
        ".mti-brain/cortex_test.mmap",
        num_neurons=num_neurons,
        dim=dim
    )
    
    # Benchmark JSON
    json_results = benchmark_store(
        JSONNeuronStore,
        ".mti-brain/cortex_test.json", 
        num_neurons=num_neurons,
        dim=dim
    )
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} {'MMap':>12} {'JSON':>12} {'Speedup':>12}")
    print("-"*60)
    
    for key in ['write_ms', 'flush_ms', 'load_ms', 'read_1k_ms', 'file_mb']:
        mmap_val = mmap_results[key]
        json_val = json_results[key]
        if json_val > 0:
            speedup = json_val / mmap_val if mmap_val > 0 else float('inf')
        else:
            speedup = 1.0
        print(f"{key:<25} {mmap_val:>10.2f}  {json_val:>10.2f}  {speedup:>10.1f}x")
    
    print("\n✅ Benchmark complete!")
    print("🧠 Recommendation: Use mmap for production, JSON for debugging")


if __name__ == "__main__":
    main()
