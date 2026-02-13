import pytest
import time
import json
import os
import numpy as np
from mti_evo.core.lattice import HolographicLattice
from mti_evo.core.config import MTIConfig

@pytest.mark.benchmark
class TestBatchingBenchmarks:
    """
    Benchmarks for Batch Processing (Opt-3).
    """

    @pytest.fixture
    def bench_config(self):
        return MTIConfig(
            embedding_dim=512,
            capacity_limit=5000,
            initial_lr=0.1,
            random_seed=42,
            deterministic=True
        )

    @pytest.fixture
    def lattice(self, bench_config):
        return HolographicLattice(bench_config)

    def test_batch_throughput_inference(self, lattice):
        """
        Compare Serial Loop vs Batch Vectorization for Inference (learn=False).
        """
        BATCH_SIZE = 100
        N_BATCHES = 10
        TOTAL_ITEMS = BATCH_SIZE * N_BATCHES
        
        rng = np.random.default_rng(42)
        # Pre-generate inputs
        all_seeds = rng.integers(0, 10000, size=(N_BATCHES, BATCH_SIZE))
        all_signals = rng.random((N_BATCHES, BATCH_SIZE, lattice.config.embedding_dim), dtype=np.float32)
        
        # 1. Warmup
        # Fill lattice somewhat to have active neurons
        warm_seeds = rng.integers(0, 10000, size=500)
        warm_signals = rng.random((500, lattice.config.embedding_dim), dtype=np.float32)
        lattice.stimulate_batch(warm_seeds, warm_signals, learn=True)
        
        print(f"\nActive Neurons after warmup: {len(lattice.active_tissue)}")
        
        # 2. Serial Benchmark
        start_serial = time.perf_counter()
        for b in range(N_BATCHES):
            seeds = all_seeds[b]
            signals = all_signals[b]
            for i in range(BATCH_SIZE):
                lattice.stimulate([seeds[i]], signals[i], learn=False)
        end_serial = time.perf_counter()
        time_serial = end_serial - start_serial
        
        # 3. Batch Benchmark
        start_batch = time.perf_counter()
        for b in range(N_BATCHES):
            lattice.stimulate_batch(all_seeds[b], all_signals[b], learn=False)
        end_batch = time.perf_counter()
        time_batch = end_batch - start_batch
        
        # Stats
        speedup = time_serial / time_batch if time_batch > 0 else 0
        
        stats = {
            "test_name": "test_batch_throughput_inference",
            "total_items": TOTAL_ITEMS,
            "batch_size": BATCH_SIZE,
            "time_serial_sec": time_serial,
            "time_batch_sec": time_batch,
            "speedup_factor": speedup,
            "items_per_sec_serial": TOTAL_ITEMS / time_serial,
            "items_per_sec_batch": TOTAL_ITEMS / time_batch
        }
        
        self._save_perf_log(stats)
        
        print(f"\nBatch Benchmark Results: {json.dumps(stats, indent=2)}")
        
        # Assert sensible speedup (Batch should be faster)
        # Note: minimal overhead might make small batches slower, but 100 should range good.
        assert speedup > 2.0, f"Batching should be >2x faster, got {speedup:.2f}x"

    def _save_perf_log(self, stats):
        log_dir = "perf_logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = int(time.time())
        filename = f"{log_dir}/bench_{timestamp}_{stats['test_name']}.json"
        with open(filename, "w") as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    pytest.main([__file__])
