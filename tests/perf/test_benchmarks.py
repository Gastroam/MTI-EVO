import pytest
import time
import json
import os
import numpy as np
from typing import Dict, Any, List
from mti_evo.core.lattice import HolographicLattice
from mti_evo.core.config import MTIConfig

@pytest.mark.benchmark
class TestCoreBenchmarks:
    """
    Performance benchmarks for MTI-EVO Core.
    Run with: pytest tests/perf/test_benchmarks.py
    """

    @pytest.fixture
    def bench_config(self):
        return MTIConfig(
            embedding_dim=512,
            capacity_limit=1000,  # Small capacity to trigger evictions
            initial_lr=0.1,
            # Ensure deterministic behavior
            random_seed=42,
            deterministic=True
        )

    @pytest.fixture
    def lattice(self, bench_config):
        """Create a lattice in ephemeral mode (no persistence for raw compute bench)."""
        # For baseline compute, we don't need persistence yet.
        # If we add persistence later, we'll inject it via config.
        lattice = HolographicLattice(bench_config)
        yield lattice

    def test_stimulate_throughput(self, lattice):
        """
        Measure throughput of stimulate() calls.
        """
        N_STEPS = 500
        # Generate random signals
        rng = np.random.default_rng(42)
        signals = rng.random((N_STEPS, lattice.config.embedding_dim), dtype=np.float32)
        # Generate some random seeds (e.g., 5 seeds per step)
        seeds_per_step = 5
        seed_streams = rng.integers(0, 10000, size=(N_STEPS, seeds_per_step))
        
        start_time = time.perf_counter()
        
        for i in range(N_STEPS):
            # Feed signal
            lattice.stimulate(seed_streams[i], signals[i])
            
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time = total_time / N_STEPS
        
        stats = {
            "test_name": "test_stimulate_throughput",
            "n_steps": N_STEPS,
            "total_time_sec": total_time,
            "avg_step_time_sec": avg_time,
            "steps_per_sec": N_STEPS / total_time,
            "final_active_neurons": len(lattice.active_tissue),
            # Access internal eviction counter if available, or just note active size
            # "evictions": lattice.eviction_count if hasattr(lattice, 'eviction_count') else -1
        }
        
        self._save_perf_log(stats)
        
        # Basic assertion to pass CI
        assert total_time > 0
        print(f"\nBenchmark Results: {json.dumps(stats, indent=2)}")

    def _save_perf_log(self, stats: Dict[str, Any]):
        """Save benchmark stats to perf_logs/."""
        log_dir = "perf_logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = int(time.time())
        filename = f"{log_dir}/bench_{timestamp}_{stats['test_name']}.json"
        with open(filename, "w") as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    # Allow running directly for quick check
    pytest.main([__file__])
