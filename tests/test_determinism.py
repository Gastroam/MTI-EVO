
import unittest
import numpy as np
import json
import logging
import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from mti_evo.core.lattice import HolographicLattice
from mti_evo.mti_config import MTIConfig

# Configure logging to avoid noise during tests
logging.basicConfig(level=logging.ERROR)

class DeterministicClock:
    def __init__(self, start_time=1000.0, tick_step=1.0):
        self.current_time = start_time
        self.tick_step = tick_step

    def __call__(self):
        t = self.current_time
        self.current_time += self.tick_step
        return t

    def reset(self, start_time=None):
        if start_time:
            self.current_time = start_time
        else:
            self.current_time = 1000.0

class TestDeterminism(unittest.TestCase):

    def run_simulation(self, seed, steps=10):
        # 1. Config
        cfg = MTIConfig()
        cfg.random_seed = seed
        cfg.deterministic = True
        cfg.capacity_limit = 100 # Small capacity to force evictions if needed
        cfg.eviction_sample_size = 10
        
        # 2. Clock
        clock = DeterministicClock(start_time=1000.0, tick_step=0.1)

        # 3. Lattice
        lattice = HolographicLattice(config=cfg, time_fn=clock)

        # 4. Input Stream (Deterministic)
        # Generate some synthetic data using a LOCAL rng (not the lattice's)
        # to ensure the input data itself is consistent across runs.
        input_rng = np.random.default_rng(42) 
        
        # 5. Stimulation Loop
        for i in range(steps):
            # Random seed for the "concept"
            concept_id = input_rng.integers(0, 1000)
            # Random vector
            vector = input_rng.normal(0, 1, size=(5,))
            
            lattice.stimulate(
                seed_stream=[concept_id], 
                input_signal=vector, 
                learn=True
            )
            
        return lattice.snapshot(deterministic=True)

    def test_reproducibility(self):
        """Test that same seed + same inputs = identical state."""
        SEED_A = 12345
        
        print("\nRunning Run 1 (Seed A)...")
        snapshot_1 = self.run_simulation(SEED_A, steps=50)
        
        print("Running Run 2 (Seed A)...")
        snapshot_2 = self.run_simulation(SEED_A, steps=50)
        
        # Convert to JSON string for easy diffing if needed, or just compare
        json_1 = json.dumps(snapshot_1, sort_keys=True)
        json_2 = json.dumps(snapshot_2, sort_keys=True)
        
        self.assertEqual(json_1, json_2, "Snapshots should be byte-for-byte identical for same seed.")
        print("✅ Run 1 and Run 2 match perfectly.")

    def test_divergence(self):
        """Test that different seeds produce different states."""
        SEED_A = 12345
        SEED_B = 67890 # Different seed
        
        print("Running Run 3 (Seed B)...")
        snapshot_a = self.run_simulation(SEED_A, steps=50)
        snapshot_b = self.run_simulation(SEED_B, steps=50)
        
        json_a = json.dumps(snapshot_a, sort_keys=True)
        json_b = json.dumps(snapshot_b, sort_keys=True)
        
        self.assertNotEqual(json_a, json_b, "Snapshots should differ for different seeds.")
        print("✅ Run A and Run B diverged as expected.")
        
    def test_eviction_determinism(self):
        """Test that eviction logic is also deterministic."""
        # Use small capacity to force evictions
        cfg = MTIConfig()
        cfg.capacity_limit = 5
        cfg.random_seed = 999
        clock = DeterministicClock()
        
        lattice = HolographicLattice(config=cfg, time_fn=clock)
        
        # Fill capacity
        input_rng = np.random.default_rng(42)
        for i in range(10): # Should trigger ~5 evictions
            lattice.stimulate([i], np.zeros((1,)), learn=True)
            
        snap_1 = lattice.snapshot()
        
        # Repeat
        clock = DeterministicClock()
        lattice2 = HolographicLattice(config=cfg, time_fn=clock)
        input_rng = np.random.default_rng(42)
        for i in range(10):
            lattice2.stimulate([i], np.zeros((1,)), learn=True)
            
        snap_2 = lattice2.snapshot()
        
        self.assertEqual(json.dumps(snap_1, sort_keys=True), json.dumps(snap_2, sort_keys=True))
        print("✅ Eviction logic is deterministic.")

if __name__ == "__main__":
    unittest.main()
