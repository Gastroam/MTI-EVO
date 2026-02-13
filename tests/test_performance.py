
import unittest
import numpy as np
import time
from mti_evo.core.config import MTIConfig
from mti_evo.core.lattice import HolographicLattice

class TestPerformance(unittest.TestCase):
    
    def test_batch_equivalence(self):
        """
        D2: Verify that stimulate_batch produces identical results to serial stimulate loop.
        """
        cfg = MTIConfig()
        cfg.random_seed = 42
        cfg.learning_rate = 0.1
        
        # Two identical lattices
        lat_serial = HolographicLattice(config=cfg)
        lat_batch = HolographicLattice(config=cfg)
        
        # Data
        seeds = list(range(100))
        input_vec = [0.5] * 10
        
        # 1. Serial Run
        out_serial = []
        for s in seeds:
            res = lat_serial.stimulate([s], input_vec, learn=True)
            out_serial.append(res)
            
        # 2. Batch Run
        out_batch = lat_batch.stimulate_batch(seeds, input_vec, learn=True)
        
        # Compare Outputs
        np.testing.assert_allclose(out_serial, out_batch, rtol=1e-5, err_msg="Batch output differs from Serial output")
        
        # Compare State (Weights)
        for s in seeds:
            w_serial = lat_serial.active_tissue[s].weights
            w_batch = lat_batch.active_tissue[s].weights
            np.testing.assert_allclose(w_serial, w_batch, rtol=1e-5, err_msg=f"Weight mismatch for seed {s}")

            
        print("✅ Batch Equivalence Verified.")

    def test_batch_speedup(self):
        """
        Measure speedup of batch processing.
        """
        cfg = MTIConfig()
        cfg.random_seed = 42
        
        lattice = HolographicLattice(config=cfg)
        
        # Large batch
        N = 1000
        seeds = list(range(N))
        input_vec = [0.5] * 128 # Larger vector
        
        # Warmup
        lattice.stimulate_batch(seeds[:10], input_vec, learn=False)
        
        # Measure Serial
        t0 = time.time()
        for s in seeds:
            lattice.stimulate([s], input_vec, learn=True)
        t_serial = time.time() - t0
        
        # Reset Lattice for fair comparison (or use new one)
        lattice_batch = HolographicLattice(config=cfg)
        # Pre-seed to avoid genesis cost skewing "stimulation" perf? 
        # Actually genesis is part of the flow.
        
        # Measure Batch
        t0 = time.time()
        lattice_batch.stimulate_batch(seeds, input_vec, learn=True)
        t_batch = time.time() - t0
        
        speedup = t_serial / max(1e-9, t_batch)
        print(f"Serial: {t_serial:.4f}s | Batch: {t_batch:.4f}s | Speedup: {speedup:.2f}x")
        
        # We expect SOME speedup, but maybe not 10x yet due to object overhead.
        self.assertGreater(speedup, 1.0, "Batching should be faster than serial")

if __name__ == "__main__":
    unittest.main()
