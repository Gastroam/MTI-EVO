import pytest
import numpy as np
import copy
from mti_evo.core.lattice import HolographicLattice
from mti_evo.core.config import MTIConfig

class TestBatchEquivalence:
    """
    Ensures stimulate_batch produces identical results to serial stimulate.
    """
    
    @pytest.fixture
    def config(self):
        return MTIConfig(
            embedding_dim=64,
            initial_lr=0.1,
            momentum=0.9,
            gravity=20.0,
            random_seed=42,
            deterministic=True
        )

    def test_batch_vs_serial_inference(self, config):
        """
        Compare serial vs batch simulation for INFERENCE (learn=False).
        Results should be bit-exact since no weight updates occur.
        """
        # Create deterministic lattice
        # Use same seed so weights init identical
        config.random_seed = 999
        serial_lattice = HolographicLattice(config)
        batch_lattice = HolographicLattice(config)
        
        # Verify init state match
        # (Assuming config deterministic works)
        # Force same weights if needed? No, seed should handle it.
        
        # Generate inputs
        batch_size = 20
        rng = np.random.default_rng(123)
        seeds = rng.integers(0, 500, size=batch_size) # Some overlaps likely
        signals = rng.random((batch_size, 64), dtype=np.float32)
        
        # 1. Warmup (CREATE neurons) match
        # We must ensure both lattices have the same neurons created.
        # Batch neurogenesis logic must match Serial neurogenesis.
        # Let's run a "creation pass" first.
        print("Warming up (Serial)...")
        for i in range(batch_size):
            serial_lattice.stimulate([seeds[i]], signals[i], learn=True)
            
        print("Warming up (Batch)...")
        # For batch lattice, we also use SERIAL setup to ensure identical starting state
        # (Avoid debugging Batch Creation logic divergence yet)
        for i in range(batch_size):
            batch_lattice.stimulate([seeds[i]], signals[i], learn=True)
            
        # Verify states are identical after warmup
        assert sorted(serial_lattice.active_tissue.keys()) == sorted(batch_lattice.active_tissue.keys())
        # Check one neuron
        s0 = seeds[0]
        if s0 in serial_lattice.active_tissue:
            n_s = serial_lattice.active_tissue[s0]
            n_b = batch_lattice.active_tissue[s0]
            assert np.allclose(n_s.weights, n_b.weights), "Warmup failed to produce identical state"

        # 2. INFERENCE Pass (learn=False)
        # Now we compare output values
        inputs_inf = rng.random((batch_size, 64), dtype=np.float32)
        
        # Serial Output
        serial_outputs = []
        for i in range(batch_size):
            # stimulate returns avg resonance. logic handles list vs scalar?
            # lattice.stimulate returns scalar avg_resonance_pre (float)
            res = serial_lattice.stimulate([seeds[i]], inputs_inf[i], learn=False)
            serial_outputs.append(res)
            
        # Batch Output
        batch_outputs = batch_lattice.stimulate_batch(seeds, inputs_inf, learn=False)
        
        # Compare
        serial_outputs = np.array(serial_outputs, dtype=float)
        batch_outputs = np.array(batch_outputs, dtype=float)
        
        print(f"\nSerial Out: {serial_outputs[:3]}")
        print(f"Batch  Out: {batch_outputs[:3]}")
        
        assert np.allclose(serial_outputs, batch_outputs, atol=1e-5), \
            f"Inference Mismatch! Max diff: {np.max(np.abs(serial_outputs - batch_outputs))}"
            
        print("Batch Inference Equivalence Verified!")
