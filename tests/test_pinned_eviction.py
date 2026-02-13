
import unittest
import os
import shutil
import tempfile
import json
import time
import numpy as np
from unittest.mock import MagicMock
import sys

# Mock imports if needed (but we use real logic mostly)
# We assume mti_evo is installed or in path.
# If running via run_command, we need to ensure paths.
sys.path.insert(0, os.path.abspath("src"))

from mti_evo.core.config import MTIConfig
from mti_evo.core.lattice import HolographicLattice

class TestPinnedEviction(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.anchor_file = os.path.join(self.test_dir, "anchors_pinned.json")
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_pinned_vs_unpinned_attractor(self):
        """
        Verify that:
        1. Pinned Anchor (Seed 1) is NEVER evicted, even if weak/old.
        2. Unpinned Anchor (Seed 2) IS evicted if weak/old.
        3. Unpinned Anchor is resurrected by reinforcement.
        """
        # Create Anchors
        anchors_data = {
            "anchors": [
                {
                    "seed": 1, 
                    "pattern": [1.0, 0.0], 
                    "tags": ["pinned_weak"],
                    "pinned": True,
                    "target_pre": 0.1,
                    "reinforce_steps": 1
                },
                {
                    "seed": 2, 
                    "pattern": [0.0, 1.0], 
                    "tags": ["unpinned_attractor"],
                    "pinned": False,
                    "target_pre": 0.1,
                    "reinforce_steps": 1
                }
            ]
        }
        with open(self.anchor_file, 'w') as f:
            json.dump(anchors_data, f)
            
        # Config
        config = MTIConfig()
        config.deterministic = True
        config.capacity_limit = 3 # Very small: 1, 2, +1 dummy -> Full.
        config.anchor_file = self.anchor_file
        config.anchor_reinforcement_freq = 0 # Manual reinforcement trigger
        config.eviction_mode = "full_scan" # Ensure deterministic metabolic check
        config.grace_period = 0 # Allow immediate eviction
        config.passive_decay_rate = 0.1 # Fast decay
        
        lattice = HolographicLattice(config)
        
        # 1. Initialize Anchors (Lazy load happens on access/reinforce, but they are registered)
        # We need to manually stimulate them to put them in tissue? 
        # Or rely on reinforcement to create them?
        # Let's stimulate them to existing state.
        lattice.stimulate([1], [1.0, 0.0], learn=True)
        lattice.stimulate([2], [0.0, 1.0], learn=True)
        lattice.stimulate([3], [0.5, 0.5], learn=True) # Dummy filler
        
        self.assertEqual(len(lattice.active_tissue), 3)
        self.assertIn(1, lattice.active_tissue)
        self.assertIn(2, lattice.active_tissue)
        
        # 2. Degrade Anchors Manually
        # Make them very old and weak
        now = time.time()
        old_time = now - 10000 
        
        neuron1 = lattice.active_tissue[1]
        neuron1.last_accessed = old_time
        neuron1.weights = np.array([0.01, 0.0]) # Weak
        
        neuron2 = lattice.active_tissue[2]
        neuron2.last_accessed = old_time
        neuron2.weights = np.array([0.0, 0.01]) # Weak
        
        # 3. Force Eviction
        # Add Seed 4. Capacity 3. Should evict weakest.
        # Seed 1 is Pinned -> Protected.
        # Seed 2 is Unpinned -> Should be evicted (it's weak/old).
        # Seed 3 is young -> Should stay.
        
        print("Triggering Eviction by adding Seed 4...")
        lattice.stimulate([4], [0.5, 0.5], learn=True)
        
        self.assertIn(1, lattice.active_tissue, "Pinned Anchor 1 must NOT be evicted")
        self.assertNotIn(2, lattice.active_tissue, "Unpinned Anchor 2 SHOULD be evicted")
        self.assertIn(4, lattice.active_tissue, "New seed 4 should be present")
        
        # 4. Resurrect Unpinned Anchor
        # Trigger reinforcement for seed 2
        print("Triggering Reinforcement for Anchor 2...")
        # We need to bypass the freq check or just call reinforce_batch manually
        lattice.anchor_manager.reinforce_batch(lattice, lattice.step_counter)
        
        self.assertIn(2, lattice.active_tissue, "Anchor 2 should be resurrected")
        
    def test_saturation_threshold(self):
        """
        Verify anchor is skipped if resonance > threshold.
        """
        anchors_data = {
            "anchors": [
                {
                    # Seed 10 is Pinned
                    "seed": 10, 
                    "pattern": [1.0, 0.0], 
                    "pinned": True,
                    "target_post": 0.8,
                    "cooldown_steps": 0
                }
            ]
        }
        with open(self.anchor_file, 'w') as f:
            json.dump(anchors_data, f)
            
        config = MTIConfig()
        config.anchor_file = self.anchor_file
        config.anchor_saturation_threshold = 0.90
        
        lattice = HolographicLattice(config)
        
        # Create strong neuron
        lattice.stimulate([10], [1.0, 0.0], learn=True) # Init
        for _ in range(10): 
            lattice.stimulate([10], [1.0, 0.0], learn=True) # Strengthen
            
        # Check resonance
        res = lattice.stimulate([10], [1.0, 0.0], learn=False)
        print(f"Current Resonance: {res}")
        self.assertGreater(res, 0.90)
        
        # Reset last_reinforced to ensure it's not cooldown blocking
        lattice.anchor_manager.anchors[10].last_reinforced = -1
        
        # Trigger Reinforce
        lattice.anchor_manager._reinforce_single(lattice, lattice.anchor_manager.anchors[10], 100)
        
        # Check logs or side effects?
        # Ideally we check returned value or logs.
        # But _reinforce_single returns None.
        # We can check validation: last_reinforced should NOT be updated if skipped?
        # The logic:
        # if pre >= target_hi: return (skipping last_reinforced update)
        
        self.assertEqual(lattice.anchor_manager.anchors[10].last_reinforced, -1, 
                         "Should not update last_reinforced if skipped due to saturation")

if __name__ == "__main__":
    unittest.main()
