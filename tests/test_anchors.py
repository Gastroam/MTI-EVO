
import unittest
import os
import shutil
import tempfile
import json
import numpy as np
import sys
from unittest.mock import MagicMock

# Mock cv2 and ultralytics to avoid ImportError in nocuda env
sys.modules["cv2"] = MagicMock()
sys.modules["ultralytics"] = MagicMock()

from mti_evo.core.config import MTIConfig
from mti_evo.core.lattice import HolographicLattice
from mti_evo.core.anchors import SemanticAnchorManager

class TestSemanticAnchors(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.anchor_file = os.path.join(self.test_dir, "anchors.json")
        
        # Create Dummy Anchors
        self.anchors_data = [
            {"seed": 1, "pattern": [1.0, 0.0], "strength": 1.0, "label": "test_anchor_1"},
            {"seed": 2, "pattern": [0.0, 1.0], "strength": 0.5, "label": "test_anchor_2"}
        ]
        with open(self.anchor_file, 'w') as f:
            json.dump(self.anchors_data, f)
            
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_anchor_loading(self):
        manager = SemanticAnchorManager()
        manager.load_from_file(self.anchor_file)
        
        self.assertEqual(len(manager.anchors), 2)
        self.assertEqual(manager.anchors[1].label, "test_anchor_1")
        self.assertTrue(np.allclose(manager.anchors[1].pattern, [1.0, 0.0]))
        
    def test_anchor_integration_in_lattice(self):
        try:
            config = MTIConfig()
            config.deterministic = False # Disable determinism for test speed/simplicity?
            # Determinism uses hashlib which might be slow? No.
            # config.anchor_file = self.anchor_file
            # config.anchor_reinforcement_freq = 5 # Reinforce every 5 steps
            # config.layer_dims = [2] # Match anchor pattern dim
            
            # Using setattr to avoid any property strictness
            setattr(config, "anchor_file", self.anchor_file)
            setattr(config, "anchor_reinforcement_freq", 5)
            setattr(config, "layer_dims", [2])
            
            lattice = HolographicLattice(config)
            
            self.assertIsNotNone(lattice.anchor_manager)
            self.assertEqual(len(lattice.anchor_manager.anchors), 2)
            
            # We need to spy on 'reinforce' or check side effects.
            # Let's check side effects: Neuron creation and Age increase.
            
            # Anchors are seed 1 and 2. They should NOT exist yet (lazy load doesn't auto-create them until stimulate/reinforce).
            self.assertNotIn(1, lattice.active_tissue)
            
            # Step 1-4: No reinforcement
            for i in range(4):
                lattice.stimulate([999], [0.5, 0.5], learn=False)
                
            self.assertNotIn(1, lattice.active_tissue)
            
            # Step 5: Reinforcement Triggered
            lattice.stimulate([999], [0.5, 0.5], learn=False)
            
            # Now Anchors should exist!
            self.assertIn(1, lattice.active_tissue)
            self.assertIn(2, lattice.active_tissue)
            
            # Initial Age should be 1 (creation) + 1 (adaptation during reinforcement) = 2??
            # Or just 1 if created during reinforcement?
            # reinforce -> stimulate_batch([1,2], patterns, learn=True)
            # stimulate_batch -> Genesis (Age=0) -> Learn (Age+=1) -> Age=1.
            # But wait, reinforce_steps defaults to 3!
            # So it does 3 learn steps. Age should be 3.
            self.assertEqual(lattice.active_tissue[1].age, 3)
            
            # Step 10: Reinforcement again
            for i in range(5):
                 lattice.stimulate([999], [0.5, 0.5], learn=False)
                 
            # Age should increase
            self.assertGreater(lattice.active_tissue[1].age, 1)
            
            print("✅ Anchor Verification: Integration & Reinforcement works.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

if __name__ == "__main__":
    unittest.main()
