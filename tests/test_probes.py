
import unittest
import os
import shutil
import tempfile
import json
import numpy as np
from unittest.mock import MagicMock

# Mock Lattice
class MockNeuron:
    def __init__(self, weights):
        self.weights = np.array(weights)
        self.semantic_vector = None # Test default fallback first

class MockLattice:
    def __init__(self):
        self.active_tissue = {}

from mti_evo.core.probes import SemanticProbeRunner, ProbePair, ProbeResult

class TestSemanticProbes(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.probe_file = os.path.join(self.test_dir, "probes.json")
        
        # Setup Lattice
        self.lattice = MockLattice()
        self.runner = SemanticProbeRunner(self.lattice)
        
        # Add Mock Neurons
        # Seed 1: Vector [1, 0]
        # Seed 2: Vector [0, 1] -> Orthogonal (Cos=0)
        # Seed 3: Vector [1, 0] -> Same as 1 (Cos=1)
        # Seed 4: Vector [-1, 0] -> Opposite (Cos=-1)
        self.lattice.active_tissue[1] = MockNeuron([1.0, 0.0])
        self.lattice.active_tissue[2] = MockNeuron([0.0, 1.0])
        self.lattice.active_tissue[3] = MockNeuron([1.0, 0.0])
        self.lattice.active_tissue[4] = MockNeuron([-1.0, 0.0])
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_cosine_similarity(self):
        # 1 vs 3 (Identical -> 1.0)
        self.runner.probes = [
            ProbePair(1, 3, min_cos=0.9)
        ]
        results = self.runner.run()
        self.assertTrue(results[0].passed)
        self.assertAlmostEqual(results[0].cosine_sim, 1.0)
        
        # 1 vs 2 (Orthogonal -> 0.0)
        self.runner.probes = [
            ProbePair(1, 2, max_cos=0.1)
        ]
        results = self.runner.run()
        self.assertTrue(results[0].passed)
        self.assertAlmostEqual(results[0].cosine_sim, 0.0)
        
        # 1 vs 4 (Opposite -> -1.0)
        self.runner.probes = [
            ProbePair(1, 4, max_cos=-0.9)
        ]
        results = self.runner.run()
        self.assertTrue(results[0].passed)
        self.assertAlmostEqual(results[0].cosine_sim, -1.0)
        
    def test_semantic_vector_preference(self):
        # Inject explicit semantic_vector
        neuron = self.lattice.active_tissue[1]
        neuron.semantic_vector = np.array([0.0, 1.0]) # Changed to UP
        # Weights are still RIGHT [1, 0]
        
        # Now 1 vs 2 (UP vs UP) should correspond to 1.0
        # If it used weights, it would be 0.0
        
        self.runner.probes = [
            ProbePair(1, 2, min_cos=0.9)
        ]
        results = self.runner.run()
        self.assertTrue(results[0].passed)
        # Check that semantic_vector was used for at least one side
        self.assertIn("semantic_vector", results[0].vector_source)
        
    def test_load_from_file(self):
        data = {
            "pairs": [
                {"seed_a": 1, "seed_b": 3, "expect": "high", "min_cos": 0.9}
            ]
        }
        with open(self.probe_file, 'w') as f:
            json.dump(data, f)
            
        self.runner.load_probes(self.probe_file)
        self.assertEqual(len(self.runner.probes), 1)
        self.assertEqual(self.runner.probes[0].seed_a, 1)

if __name__ == "__main__":
    unittest.main()
