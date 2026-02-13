
import unittest
import os
import shutil
import tempfile
import time
import struct
import numpy as np

from mti_evo.core.persistence.mmap import MMapNeuronStore
from mti_evo.core.persistence.jsonl import JsonlPersistence
from mti_evo.core.persistence.manager import PersistenceManager
from mti_evo.core.config import MTIConfig

class TestMMapCorrectness(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mmap_path = os.path.join(self.test_dir, "test.mmap")
        self.dim = 64
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            try:
                # MMap needs explicit close often on Windows or it locks
                pass 
            except: pass
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_basic_io(self):
        """Test put and get with tag matching."""
        store = MMapNeuronStore(self.mmap_path, dim=self.dim, capacity=100)
        
        weights = np.ones(self.dim, dtype='float32')
        velocity = np.zeros(self.dim, dtype='float32')
        
        # Put Seed 1
        store.put(1, weights, velocity, 0.5, 10.0, 100, 1234.5)
        store.flush()
        
        # Get Seed 1
        data = store.get(1)
        self.assertIsNotNone(data)
        np.testing.assert_array_equal(data['weights'], weights)
        self.assertEqual(data['bias'], 0.5)
        self.assertEqual(data['gravity'], 10.0)
        self.assertEqual(data['last_accessed'], 1234.5)
        
        store.close()

    def test_collision_handling(self):
        """Test that different seeds mapping to same slot handle collision."""
        # Force small capacity so 1 and 1+capacity collide
        cap = 10
        store = MMapNeuronStore(self.mmap_path, dim=self.dim, capacity=cap)
        
        seed_a = 5
        seed_b = 5 + cap # Maps to same slot 5
        
        weights_a = np.full(self.dim, 1.0, dtype='float32')
        weights_b = np.full(self.dim, 2.0, dtype='float32')
        
        # 1. Put A
        store.put(seed_a, weights_a, weights_a, 0, 0, 0, 0)
        
        # 2. Get A -> Hit
        data_a = store.get(seed_a)
        self.assertIsNotNone(data_a)
        np.testing.assert_array_equal(data_a['weights'], weights_a)
        
        # 3. Get B -> Miss (Tag Mismatch)
        # Even though slot is occupied, tag is seed_a (5), not seed_b (15)
        data_b = store.get(seed_b)
        self.assertIsNone(data_b, "Should return None on collision/tag mismatch")
        
        # 4. Put B (Overwrite A)
        store.put(seed_b, weights_b, weights_b, 0, 0, 0, 0)
        
        # 5. Get B -> Hit
        data_b2 = store.get(seed_b)
        self.assertIsNotNone(data_b2)
        np.testing.assert_array_equal(data_b2['weights'], weights_b)
        
        # 6. Get A -> Miss (Evicted by B)
        data_a2 = store.get(seed_a)
        self.assertIsNone(data_a2, "A should have been overwritten by B")
        
        store.close()

    def test_delete_logic(self):
        store = MMapNeuronStore(self.mmap_path, dim=self.dim, capacity=100)
        
        store.put(1, np.zeros(64), np.zeros(64), 0,0,0,0)
        self.assertIsNotNone(store.get(1))
        
        store.delete(1)
        self.assertIsNone(store.get(1))
        
        # Count check
        self.assertEqual(store.get_active_count(), 0)
        
        store.close()

class TestPersistenceManagerIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = MTIConfig()
        self.config.persistence_dir = self.test_dir
        self.config.mmap_capacity = 100
        self.config.embedding_dim = 64
        
        self.pm = PersistenceManager(self.config)
        
    def tearDown(self):
        self.pm.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_tiered_write_read(self):
        """Put -> MMAP Hit. Delete -> MMAP Miss."""
        seed = 101
        state = {
            "weights": [0.1]*64,
            "velocity": [0.0]*64,
            "bias": 1.0,
            "gravity": 5.0,
            "age": 10,
            "last_accessed": time.time(),
            "label": "tiered_test"
        }
        
        # 1. Put
        self.pm.put_neuron(seed, state)
        
        # Verify in MMAP
        from_mem = self.pm.get_neuron(seed)
        self.assertIsNotNone(from_mem)
        self.assertEqual(from_mem['gravity'], 5.0)
        
        # Verify WAL file exists (it is flushed in upsert_neurons?)
        # upsert_neurons clears WAL after writing to JSONL.
        # So JSONL should have it.
        jsonl_path = os.path.join(self.test_dir, "neurons.jsonl")
        self.assertTrue(os.path.exists(jsonl_path))
        
        # 2. Delete
        self.pm.delete_neuron(seed)
        
        # Verify MMAP miss
        self.assertIsNone(self.pm.get_neuron(seed))
        
        # Verify JSONL index miss
        # We can't easily check internal index of JSONL inside PM without access
        # But we can check if we reload PM
        self.pm.close()
        
        # Reload
        pm2 = PersistenceManager(self.config)
        self.assertIsNone(pm2.get_neuron(seed), "Should be deleted in WAL/JSONL too")
        pm2.close()

if __name__ == "__main__":
    unittest.main()
