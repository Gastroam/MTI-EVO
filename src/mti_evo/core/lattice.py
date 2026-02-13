"""
Holographic lattice orchestration logic.
"""

import time
import sys
import json
import os
import hashlib
import math
import random
import numpy as np
import os
import weakref

from typing import List, Dict, Any, Optional

from mti_evo.core.eviction import get_policy
from mti_evo.core.eviction.policy import EvictionPolicy
from mti_evo.core.neuron import MTINeuron
from mti_evo.core.anchors import SemanticAnchorManager
from mti_evo.core.config import MTIConfig
from mti_evo.core.logger import get_logger
from mti_evo.telemetry import TelemetrySystem
# Helper for cosine similarity
from mti_evo.core.neuron import cosine_similarity
# Plugin System
from mti_evo.core.plugins import PluginManager

try:
    from mti_evo.mti_proprioceptor import MTIProprioceptor
except ImportError:
    MTIProprioceptor = None


logger = get_logger("MTI-Core")


class HolographicLattice:
    """
    MTI-EVO Virtual Cortex (Sparse Auto-Encoder).
    Implements infinite addressable memory using Resonant Seeds as hash keys.
    Optimized for systems with limited VRAM by using Lazy Instantiation.
    """

    def __init__(self, config: MTIConfig = None, time_fn=None):
        if config is None:
            config = MTIConfig()

        self.config = config
        
        # Time Source (Dependency Injection for Determinism)
        self.time_fn = time_fn if time_fn else time.time

        # The Hologram: A sparse map of Active Neurons.
        self.active_tissue = {}
        self.capacity_limit = config.capacity_limit
        self.grace_period = config.grace_period
        
        # RNG Setup (Deterministic)
        # We always use the config seed unless explicitly told otherwise
        seed = getattr(config, "random_seed", 1337)
        if not getattr(config, "deterministic", True):
             seed = None # OS Entropy
        self.rng = np.random.default_rng(seed)
        
        # Telemetry
        self.telemetry = TelemetrySystem() if config.telemetry_enabled else None

        # Persistence (Tiered Manager)
        # [Phase J] We instantiate the manager if not provided injection-style.
        self.persistence_manager = getattr(config, "persistence_manager", None)
        
        if self.persistence_manager is None:
             # Pure ephemeral mode if not injected.
             # This aligns with "Core should not import upper layers" (if PersistenceManager was upper)
             # And "CortexMemory should be the only owner" (so it must be passed in).
             logger.warning("No Persistence Manager injected. Holographic Lattice running in EPHEMERAL mode.")


        # [SELF-AWARENESS] Proprioception (Internal Sense of State)
        self.proprioceptor = None
        if MTIProprioceptor:
            self.proprioceptor = MTIProprioceptor(self)
        self.last_stimulation_metrics = None
        
        # [Phase F1] Semantic Anchors
        self.anchor_manager = None
        if hasattr(self.config, "anchor_file"):
             from mti_evo.core.anchors import SemanticAnchorManager
             self.anchor_manager = SemanticAnchorManager(self.config)
        
        # [Phase H] Plugin System
        from mti_evo.core.plugins import PluginManager
        self.plugin_manager = PluginManager()
        
        if getattr(self.config, "enable_hive", False):
            try:
                # Try standard import first
                try:
                    from mti_evo_plugins.hive.plugin import HivePlugin
                    hive = HivePlugin(self.config)
                    self.plugin_manager.register(hive, self)
                    logger.info("Hive Plugin Attached.")
                except ImportError:
                    # Fallback: We do not modify sys.path.
                    # User must ensure mti_evo_plugins is in path.
                    logger.warning("Hive Plugin not found in path. Ensure PYTHONPATH includes it.")
                    # from mti_evo_plugins.hive.plugin import HivePlugin
                    # hive = HivePlugin(self.config)
                    # self.plugin_manager.register(hive, self)
            except Exception as e:
                logger.error(f"Failed to initialize Hive Plugin: {e}")

        # Eviction Policy (Pluggable)
        policy_name = getattr(config, "eviction_policy_name", "standard")
        self.eviction_policy = get_policy(policy_name)
        self.step_counter = 0

        # Reproducibility Report
        import platform
        
        # Deterministic config hash
        cfg_dict = vars(config) if hasattr(config, "__dict__") else {}
        # Simple stable string for hashing (filtering out complex objects if any)
        # We focus on the fields defined in MTIConfig dataclass
        stable_cfg = {k: v for k, v in cfg_dict.items() if isinstance(v, (int, float, str, bool, tuple))}
        cfg_str = json.dumps(stable_cfg, sort_keys=True, separators=(",", ":"), default=str)
        # Full SHA256 as requested for lab-grade reproducibility
        cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()
        
        self.repro_report = {
            "random_seed": seed,
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "config_hash": cfg_hash,
            "config_dump": stable_cfg # Full dump
        }

        logger.info(
            f"Holographic Lattice initialized. Capacity: {self.capacity_limit}, Grace: {self.grace_period}"
        )
        # Log concise report, full dump is in self.repro_report
        logger.info(f"Reproducibility Report (Hash): {cfg_hash}")
        
        # [Phase B3] Save Repro Report to Artifacts
        self._save_repro_report()

    def _save_repro_report(self):
        """
        Writes the reproducibility report to a JSON file in 'repro_logs/.
        This creates an immutable audit trail for every lattice instantiation.
        """
        try:
            log_dir = "repro_logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Use self.time_fn() for timestamp if it returns comparable number, 
            # otherwise fallback to time.time() for filename uniqueness.
            # actually time_fn might be a mock object.
            try:
                ts = int(time.time()) 
            except:
                ts = 0
                
            filename = f"{log_dir}/{ts}_{self.repro_report['config_hash']}.json"
            
            with open(filename, "w") as f:
                json.dump(self.repro_report, f, indent=2, sort_keys=True, default=str)
        except Exception as e:
            logger.warning(f"Failed to write repro report: {e}")

    def snapshot(self, deterministic=True):
        """
        Returns a deterministic state snapshot for auditing/verification.
        deterministic=True ensures the output is sorted and float-stable.
        """
        if not deterministic:
            # Return raw state (not guaranteed stable iteration order)
            return self.active_tissue
            
        # Stable Snapshot
        # Returns a sorted list of neuron states
        snapshot_data = []
        for seed in sorted(self.active_tissue.keys()):
            neuron = self.active_tissue[seed]
            snapshot_data.append({
                "seed": int(seed), # Ensure int for JSON serialization
                "weights": neuron.weights.tolist(), # Convert to list for stable float comparison (better than bytes sometimes, but exact matches required)
                "bias": float(neuron.bias),
                "velocity": neuron.velocity.tolist(),
                "age": int(neuron.age),
                "gravity": float(neuron.gravity)
            })
        return snapshot_data

    def _prune_weakest(self):
        """
        Eviction strategy delegated to Policy.
        Returns 1 if eviction occurred, 0 otherwise.
        """
        # Pass both rng and time_fn for determinism
        target_seed, fallback, meta = self.eviction_policy.pick_candidate(
            self.active_tissue, 
            self.rng, 
            self.config,
            time_fn=self.time_fn
        )
        if target_seed is None:
            return 0 # Nothing to evict

        # Fire Event
        self.plugin_manager.trigger("on_eviction", {"seed": target_seed, "reason": meta.get('reason', 'unknown')})

        # Delete from tissue
        del self.active_tissue[target_seed]

        if self.telemetry:
            # enrich with reason/scores if available in meta
            self.telemetry.record_eviction(target_seed, meta)

        logger.debug(f"Evicted seed {target_seed} ({meta.get('reason', 'unknown')})")
        return 1

    def stimulate(self, seed_stream, input_signal, learn=True, labels=None):
        """
        Process a signal trace through specific resonance coordinates.
        input_signal: Scalar or Vector pattern to be processed.
        learn: If True, the activated neurons will Adapt (train) to resonate with this signal.
        labels: Optional dict mapping {seed: "Label"} to assigning meaning to neurons.
        """
        start_time = time.time()
        layer_response_pre = []
        layer_response_post = []
        evictions = 0

        # Pre-process Input Signal
        x_in = np.atleast_2d(input_signal)
        input_size = x_in.shape[1]

        x_in = np.atleast_2d(input_signal)
        input_size = x_in.shape[1]

        self.step_counter += 1
        if self.anchor_manager:
            self.anchor_manager.check_and_reinforce(self, self.step_counter)

        for seed in seed_stream:
            # 1. NEUROGENESIS (Lazy Instantiation)
            if seed not in self.active_tissue:
                # [LAZY LOADING] Check Hippocampus first
                loaded_neuron = None
                if self.persistence_manager:
                    loaded_neuron = self.persistence_manager.get_neuron(seed)

                if loaded_neuron:
                    # RECALL: Found in MMAP/WAL
                    if isinstance(loaded_neuron, dict):
                        # Rehydrate from dict
                        # We need to create a new neuron and populate it
                        # This duplicates logic in load() - refactor candidate
                        n = MTINeuron(
                            input_size=input_size,
                            config=self.config,
                            trainable_bias=True,
                            rng=self.rng,
                            time_fn=self.time_fn
                        )
                        n.weights = np.array(loaded_neuron['weights'])
                        n.velocity = np.array(loaded_neuron['velocity'])
                        n.bias = loaded_neuron.get('bias', 0.0)
                        n.gravity = loaded_neuron.get('gravity', 20.0)
                        n.age = loaded_neuron.get('age', 0)
                        n.last_accessed = loaded_neuron.get('last_accessed', 0.0)
                        if 'label' in loaded_neuron:
                             n.label = loaded_neuron['label']
                        self.active_tissue[seed] = n
                    else:
                        # Assume it's already an object (legacy support)
                        self.active_tissue[seed] = loaded_neuron
                else:
                    # GENESIS: New Concept
                    if len(self.active_tissue) >= self.capacity_limit:
                        evictions += int(self._prune_weakest())

                    # If we are pruning, we might have freed space.
                    if len(self.active_tissue) >= self.capacity_limit:
                        layer_response_pre.append(0.0)
                        if learn:
                            layer_response_post.append(0.0)
                        continue

                    # Pass config + lattice rng to new neuron.
                    self.active_tissue[seed] = MTINeuron(
                        input_size=input_size,
                        config=self.config,
                        trainable_bias=True,
                        rng=self.rng,         # Shared Lattice RNG
                        time_fn=self.time_fn  # Shared Time Source
                    )
                    if self.telemetry:
                        self.telemetry.record_neurogenesis()

            # [PHASE 27] Semantic Labeling
            # If a label is provided for this seed, stamp it onto the neuron (DNA).
            if labels and seed in labels:
                self.active_tissue[seed].label = labels[seed]

            neuron = self.active_tissue[seed]

            # 2. ACTIVATION (Perception - First Impression)
            # Measure Resonance BEFORE training to capture Curiosity/Surprise.
            output_pre = neuron.perceive(x_in)
            layer_response_pre.append(float(np.mean(np.atleast_1d(output_pre))))

            # 3. ADAPTATION (Learning)
            # If we are in Learning Mode, we train the neuron to output HIGH (1.0)
            # for this pattern. We are creating an Attractor.
            if learn:
                # Target is Resonance (1.0) - "I recognize this."
                neuron.adapt(x_in, y_true=1.0)
                output_post = neuron.perceive(x_in)
                layer_response_post.append(float(np.mean(np.atleast_1d(output_post))))

        avg_resonance_pre = float(np.mean(layer_response_pre)) if layer_response_pre else 0.0
        avg_resonance_post = (
            float(np.mean(layer_response_post)) if layer_response_post else avg_resonance_pre
        )
        duration_ms = (time.time() - start_time) * 1000

        metrics = {
            "avg_resonance_pre": avg_resonance_pre,
            "active_count": len(self.active_tissue),
            "evictions": evictions,
            "latency_ms": duration_ms,
        }
        if learn:
            metrics["avg_resonance_post"] = avg_resonance_post
        self.last_stimulation_metrics = metrics

        # Telemetry Recording
        if self.telemetry:
            self.telemetry.record_pulse(
                active_count=len(self.active_tissue),
                resonance=avg_resonance_pre,
                duration_ms=duration_ms,
            )

        if getattr(self.config, "stimulate_return_metrics", False):
            return metrics
        return avg_resonance_pre

    def stimulate_batch(self, seed_stream, input_signal, learn=True, labels=None):
        """
        Vectorized stimulation matching MTINeuron physics.
        """
        start_time = time.time()
        input_size = len(self.config.layer_dims) if hasattr(self.config, 'layer_dims') else len(np.atleast_1d(input_signal))
        x_in = np.array(input_signal)
        if x_in.ndim == 1:
            x_in = np.tile(x_in, (len(seed_stream), 1))
        
        batch_size = len(seed_stream)

        self.step_counter += 1
        if self.anchor_manager:
            self.anchor_manager.check_and_reinforce(self, self.step_counter)
        
        # 1. Genesis / Retrieval (Serial fallback)
        new_seeds = [s for s in seed_stream if s not in self.active_tissue]
        if new_seeds:
            # Lazy Load
            if self.persistence_manager:
                for s in new_seeds:
                     loaded = self.persistence_manager.get_neuron(s)
                     if loaded:
                         self.active_tissue[s] = loaded
            
            # Create remaining
            truly_new = [s for s in new_seeds if s not in self.active_tissue]
            current_count = len(self.active_tissue)
            space_needed = len(truly_new)
            
            while current_count + space_needed > self.capacity_limit:
                 if self._prune_weakest():
                     current_count -= 1
                 else:
                     break
                     
            for s in truly_new:
                if len(self.active_tissue) >= self.capacity_limit:
                    break
                dim = x_in.shape[1] if x_in.ndim > 1 else input_size
                self.active_tissue[s] = MTINeuron(
                    input_size=dim,
                    config=self.config,
                    trainable_bias=True,
                    rng=self.rng,
                    time_fn=self.time_fn
                )
                if self.telemetry:
                    self.telemetry.record_neurogenesis()
                if labels and s in labels:
                    self.active_tissue[s].label = labels[s]

        # 2. Collect Valid Neurons
        valid_indices = []
        valid_neurons = []
        for i, s in enumerate(seed_stream):
            if s in self.active_tissue:
                valid_indices.append(i)
                valid_neurons.append(self.active_tissue[s])
        
        if not valid_neurons:
            return [0.0] * batch_size

        # 3. Vectorization (Object -> Arrays)
        # We need W, B, Velocity, Age, Gravity
        W = np.stack([n.weights for n in valid_neurons]) # (N, Dim)
        B = np.array([n.bias for n in valid_neurons])    # (N,)
        V = np.stack([n.velocity for n in valid_neurons]) # (N, Dim)
        Age = np.array([n.age for n in valid_neurons])   # (N,)
        Gravity = np.array([n.gravity for n in valid_neurons]) # (N,)
        
        X = x_in[valid_indices] # (N, Dim)
        
        # 4. Perception: Sigmoid(W . X + b)
        # Note: MTINeuron uses dot(inputs, weights) which is sum(x*w)
        logits = np.einsum('ij,ij->i', X, W) + B
        
        # Sigmoid: 1 / (1 + exp(-x))
        # Stable sigmoid
        output = np.zeros_like(logits)
        mask_pos = logits >= 0
        mask_neg = ~mask_pos
        
        z_pos = np.exp(-logits[mask_pos])
        output[mask_pos] = 1 / (1 + z_pos)
        
        z_neg = np.exp(logits[mask_neg])
        output[mask_neg] = z_neg / (1 + z_neg)
        
        results = [0.0] * batch_size
        for idx, val in zip(valid_indices, output):
            results[idx] = float(val)

        # 5. Learning (Adapt)
        if learn:
            # 5.1 Update Age & Access
            time_now = self.time_fn()
            
            # Match MTINeuron.adapt: Age increments BEFORE LR calculation
            Age += 1
            
            # 5.2 Gravity Update (LTP)
            # self.gravity = min(self.gravity + 0.05, 50.0)
            Gravity = np.minimum(Gravity + 0.05, 50.0)
            
            # 5.3 Loss Calculation
            # "Signal Reinforcer": y_true = 1.0 (Attractor)
            y_true = 1.0
            epsilon = 1e-15
            y_pred_clamped = np.clip(output, epsilon, 1 - epsilon)
            
            # Weight factor: where(y_true==1, gravity, 1.0) -> Gravity since y_true=1
            weight_factor = Gravity
            
            # Error = y_pred - y_true
            error = output - y_true
            weighted_error = error * weight_factor # (N,)
            
            # Gradient: x * weighted_error
            # (N, Dim) = (N, Dim) * (N, 1)
            gradient = X * weighted_error[:, np.newaxis]
            bias_gradient = weighted_error
            
            # 5.4 LR Calculation with Diminishing Returns
            initial_lr = self.config.initial_lr
            decay = self.config.decay_rate
            # current_lr = initial / (1 + decay * age)
            current_lr_vec = initial_lr / (1 + decay * Age)
            
            if getattr(self.config, "diminishing_returns", True):
                # current_magnitude = mean(abs(weights))
                # damping = 1 / (1 + mag/10)
                mags = np.mean(np.abs(W), axis=1)
                damping = 1.0 / (1.0 + (mags / 10.0))
                current_lr_vec *= damping
                
            # 5.5 Momentum Update
            momentum = self.config.momentum
            # vel = (mom * vel) - (lr * grad)
            V = (momentum * V) - (current_lr_vec[:, np.newaxis] * gradient)
            
            # 5.6 Apply to Weights
            W += V
            
            if hasattr(self.config, "trainable_bias") and self.config.trainable_bias: # Or infer from neurons? Assume True for batch
                 B -= current_lr_vec * bias_gradient
                 
            # 5.7 Weight Cap & Normalization
            max_w = getattr(self.config, "weight_cap", 80.0)
            norms = np.linalg.norm(W, axis=1)
            
            mask_cap = norms > max_w
            if np.any(mask_cap):
                scale_factors = max_w / norms[mask_cap]
                W[mask_cap] *= scale_factors[:, np.newaxis]
                V[mask_cap] *= 0.5 # Drain kinetic energy
                
            # 6. Scatter Back to Objects
            # This overhead is unavoidable until we refactor to pure tensor lattice.
            # But the math was vectorized.
            for idx, n in enumerate(valid_neurons):
                 n.weights = W[idx]
                 n.bias = float(B[idx])
                 n.velocity = V[idx]
                 n.age = int(Age[idx]) # Already incremented
                 n.gravity = float(Gravity[idx])
                 n.last_accessed = time_now

        else:
             time_now = self.time_fn()
             for n in valid_neurons:
                 n.last_accessed = time_now

        return results

    def load(self, persistence_manager):
        """
        Rehydrates the lattice from disk.
        """
        # NOTE: persistence_manager might be MTIHippocampus or a dict-like adapter.
        # We assume it returns a dict of neuron states.
        # If persistence_manager is the class MTIHippocampus, we need to call .recall()
        data = {}
        if hasattr(persistence_manager, "recall"):
            data = persistence_manager.recall()
        elif hasattr(persistence_manager, "load"):
            data = persistence_manager.load()
        else:
            # Assume it's already data or fail
            data = persistence_manager

        if not isinstance(data, dict):
            return  # Failed or empty

        for seed_str, state in data.items():
            seed = int(seed_str)

            # Defensive check for weights
            if "weights" in state:
                w_data = state["weights"]
                w_array = np.array(w_data)
                input_size = w_array.shape[0] if w_array.ndim > 0 else 1
            else:
                # Legacy fallback
                input_size = 1

            # Re-birth the neuron with correct dimension
            self.active_tissue[seed] = MTINeuron(
                input_size=input_size,
                config=self.config,  # Re-inject config
                trainable_bias=False,  # Assuming Holographic
                rng=self.rng,
                time_fn=self.time_fn
            )

            # Restore state
            neuron = self.active_tissue[seed]
            neuron.weights = np.array(state["weights"])
            neuron.bias = state.get("bias", 0.0)
            neuron.velocity = np.array(state.get("velocity", np.zeros(input_size)))
            neuron.age = state.get("age", 0)
            neuron.gravity = state.get("gravity", 20.0)
            neuron.last_accessed = state.get("last_active", time.time())

    def save(self, persistence_manager):
        """
        Consolidates memory to disk.
        """
        if hasattr(persistence_manager, "consolidate"):
            persistence_manager.consolidate(self.active_tissue)
        else:
            raise NotImplementedError("Persistence manager must have .consolidate()")
