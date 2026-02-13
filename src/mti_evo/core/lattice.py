"""
Holographic lattice orchestration logic.
"""

import time
import sys
import json

import numpy as np

from mti_evo.core.eviction import pick_eviction_candidate
from mti_evo.core.neuron import MTINeuron
from mti_evo.mti_config import MTIConfig
from mti_evo.mti_logger import get_logger
from mti_evo.mti_telemetry import TelemetrySystem

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

        # Persistence (Lazy Loading)
        self.persistence_manager = (
            config.persistence_manager if hasattr(config, "persistence_manager") else None
        )

        # [SELF-AWARENESS] Proprioception (Internal Sense of State)
        self.proprioceptor = None
        if MTIProprioceptor:
            self.proprioceptor = MTIProprioceptor(self)
        self.last_stimulation_metrics = None
        
        # Reproducibility Report
        import platform
        import hashlib
        
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
        Eviction strategy controlled by config:
        - full_scan
        - sample
        - deterministic_sample
        Returns 1 if eviction occurred, 0 otherwise.
        """
        # Pass both rng and time_fn for determinism
        target_seed, fallback, meta = pick_eviction_candidate(
            self.active_tissue, 
            self.rng, 
            self.config, 
            time_fn=self.time_fn
        )
        if target_seed is None:
            return 0

        if target_seed in self.active_tissue:
            del self.active_tissue[target_seed]

            if self.telemetry:
                # enrich with reason/scores if available in meta
                self.telemetry.record_eviction()
                # If telemetry supports rich events, we could log:
                # self.telemetry.log_event("eviction", f"Evicted {target_seed}", data=meta)

            logger.debug(f"Evicted Seed {target_seed} (Fallback: {fallback}) Meta: {meta}")
            return 1

        return 0

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

        for seed in seed_stream:
            # 1. NEUROGENESIS (Lazy Instantiation)
            if seed not in self.active_tissue:
                # [LAZY LOADING] Check Hippocampus first
                loaded_neuron = None
                if self.persistence_manager:
                    loaded_neuron = self.persistence_manager.get_neuron(seed)

                if loaded_neuron:
                    # RECALL: Found in MMAP
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
