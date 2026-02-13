
import numpy as np
import enum
import time
from types import SimpleNamespace

class CognitiveState(enum.Enum):
    FLOW = "FLOW"         # High Resonance, Low Entropy (Mastery)
    EMERGENCE = "EMERGENCE" # High Resonance, High Entropy (Growth)
    LEARNING = "LEARNING" # Mid Resonance, High Entropy (Adaptation)
    CHAOS = "CHAOS"       # Low Resonance, High Entropy (Disintegration)
    IDLE = "IDLE"         # Startup/Inactive

class StateResult(dict):
    """
    Hybrid Dictionary/Object to support both API (JSON) and Test (Attribute) access.
    """
    def __init__(self, *args, **kwargs):
        super(StateResult, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __setitem__(self, key, value):
        # Allow updating enum to string for JSON output
        if key == "state" and isinstance(value, enum.Enum):
             value = value.value
        super(StateResult, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def update(self, *args, **kwargs):
         if args:
              if len(args) > 1:
                   raise TypeError("update expected at most 1 arguments, "
                                   "got %d" % len(args))
              other = dict(args[0])
              for key in other:
                   self[key] = other[key]
         for key in kwargs:
              self[key] = kwargs[key]

class MTIProprioceptor:
    """
    Substrate Introspection Module.
    Detects the 'Cognitive State' of the lattice based on Resonance Phenomenology.
    """
    def __init__(self, cortex=None):
        self.cortex = cortex
        self.history = []
        self.last_state = CognitiveState.IDLE
        self.transition_time = time.time()
        
    def sense_state(self, resonance=None, entropy=None):
        """
        Samples the current lattice metrics and determines state.
        Can be overridden with manual metrics for testing.
        Returns metrics dict with 'state' and 'governor_active'.
        """
        metrics = {
             "resonance": 0.0,
             "entropy": 0.0,
             "velocity": 0.0,
             "state": CognitiveState.IDLE.value,
             "governor_active": False
        }
        
        # 1. Measure or Use Override
        if resonance is not None and entropy is not None:
             # Manual Test Mode
             avg_resonance = resonance
             avg_entropy = entropy
        elif self.cortex and self.cortex.active_tissue:
            # Measure Resonance Mass (Avg Weight Magnitude)
            total_mass = 0.0
            active_count = 0
            variances = []
            
            # Sample a subset for performance if lattice is huge
            sample_seeds = list(self.cortex.active_tissue.keys())
            if len(sample_seeds) > 100:
                import random
                sample_seeds = random.sample(sample_seeds, 100)
                
            for s in sample_seeds:
                n = self.cortex.active_tissue[s]
                w_mean = n.weights.mean()
                w_var = np.std(n.weights)
                
                total_mass += abs(w_mean)
                variances.append(w_var)
                active_count += 1
                
            avg_resonance = total_mass / max(1, active_count)
            avg_entropy = np.mean(variances) if variances else 0.0
        else:
             return StateResult(metrics) # IDLE
             
        # 2. Velocity (Simulated for now, usually needs Chronos delta)
        velocity = 0.0
        if self.history:
            last_res = self.history[-1]['resonance']
            velocity = abs(avg_resonance - last_res)
            
        # 3. State Classification
        new_state = CognitiveState.LEARNING # Default
        governor_active = False
        
        # THRESHOLDS (Phase 60 Calibration)
        # EMERGENCE: High Stability AND High Complexity.
        # This is the "Safe Zone" for Ontological Shock.
        if avg_resonance > 0.6 and avg_entropy > 0.3:
            new_state = CognitiveState.EMERGENCE
            governor_active = False # ALLOW growth
            
        elif avg_resonance > 0.7 and avg_entropy < 0.3:
            new_state = CognitiveState.FLOW
            governor_active = False
            
        elif avg_resonance < 0.4:
            new_state = CognitiveState.CHAOS # Was VOID
            governor_active = True # LOCKDOWN
            
        else:
            new_state = CognitiveState.LEARNING
            governor_active = False
            
        # 4. Update History
        result = StateResult({
            "state": new_state.value, # Dict needs value
            "resonance": float(avg_resonance),
            "entropy": float(avg_entropy),
            "velocity": float(velocity),
            "governor_active": governor_active
        })
        
        self.history.append(result)
        if len(self.history) > 50: 
            self.history.pop(0)
            
        self.last_state = new_state
        self.last_state = new_state
        return result

    def self_tune(self, config):
        """
        Meta-Plasticity (Phase 67):
        The Governor autonomously tunes system physics based on Cognitive State.
        """
        if self.last_state == CognitiveState.EMERGENCE:
            # High Value State: Protect fragile insights
            # Lower entropy decay to allow crystallization
            if config.decay_rate > 0.1:
                config.decay_rate = 0.08
                # In a real system, we might log this meta-action
                return True, "Decay rate lowered to 0.08 (EMERGENCE Protection)"
                
        elif self.last_state == CognitiveState.CHAOS:
             # Crisis State: Stabilize
             # Increase decay to prune noise, or lock input (handled in sense_state return)
             pass
             
        elif self.last_state == CognitiveState.FLOW:
             # Ideal State: Restore standard physics if needed
             if config.decay_rate < 0.15:
                 config.decay_rate = 0.15 # Restore baseline
                 return True, "Decay rate restored to 0.15 (FLOW)"
                 

# Alias for cleaner external API
IntrospectionEngine = MTIProprioceptor
