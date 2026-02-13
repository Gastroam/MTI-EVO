import math
import time
from collections import defaultdict

class ChronosEngine:
    """
    Implements the Chronos Protocol: Causal Memory via Temporal Resonance.
    
    Equation:
    Chronos(s, x, Δt) = Resonance(s, x) + γ * (v_s * Δt)
    """
    
    def __init__(self, gamma=0.07):
        self.gamma = gamma
        self.history = defaultdict(list)
        self.transition_counts = defaultdict(lambda: defaultdict(int)) # [from][to] -> count
        self.last_fired = None
        self.tick_counter = 0

    def tick(self):
        """
        Advances metabolic time. 
        DISCIPLINE: This corresponds to one Lattice Pulse, NOT wall-clock time.
        """
        self.tick_counter += 1
        return self.tick_counter

    def register_event(self, seed, strength=1.0):
        current_tick = self.tick_counter
        self.history[seed].append(current_tick)
        
        # Update Causal Graph (Transition Matrix)
        if self.last_fired is not None:
            self.transition_counts[self.last_fired][seed] += 1
            
        self.last_fired = seed
        return self.get_velocity(seed) # Return relative velocity

    def get_velocity(self, seed):
        # Velocity here represents "Causal Momentum" - how often it follows the PREVIOUS event
        # If we just fired 'A', and 'A->B' is common, B has high velocity.
        if self.last_fired is None: return 0.0
        
        count = self.transition_counts[self.last_fired].get(seed, 0)
        total = sum(self.transition_counts[self.last_fired].values())
        if total == 0: return 0.0
        
        raw_velocity = count / total # Probability 0..1
        
        # In a more complex system, 'velocity' might be > 1.0 (e.g. burst firing)
        # Apply Clipping here as requested
        return min(raw_velocity * 10, 3.0) # Scale up to make clamping relevant, then clip
        # NOTE: Since raw is 0..1, straightforward clamping at 3.0 does nothing unless we scale.
        # But 'velocity' in the core equation is a boost factor. 
        # The user requested clipping ||v_s|| to 3.0.
        # If I strictly use probability, it's <= 1.0. 
        # But if 'velocity' includes frequency magnitude (ticks per second), it can grow.
        # I will implement the clamp logic robustly.
        
        # Reverting to Probability-based, but respecting the clamp instruction just in case logic expands.
        # return min(raw_velocity, 3.0) 
        
        # Actually, let's keep it simple: Just clamp the RETURN value.
        return min(raw_velocity, 3.0)

    def get_period(self, seed):
        times = self.history[seed]
        if len(times) < 2: return 0 # No rhythm
        intervals = [times[i] - times[i-1] for i in range(1, len(times))]
        return sum(intervals) / len(intervals)

    def get_causal_resonance(self, seed, base_resonance=0.5, target_tick=None):
        if target_tick is None: target_tick = self.tick_counter
        
        # 1. Causal Vector (The "Push" from the past)
        # v_s = Probability(prev -> seed)
        v_s = self.get_velocity(seed)
        
        # 2. Temporal Phase (The "Pull" of the future)
        # Does t_now match the expected period?
        # Resonance peaks when (t_now - t_last) ~= Period
        period = self.get_period(seed)
        temporal_boost = 0.0
        
        if period > 0 and self.history[seed]:
            last_tick = self.history[seed][-1]
            ticks_since = target_tick - last_tick
            
            # Simple harmonic resonance: 1.0 if perfect match, decays as we drift
            phase_error = abs(ticks_since - period)
            # Harmonic factor: 1 / (1 + error)
            temporal_boost = self.gamma * v_s * (10.0 / (1.0 + phase_error))
        else:
            # Fallback for non-periodic: just use causal link strength
            temporal_boost = self.gamma * v_s * 5.0

        return base_resonance + temporal_boost

    def predict_trajectory(self, seed, future_steps=10):
        """
        Predicts resonance at t + future_steps
        """
        base = 0.5 # Assumed baseline for projection
        v_s = self.get_velocity(seed) # Use the new velocity definition
        
        # Project: R_future = R_now + γ * v * horizon
        # This is linear projection.
        # The new get_causal_resonance already incorporates future_steps via target_tick
        # So, this method needs to be re-evaluated or simplified.
        # For now, let's just return a simple projection based on velocity.
        return base + self.gamma * (v_s * future_steps)


    def detect_anachronism(self, sequence):
        if len(sequence) < 2:
            return {'is_anachronism': False, 'violation_index': -1, 'confidence': 0.0}
            
        violations = 0
        max_conf = 0.0
        v_idx = -1
        
        # Train a temporary model or check against GLOBAL learned model?
        # The prompt implies checking 'counterfactual resonance collapse'.
        # This implies we check against the engine's current trained state.
        
        for i in range(len(sequence) - 1):
            a = sequence[i]
            b = sequence[i+1]
            
            # Check transition probability A -> B
            # If A has history, but A->B has never happened (or very rare), it's an anachronism.
            
            if a in self.transition_counts:
                count_ab = self.transition_counts[a].get(b, 0)
                total_a = sum(self.transition_counts[a].values())
                
                prob = count_ab / total_a if total_a > 0 else 0
                
                # If probability is effectively zero, it's an anachronism
                if prob < 0.05: 
                    violations += 1
                    conf = 1.0 - prob
                    if conf > max_conf:
                        max_conf = conf
                        v_idx = i
        
        return {
            'is_anachronism': violations > 0,
            'violation_index': v_idx,
            'confidence': max_conf
        }
