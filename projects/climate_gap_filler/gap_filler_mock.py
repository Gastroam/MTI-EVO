import random
import math

class ClimateDataMock:
    """Generates synthetic sensor data with gaps."""
    def __init__(self):
        self.data = []
        
    def generate_diurnal_cycle(self, hours=24):
        """Simulates 24h temp cycle (Sine wave + Noise)."""
        base_temp = 20.0
        amplitude = 10.0
        self.data = []
        for h in range(hours):
            # Peak at 14:00 (h=14)
            temp = base_temp + amplitude * math.sin((h - 8) * math.pi / 12)
            noise = random.uniform(-0.5, 0.5)
            self.data.append(round(temp + noise, 2))
        return self.data

    def inject_gap(self, start, length):
        """Removes data to simulate sensor failure."""
        for i in range(start, start + length):
            self.data[i] = None
        return self.data

class PhysicsPillar:
    """Enforces Physical Laws (The Critic)."""
    def __init__(self, max_rate_change=3.0):
        self.max_rate_change = max_rate_change # Max 3 degrees per hour allowed
        
    def validate_segment(self, segment, previous_val, next_val):
        """Checks if a filled segment obeys thermal inertia."""
        # Check boundary continuity
        if abs(segment[0] - previous_val) > self.max_rate_change:
            return False, "Violation: Initial Jump too high"
        
        if abs(segment[-1] - next_val) > self.max_rate_change:
            return False, "Violation: Final Jump too high"
            
        # Check internal continuity
        for i in range(len(segment) - 1):
            delta = abs(segment[i+1] - segment[i])
            if delta > self.max_rate_change:
                return False, f"Violation: Internal Spike ({delta:.2f} > {self.max_rate_change})"
                
        return True, "Valid"

class HallucinationGhost:
    """Generates plausible patterns (The Dreamer)."""
    def dream_candidates(self, start_val, end_val, steps, num_candidates=5):
        candidates = []
        for _ in range(num_candidates):
            candidate = []
            current = start_val
            # Simple Linear Interpolation + Random Walk
            step_size = (end_val - start_val) / (steps + 1)
            
            for i in range(steps):
                # Add randomness (hallucination)
                drift = random.uniform(-2, 2) 
                # Some candidates are "Linear" (low drift), some "Chaotic" (high drift)
                val = start_val + (step_size * (i+1)) + drift
                candidate.append(round(val, 2))
            candidates.append(candidate)
        return candidates

class GapFillerEngine:
    """Orchestrates the filling."""
    def __init__(self):
        self.physics = PhysicsPillar(max_rate_change=2.5) # Strict physics
        self.ghost = HallucinationGhost()
        
    def fill(self, data):
        # 1. Detect Gap
        gap_start = -1
        gap_len = 0
        
        for i, val in enumerate(data):
            if val is None:
                if gap_start == -1: gap_start = i
                gap_len += 1
            elif gap_start != -1:
                # Gap ended
                self._process_gap(data, gap_start, gap_len)
                gap_start = -1
                gap_len = 0
                
        return data

    def _process_gap(self, data, start, length):
        prev_val = data[start-1]
        next_val = data[start+length]
        
        print(f"🔧 Filling Gap at Index {start} (Len: {length})...")
        print(f"   Context: {prev_val} -> [?] -> {next_val}")
        
        # 2. Ghost Dreams Candidates
        candidates = self.ghost.dream_candidates(prev_val, next_val, length)
        
        best_candidate = None
        min_error = float('inf')
        
        # 3. Pillar Validates
        for i, cand in enumerate(candidates):
            is_valid, reason = self.physics.validate_segment(cand, prev_val, next_val)
            print(f"   Candidate {i}: {cand} -> {reason}")
            
            if is_valid:
                # Calculate "Smoothness Energy" (Simulated Bridge Matrix)
                # Ideally prefer smoother curves
                energy = sum(abs(cand[j+1] - cand[j]) for j in range(len(cand)-1))
                if energy < min_error:
                    min_error = energy
                    best_candidate = cand
        
        # 4. Integrate Best
        if best_candidate:
            print(f"✅ Selected Candidate: {best_candidate} (Energy: {min_error:.2f})")
            for i in range(length):
                data[start+i] = best_candidate[i]
        else:
            print("❌ All candidates rejected by Physics! Using Linear Fallback.")
            # Fallback linear
            step = (next_val - prev_val) / (length + 1)
            for i in range(length):
                 data[start+i] = round(prev_val + step * (i+1), 2)

if __name__ == "__main__":
    print("--- Project 2: Climate Data Gap-Filler Mockup ---")
    mock = ClimateDataMock()
    data = mock.generate_diurnal_cycle()
    print("Original (Subset):", data[8:18])
    
    # Create Hole
    data = mock.inject_gap(12, 3) # Gap at noon
    print("Corrupted:       ", data[8:18])
    
    # Fill
    engine = GapFillerEngine()
    filled_data = engine.fill(data)
    
    print("Restored:        ", filled_data[8:18])
