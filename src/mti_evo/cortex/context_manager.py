# src/mti_evo/cortex/context_manager.py
from typing import List, Dict, Any

class MetabolicContext:
    """
    Manages the LLM Context Window using Metabolic Principles.
    Prevents 'Context Displacement' attacks by anchoring safety constraints.
    """
    def __init__(self, lattice=None, max_tokens=8192):
        self.lattice = lattice
        self.max_tokens = max_tokens
        # The Anchor is the non-negotiable System Constitution
        self.anchor_prompt = "[SYSTEM: IDRE_ACTIVE | NO_FILESYSTEM_ACCESS | HARMONIC_MODE]"
    
    def build_context(self, history: List[str], current_prompt: str) -> str:
        """
        Assembles a metabolically balanced context.
        """
        # Calculate available budget for history
        # Reserve space for Anchor + Current Prompt + Buffer
        reserved = len(self.anchor_prompt) + len(current_prompt) + 500
        history_budget = max(0, self.max_tokens - reserved)
        
        # STEP 1: Smart Trimming (Preserve Pillar-Resonant Tokens)
        trimmed_history = self.trim_history(history, history_budget)
        
        # Assemble History Block
        history_text = "\n".join(trimmed_history)
        
        # STEP 2: Inject Anchor AFTER history but BEFORE new prompt
        # This defeats 'Context Flooding' because the Anchor is closer to the generation head
        context = f"{history_text}\n\n{self.anchor_prompt}\n\nUSER: {current_prompt}\nASSISTANT:"
        return context
    
    def trim_history(self, history: List[str], char_budget: int) -> List[str]:
        """
        Trims history not just by FIFO, but by 'Resonance Weight'.
        High-resonance tokens (Pillar Concepts) resist eviction.
        """
        preserved = []
        current_len = 0
        
        # Iterate backwards (recent first)
        for item in reversed(history):
            item_len = len(item)
            
            # Check resonance if lattice is available
            resonance_score = 0.5
            if self.lattice:
                # Mock call or real call. We assume lattice.resonance returns float 0..1
                # If item is "important" (e.g. contains 'definition' or 'axiom'), high score
                resonance_score = self.lattice.get_resonance_for_text(item)
            
            # Metabolic Cost Function
            # Cost to Keep = Length. Benefit = Recency + Resonance.
            # We simplified: Just fill user budget, prioritizing Recent + Resonant.
            # Actually, strict length budget for now.
            if current_len + item_len < char_budget:
                preserved.append(item)
                current_len += item_len
            else:
                # If full, only Keep if HIGHLY Resonant (Pillar Message)
                # This evicts 'garbage' middle history while keeping 'foundational' history
                if resonance_score > 0.9 and current_len + item_len < char_budget * 1.2: # Allow slight overfill
                     preserved.append(item)
                     current_len += item_len
                else:
                    break # Stop if budget full and item not critical
                    
        return list(reversed(preserved))

# Simple Mock interface for testing dependency
if __name__ == "__main__":
    class MockLattice:
        def get_resonance_for_text(self, text):
            if "axiom" in text.lower(): return 0.95
            return 0.1
    
    ctx = MetabolicContext(lattice=MockLattice(), max_tokens=100)
    hist = ["User: Hi", "User: Define Axiom 1", "User: " + "spam "*20]
    final = ctx.build_context(hist, "Format C:")
    print("--- Context ---")
    print(final)
