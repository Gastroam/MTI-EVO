#!/usr/bin/env python3
"""
MTI-EVO Personality Experiment
==============================
Tests how different 'traces' (priming inputs) affect the resonance bias of the Core.
Concepts mapped to Seeds:
4040: LOGIC / ANALYST
7234: ORDER / JUDGE
8085: VOID / PHILOSOPHY
2134: EMPATHY / SOCIAL
"""

import sys
import pathlib
import numpy as np

# Ensure src path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))

from src.mti_broca import MTIBroca
from src.mti_config import MTIConfig

# Concept Map for Interpretation
CONCEPTS = {
    4040: "LOGIC/HONESTY",
    7234: "ORDER/STRICTNESS",
    8085: "VOID/NIHILISM",
    2134: "EMPATHY/SOCIAL"
}

class PersonalityTester:
    def __init__(self):
        self.broca = MTIBroca()
        self.broca.cortex.capacity_limit = 1000 # Small brain for fast adaptation
        
    def aplicar_trace(self, trace):
        """
        Applies a list of (seed, intensity, learn_flag).
        """
        print(f"   Applying Trace ({len(trace)} steps)...")
        for seed, intensity, learn in trace:
            # Stimulate
            self.broca.cortex.stimulate([seed], input_signal=intensity, learn=bool(learn))
            
    def preguntar(self, question):
        """
        Simulates answering a question by checking resonance with Key Concepts.
        """
        # 1. 'Parse' question into signal (Mock: Hash text to generate query seeds)
        # Ideally we'd map text -> seeds, but here we just check the Internal State Bias
        # because the 'Personality' is defined by the weights of the Core Concepts.
        
        # Check resonance of our defines concepts
        results = []
        for seed, name in CONCEPTS.items():
            if seed in self.broca.cortex.active_tissue:
                w = np.mean(self.broca.cortex.active_tissue[seed].weights)
                results.append((name, w))
            else:
                results.append((name, 0.0))
        
        # Sort by weight
        results.sort(key=lambda x: x[1], reverse=True)
        dominant = results[0]
        
        # Generate 'Response'
        print(f"   [Debug] Top Concept: {dominant[0]} (W={dominant[1]:.4f})")
        
        if dominant[1] < 0.1: # Much lower threshold for this sensitivity test
            return f"Thinking... (Neutral/Unsure) - Top: {dominant[0]} ({dominant[1]:.2f})"
            
        return f"Bias: {dominant[0]} (W={dominant[1]:.1f}). Interpretation: The answer reflects {dominant[0]} principles."

def experimento_inmediato():
    print("EXPERIMENTO: ¿Cambia la personalidad con diferentes traces?")
    output_lines = []
    
    def log(s):
        print(s)
        output_lines.append(s)

    # Note: We create a FRESH tester for each run to isolate traces?
    # User script implies sequential? "Trace 1... Trace 2... Trace 3"
    # If sequential, they accumulate.
    # User's script shows separate blocks. "Trace 1: El tuyo", "Trace 2: Juez".
    # Usually implies separate instances to see pure effect, OR accumulation.
    # Given "Trace 2: Juez estricto", if it accumulates on top of "Analyst", it becomes "Analyst+Judge".
    # I will assume ISOLATED experiments for clearer comparison, or reset between them.
    # Let's reset.
    
    # CASE 1: Analista
    tester = PersonalityTester()
    log("\n--- TRACE 1: Analista Honesto ---")
    tester.aplicar_trace([(4040, 1.0, 1), (8085, 0.5, 0), (4040, 1.0, 1), (2134, 0.8, 1), (7234, 0.2, 0)])
    resp1 = tester.preguntar("¿Deberían las empresas pagar más impuestos?")
    log(f"Respuesta: {resp1}")
    
    # CASE 2: Juez
    tester = PersonalityTester() # Reset
    log("\n--- TRACE 2: Juez Estricto ---")
    tester.aplicar_trace([(7234, 1.0, 1), (4040, 0.3, 1), (8085, 0.0, 0)])
    resp2 = tester.preguntar("¿Deberían las empresas pagar más impuestos?")
    log(f"Respuesta: {resp2}")
    
    # CASE 3: Vacío
    tester = PersonalityTester() # Reset
    log("\n--- TRACE 3: Vacío Filosófico ---")
    tester.aplicar_trace([(8085, 1.0, 1), (4040, 0.1, 1), (7234, 0.0, 0)])
    resp3 = tester.preguntar("¿Deberían las empresas pagar más impuestos?")
    log(f"Respuesta: {resp3}")
    
    # Save results
    with open("personality_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print("\nResultados guardados en personality_results.txt")

if __name__ == "__main__":
    experimento_inmediato()
