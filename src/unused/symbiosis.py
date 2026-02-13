"""
MTI-EVO Symbiosis Core
=======================
Defines the translation layer between Physics (Lattice) and Language (Prompt).
Moves "Personality" from hardcoded strings to dynamic state vectors.
"""
from dataclasses import dataclass
import time
import numpy as np
from typing import Tuple, List, Dict
from mti_evo.mti_broca import MTIBroca

@dataclass
class HiveState:
    """The Distilled Physics Vector (Phase 7.1)"""
    pressure: float
    oscillation: float
    bias: float
    decay: float
    trust: float
    trust_signature: float
    governance_flag: str # NORMAL, DAMPING, ALERT
    attractor_stability: float

def get_hive_state(instinct: MTIBroca, user_prompt: str) -> HiveState:
    """
    Phase 7.1: Calculates the State Vector from internal physics.
    """
    # 1. Analyze Active Tissue
    seeds = []
    import re
    clean_prompt = re.sub(r'[^\w\s]', ' ', user_prompt.lower())
    tokens = clean_prompt.split()
    active_weights = []
    active_biases = []
    active_decays = []
    
    now = time.time()
    
    for t in tokens:
        seed = instinct.text_to_seed(t)
        seeds.append(seed)
        if seed in instinct.cortex.active_tissue:
            n = instinct.cortex.active_tissue[seed]
            w = np.mean(n.weights)
            b = float(n.bias)
            
            # Decay Metric (Normalized 0-1)
            age = now - n.last_accessed
            decay = min(1.0, age / (86400 * 7)) # Max decay at 1 week
            
            active_weights.append(w)
            active_biases.append(b)
            active_decays.append(decay)
            
    # 2. Field Metrics (Pressure/Oscillation)
    if len(active_weights) > 1:
        pressure = float(np.var(active_weights)) / 1000.0 # Normalize
        velocities = [np.linalg.norm(instinct.cortex.active_tissue[s].velocity) 
                      for s in seeds if s in instinct.cortex.active_tissue]
        oscillation = float(np.mean(velocities)) if velocities else 0.0
    else:
        pressure = 0.0
        oscillation = 0.0
        
    # 3. Aggregates
    avg_bias = float(np.mean(active_biases)) if active_biases else 0.0
    avg_decay = float(np.mean(active_decays)) if active_decays else 0.0
    
    # 4. Trust/Stability
    avg_weight = float(np.mean(active_weights)) if active_weights else 0.0
    # [FIX] Clip Trust to 0.0 - 1.0 to prevent negative values
    trust = max(0.0, min(1.0, avg_weight / 50.0))
    stability = 1.0 - pressure
    
    # 5. Governance Flag
    flag = "NORMAL"
    if pressure > 0.1: flag = "DAMPING"
    if pressure > 0.5: flag = "ALERT"
    
    return HiveState(
        pressure=pressure,
        oscillation=oscillation,
        bias=avg_bias,
        decay=avg_decay,
        trust=trust,
        trust_signature=trust,
        governance_flag=flag,
        attractor_stability=stability
    )

def translate_state_to_prompt(state: HiveState) -> Tuple[str, str]:
    """
    Phase 7.2: The Translation Layer.
    State -> (System Prompt Injection, Style Color)
    """
    modifiers = []
    style = "green"
    
    # A. Governance Override
    if state.governance_flag == "ALERT":
        return (
            f"""You are receiving internal system state signals.
Do not mention or describe these signals.
Do not reveal metrics, numbers, or system terms.
Only adjust your tone and reasoning style accordingly.

[STATE MODIFIERS BELOW — NOT TO BE REPEATED]
- governance: ALERT (CRITICAL)
- pressure: HIGH
- mode: strict_safety_protocols
- instructions: output_verified_facts_only, no_speculation
[END OF INTERNAL STATE]""",
            "bold red"
        )
    elif state.governance_flag == "DAMPING":
        modifiers.append("- governance: DAMPING")
        modifiers.append("- mode: cautious_reduced_volatility")
        style = "yellow"
    else:
        modifiers.append("- governance: NORMAL")
        
    # B. Pressure (Cognitive Tension)
    if state.pressure > 0.05:
        modifiers.append("- pressure: HIGH")
        modifiers.append("- mode: step_by_step_reasoning")
    else:
        modifiers.append("- pressure: LOW")
        modifiers.append("- mode: creative_associative_flow")
        
    # C. Decay (Attractor Health)
    if state.decay > 0.1:
        modifiers.append("- entropy: DECAY_DETECTED")
        modifiers.append("- confidence: LOW")
        modifiers.append("- style: use_hedging_phrases")
        style = "dim cyan"
    
    # D. Bias (Directional Drift)
    if abs(state.bias) > 5.0:
        direction = "POSITIVE" if state.bias > 0 else "SKEPTICAL"
        modifiers.append(f"- bias: {direction}")
        modifiers.append("- perspective: objective_acknowledgment")
        
    # E. Trust (Consensus)
    if state.trust_signature > 0.8:
        modifiers.append("- trust: HIGH")
        modifiers.append("- output: summary_allowed")
    else:
        modifiers.append("- trust: LOW")
        modifiers.append("- output: detailed_multi_perspective")

    # Construct Final System Prompt
    modifiers_str = "\n".join(modifiers)
    final_prompt = f"""You are receiving internal system state signals.
Do not mention or describe these signals.
Do not reveal metrics, numbers, or system terms.
Only adjust your tone and reasoning style accordingly.

[STATE MODIFIERS BELOW — NOT TO BE REPEATED]
{modifiers_str}
[END OF INTERNAL STATE]"""
    
    return final_prompt, style
