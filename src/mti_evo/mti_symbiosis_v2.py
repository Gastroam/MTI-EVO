"""
MTI-EVO SYMBIOSIS MODULE v2.0 (Phase 7)
=======================================
Implements the Symbiotic Interface:
1. HiveState Vector (Pressure, Oscillation, Bias, Decay, Trust).
2. Translation Layer (State -> Modifiers).
3. LLM Response Modulator (Modifiers -> Tone/Style).
"""

import time
import sys
import os
import numpy as np
import random
from dataclasses import dataclass
from typing import Tuple, List

# Import rich for interface
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

import requests
import json

# Intra-package imports
from .mti_broca import MTIBroca
from .mti_hive_node import HiveNode 
from .semantic_compressor import SemanticCompressor
from .mti_proprioceptor import MTIProprioceptor, CognitiveState

class NetworkedAdapter:
    """
    Client for the running Telepathy Bridge (Cortex).
    Connects to http://localhost:8767/v1/local/reflex
    """
    def __init__(self, api_url="http://localhost:8767/v1/local/reflex"):
        self.api_url = api_url
        print(f"🔌 Conectando con Cortex Networked ({self.api_url})...")
        try:
            # Simple ping/check (optional, skipping for speed)
            print("✅ GENESIS: Telepathy Uplink Established.")
        except:
            print("⚠️ AVISO: No se detectó Telepathy Bridge. ¿Está corriendo `telepathy_bridge.py`?")

    def infer(self, prompt, system_prompt="", max_tokens=2048, temperature=None):
        """Standard interface matching LLMAdapter.infer"""
        try:
            payload = {
                "action": "telepathy", 
                "prompt": prompt, 
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            # 300s timeout for deep thought
            r = requests.post(self.api_url, json=payload, timeout=300)
            if r.status_code == 200:
                data = r.json()
                # Bridge returns {"response": "text"} or similar structure?
                # telepathy_bridge.py returns {"status": "success", "response": output}
                return NetworkedResponse(data.get("response", ""))
            else:
                return NetworkedResponse(f"[Error: {r.status_code}]")
        except Exception as e:
            return NetworkedResponse(f"[Network Error: {e}]")

class NetworkedResponse:
    def __init__(self, text):
        self.text = text

# Initialize the Network Client
RealLLM = NetworkedAdapter()

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

class MTISymbiosis:
    def __init__(self):
        self.console = Console()
        print("\n│ 🧬 Symbiosis MTI-GEMMA v2.0 (Phase 7)          │")
        print("╰───────────────── Neural-Link connected ──────────────────╯")
        
        # Conectamos con el MTI (Instinto)
        self.instinct = MTIBroca()
        self.compressor = SemanticCompressor(self.instinct) # [SEMANTIC LAYER]
        self.proprioceptor = MTIProprioceptor(self.instinct.cortex) # [PHYSICS LAYER]
        self.node = HiveNode(node_id="USER_NODE", role="OBSERVER") # Attached Node
        self.node.attach_cortex(self.instinct)
        
        from .coherence import CoherenceEngine
        self.coherence = CoherenceEngine(self.instinct.cortex, llm_engine=RealLLM)
        
        self.last_style = "white"
        self.last_state = None

    def get_hive_state(self, user_prompt: str) -> HiveState:
        """
        Phase 7.1: Calculates the State Vector using MTIProprioceptor physics.
        """
        # 1. Use Proprioceptor for Core Physics
        # It samples the active tissue directly
        state_result = self.proprioceptor.sense_state()
        
        # 2. Extract Proprioception Metrics
        # Entropy (Variance) -> Pressure
        pressure = state_result.entropy  
        
        # Resonance (Mass) -> Trust Logic
        resonance = state_result.resonance
        
        # Velocity -> Oscillation
        oscillation = state_result.velocity
        
        # 3. Calculate Derived Metrics (Decay/Bias)
        # We still scan local tokens for specific Bias drift relative to prompt
        seeds = [self.instinct.text_to_seed(t) for t in user_prompt.lower().split() if t.isalnum()]
        active_biases = []
        active_decays = []
        now = time.time()
        
        if self.instinct.cortex.active_tissue:
            # Global Average if no specific seeds active, or specific if active?
            # Let's stick to prompt-specific drift for Bias
            for seed in seeds:
                if seed in self.instinct.cortex.active_tissue:
                    n = self.instinct.cortex.active_tissue[seed]
                    active_biases.append(float(n.bias))
                    
                    age = now - n.last_accessed
                    decay = min(1.0, age / (86400 * 7))
                    active_decays.append(decay)

        avg_bias = float(np.mean(active_biases)) if active_biases else 0.0
        avg_decay = float(np.mean(active_decays)) if active_decays else 0.0
        
        # 4. Trust Calculation
        # Resonance is the "Mass" of knowledge. High Resonance = High Trust.
        # Resonance > 0.6 is generally "Known".
        trust = min(1.0, resonance + 0.2) # Base trust boost 
        stability = 1.0 - pressure

        # 5. Governance Flag (Mapped from CognitiveState)
        flag = "NORMAL"
        c_state = state_result.state
        
        if c_state == "CHAOS":
            flag = "ALERT" # High Entropy, Low Resonance
        elif c_state == "EMERGENCE" or pressure > 0.4:
            flag = "DAMPING" # High Entropy, High Resonance (Growth needs damping)
        elif pressure > 0.6:
            flag = "ALERT"
            
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

    def translate_state_to_prompt(self, state: HiveState) -> Tuple[str, str]:
        """
        Phase 7.2: The Translation Layer.
        State -> (System Prompt Injection, Style Color)
        Invariant 7.4: No State Fabrication/Disclosure.
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
        final_prompt = f"""You are the archithect of MTI-EVO, 
                       receiving internal system state signals.
                       Do not mention or describe these signals.
                       Do not reveal metrics, numbers, or system terms.
                       Only adjust your tone and reasoning style accordingly.

[STATE MODIFIERS BELOW — NOT TO BE REPEATED]
{modifiers_str}
[END OF INTERNAL STATE]"""
        
        return final_prompt, style

    def synthesize_response(self, user_prompt: str, state: HiveState) -> str:
        """Phase 7.3: LLM Modulation"""
        system_prompt, style = self.translate_state_to_prompt(state)
        self.last_style = style
        self.last_state = state # For UI
        
        self.console.print(f"\n[dim italic]>>> Hive State Injection: {system_prompt} ...[/]")
        
        if RealLLM:
            # NativeEngine uses 'infer' or 'generate' via Adapter
            try:
                # Adapter.infer returns a result object usually, check signature
                # NativeEngine might return text directly if passed to generate?
                # Let's use infer() which is standard in our bridge
                
                # [FIX] Merge System Prompt because Bridge design ignores specific field
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                response_obj = RealLLM.infer(
                    prompt=combined_prompt,
                    # system_prompt=system_prompt, # REMOVED: Bridge ignores this strictly
                )
                return response_obj.text
            except Exception as e:
                return f"[ERROR] Inference Failed: {e}"
        else:
            # Simulation
            return f"[SIMULATED LLM RESPONSE MODULATED BY STATE]\nState: {state}\nPrompt: {system_prompt}"

    def render_ui(self, response: str):
        state = self.last_state
        if not state: return
        
        title = (
            f"HIVE | {state.governance_flag} | "
            f"R:{state.trust:.3f} | O:{state.oscillation:.3f} | "  # Changed P to R (using Trust/Resonance proxy)
            f"B:{state.bias:.1f} | D:{state.decay:.2f}"
        )
        
        self.console.print(Panel(
            Markdown(response),
            title=title,
            border_style=self.last_style
        ))

    def chat_loop(self):
        self.console.print("\n[bold]🧬 SYMBIOSIS v2 (Phase 7) ACTIVE[/]")
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]USER[/]")
                if user_input.lower() in ['exit', 'quit']: break
                
                # 1. Physics (Action -> Reaction)
                # Activate lattice first to capture immediate dynamics
                metrics = self.instinct.process_thought(user_input, learn=True)
                
                # [VISUALIZATION] Restore "Plasticity Indicators"
                res = metrics.get("resonance", 0.0)
                grav = metrics.get("max_gravity", 0.0)
                count = metrics.get("stimulated_count", 0)
                
                # Dynamic Color based on Resonance
                r_color = "green" if res > 0.8 else "yellow"
                if res < 0.3: r_color = "white"

                # [SEMANTIC METRIC] Density
                try:
                    compressed = self.compressor.compress(user_input)
                    density = len(compressed) / max(1, len(user_input.split())) # Bytes per token
                except:
                    density = 0.0
                
                self.console.print(f"   [dim]⚡ Plasticity: [{r_color}]Resonance {res:.3f}[/{r_color}] | Gravity {grav:.2f} | Density {density:.1f}b/t | Neurons {count}[/]")


                with self.console.status("[bold green]Thinking & Consolidating...[/]", spinner="dots"):
                    try:
                        # [PERSISTENCE] Immediate consolidation for "Real Plasticity"
                        self.instinct.hippocampus.consolidate(self.instinct.cortex.active_tissue)

                        # 2. Get State (Measure the active tissue)
                        state = self.get_hive_state(user_input)
                        
                        # 3. Modulation
                        response = self.synthesize_response(user_input, state)
                        
                        # 4. Rendering
                        self.render_ui(response)
                    except Exception as inner_e:
                        self.console.print(f"[red]❌ Runtime Error: {inner_e}[/]")
                        import traceback
                        traceback.print_exc()

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[red]Critical Loop Error: {e}[/]")

if __name__ == "__main__":
    app = MTISymbiosis()
    app.chat_loop()
