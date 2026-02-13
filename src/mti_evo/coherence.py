"""
MTI-EVO Coherence Engine (Phase 6)
==================================
Handles Epistemic Governance and Network Health.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CoherenceReport:
    seed: int
    health: float
    metabolic_score: float
    cluster_score: float
    tension_penalty: float
    verdict: str

class CoherenceEngine:
    """
    Monitors the internal consistency of the Holographic Lattice.
    Calculates Health Index based on Metabolism, Critics, and Neighbors.
    """
    
    def __init__(self, cortex, llm_engine=None):
        self.cortex = cortex
        self.llm = llm_engine # Needed for embedding generation
        self.last_scan = 0
        
    def calculate_health(self, seed: int) -> float:
        """
        Calculate Epistemic Health Index (H) for a given seed.
        H = w_m * Metabolic + w_c * Critic + w_h * Cluster - Penalty
        """
        if seed not in self.cortex.active_tissue:
            return 0.0
            
        neuron = self.cortex.active_tissue[seed]
        config = self.cortex.config
        
        # 1. Normalized Metabolic Score (0.0 - 1.0)
        # We normalize against the weight cap (80.0)
        decay_rate = getattr(config, 'passive_decay_rate', 0.00001)
        delta = time.time() - neuron.last_accessed
        raw_score = np.mean(np.abs(neuron.weights)) * np.exp(-decay_rate * delta)
        s_metabolic = min(raw_score / 20.0, 1.0) # 20.0 is "Healthy High"
        
        # 2. Critic Score (0.0 - 1.0)
        # Default to 0.8 (Benefit of Doubt) if no history
        if not neuron.critic_history:
            mu_critic = 0.8 
        else:
            mu_critic = sum(neuron.critic_history) / len(neuron.critic_history)
            
        # 3. Cluster Consistency (0.0 - 1.0)
        # Placeholder: In full implementation, this checks neighbor vectors.
        # For Alpha v2.6, we assume self-consistency if Critic is happy.
        c_cluster = mu_critic 
        
        # 4. Tension Penalty
        p_tension = 0.0 # TODO: Implement contradiction detection
        
        # Weighted Sum
        # Weights: Metabolic=0.4, Critic=0.3, Cluster=0.3
        h_index = (0.4 * s_metabolic) + (0.3 * mu_critic) + (0.3 * c_cluster) - p_tension
        
        # Clamp
        h_index = max(0.0, min(1.0, h_index))
        
        # Update Neuron State
        neuron.health_index = h_index
        
        return h_index

    def run_semantic_sweep(self):
        """
        [ASYNC] Generates embeddings for neurons missing them.
        """
        if not self.llm:
            logger.warning("CoherenceEngine: No LLM attached. Cannot generate embeddings.")
            return

        count = 0
        for seed, neuron in self.cortex.active_tissue.items():
            if neuron.semantic_vector is None:
                # We need a text representation. 
                # Ideally, the neuron stores its 'Definition' or 'Origin'.
                # Since MTI v1 stores vectors, we might need to reverse-engineer or wait for input.
                # For now, we skip if no text.
                pass
                
        return count

    def generate_report(self, top_n=10) -> List[CoherenceReport]:
        """Scan top neurons and report health."""
        reports = []
        # Sort by raw weight
        top_seeds = sorted(
            self.cortex.active_tissue.keys(), 
            key=lambda k: np.mean(np.abs(self.cortex.active_tissue[k].weights)), 
            reverse=True
        )[:top_n]
        
        for seed in top_seeds:
            h = self.calculate_health(seed)
            verdict = "HEALTHY"
            if h < 0.4: verdict = "SICK (Rotting)"
            elif h < 0.7: verdict = "AT RISK"
            
            reports.append(CoherenceReport(
                seed=seed,
                health=h,
                metabolic_score=0.0, # Filled inside calc but not returned... simplified for now
                cluster_score=0.0,
                tension_penalty=0.0,
                verdict=verdict
            ))
            
        return reports
