
import json
import os
import time
import numpy as np
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field, asdict

from mti_evo.core.logger import get_logger

logger = get_logger("SemanticProbes")

@dataclass
class ProbePair:
    seed_a: int
    seed_b: int
    expect: str = "high"  # "high", "low"
    min_cos: float = -1.0
    max_cos: float = 1.0
    description: str = ""

@dataclass
class ProbeResult:
    pair: ProbePair
    cosine_sim: float
    passed: bool
    vector_source: str # "semantic_vector" or "weights"
    timestamp: float = field(default_factory=time.time)

class SemanticProbeRunner:
    """
    Executes semantic probes to measure latent space health.
    """
    def __init__(self, lattice, config=None):
        self.lattice = lattice
        self.config = config
        self.probes: List[ProbePair] = []
        
    def load_probes(self, path: str):
        """
        Load probes from JSON file.
        Format: {"pairs": [{"seed_a": 1, "seed_b": 2, "expect": "high", ...}]}
        """
        if not os.path.exists(path):
            logger.warning(f"Probe file not found: {path}")
            return
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            runs = data.get("pairs", [])
            for p in runs:
                self.probes.append(ProbePair(
                    seed_a=p['seed_a'],
                    seed_b=p['seed_b'],
                    expect=p.get('expect', 'high'),
                    min_cos=p.get('min_cos', -1.0),
                    max_cos=p.get('max_cos', 1.0),
                    description=p.get('description', "")
                ))
            logger.info(f"Loaded {len(self.probes)} probes from {path}")
        except Exception as e:
            logger.error(f"Failed to load probes: {e}")

    def _get_vector(self, seed: int) -> tuple[Optional[np.ndarray], str]:
        """
        Retrieve the vector for a seed from the lattice.
        Priority: 
        1. Neuron.semantic_vector (if valid)
        2. Neuron.weights (normalized)
        """
        if seed not in self.lattice.active_tissue:
            return None, "missing"
            
        neuron = self.lattice.active_tissue[seed]
        
        # 1. Semantic Vector (Explicit)
        if hasattr(neuron, "semantic_vector") and neuron.semantic_vector is not None:
            # Validate dimensions/shape if needed
            return neuron.semantic_vector, "semantic_vector"
            
        # 2. Weights (Implicit)
        w = neuron.weights
        norm = np.linalg.norm(w)
        if norm > 0:
            return w / norm, "weights_norm"
        return w, "weights_raw"

    def run(self) -> List[ProbeResult]:
        """
        Execute all loaded probes against the current lattice state.
        Returns detailed results and optionally persists them.
        """
        results = []
        
        for probe in self.probes:
            vec_a, src_a = self._get_vector(probe.seed_a)
            vec_b, src_b = self._get_vector(probe.seed_b)
            
            if vec_a is None or vec_b is None:
                # One or both missing. Fail safe or skip?
                # A missing anchor is a failure of stability usually.
                logger.warning(f"Probe {probe.seed_a} <-> {probe.seed_b} skipped: Missing seeds (A:{src_a}, B:{src_b})")
                continue
                
            # Compute Cosine Similarity
            # Dot product of normalized vectors
            # Explicit normalization just in case
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            
            if norm_a == 0 or norm_b == 0:
                sim = 0.0
            else:
                sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
                
            # Validate against expectations
            passed = True
            if probe.min_cos > -1.0 and sim < probe.min_cos:
                passed = False
            if probe.max_cos < 1.0 and sim > probe.max_cos:
                passed = False
                
            res = ProbeResult(
                pair=probe,
                cosine_sim=float(sim),
                passed=passed,
                vector_source=f"{src_a}/{src_b}"
            )
            results.append(res)
            
        self._persist_results(results)
        return results
        
    def _persist_results(self, results: List[ProbeResult]):
        """
        Save probe run to repro_logs (for audit).
        """
        # Ensure directory exists
        log_dir = "repro_logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        timestamp = int(time.time())
        filename = f"{log_dir}/probes_{timestamp}.json"
        
        output = {
            "timestamp": timestamp,
            "results": []
        }
        
        pass_count = 0
        for r in results:
            if r.passed: pass_count += 1
            output["results"].append({
                "pair": asdict(r.pair),
                "cosine_sim": r.cosine_sim,
                "passed": r.passed,
                "vector_source": r.vector_source
            })
            
        output["summary"] = {
            "total": len(results),
            "passed": pass_count,
            "pass_rate": pass_count / len(results) if results else 0.0
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist probe results: {e}")

    def summary(self, results: List[ProbeResult]) -> Dict[str, Any]:
        """
        Return high-level stats for the run.
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_cos = np.mean([r.cosine_sim for r in results]) if results else 0.0
        
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "avg_similarity": float(avg_cos)
        }
