"""
Eviction policies for holographic lattice memory management.
"""

import time

import numpy as np

from mti_evo.mti_logger import get_logger

logger = get_logger("MTI-Core")


def metabolic_score(neuron, now, decay_rate):
    """
    Compute metabolic score:
    score = mean(abs(weights)) * exp(-decay_rate * inactive_time)
    """
    delta = now - neuron.last_accessed
    factor = np.exp(-decay_rate * delta)
    return np.mean(np.abs(neuron.weights)) * factor


def _select_eviction_candidates(keys, rng, config):
    mode = getattr(config, "eviction_mode", "deterministic_sample")
    sample_size = int(getattr(config, "eviction_sample_size", 50))
    sample_size = max(1, min(len(keys), sample_size))

    if mode == "full_scan":
        return list(keys)

    # Both sample and deterministic_sample now use the passed rng
    # to ensure reproducibility (no global random state).
    if mode in ["sample", "deterministic_sample"]:
        ordered_keys = sorted(keys) # Sort first for stability before sampling
        if sample_size >= len(ordered_keys):
            return ordered_keys
        
        # Use rng.choice on indices to avoid issues with object arrays
        sampled_idx = rng.choice(len(ordered_keys), size=sample_size, replace=False)
        return [ordered_keys[int(i)] for i in np.atleast_1d(sampled_idx)]

    logger.warning(f"Unknown eviction_mode='{mode}'. Falling back to deterministic_sample.")
    ordered_keys = sorted(keys)
    if sample_size >= len(ordered_keys):
        return ordered_keys
    sampled_idx = rng.choice(len(ordered_keys), size=sample_size, replace=False)
    return [ordered_keys[int(i)] for i in np.atleast_1d(sampled_idx)]


def pick_eviction_candidate(active_tissue, rng, config, time_fn=None):
    """
    Selects the seed to evict and whether the decision used fallback logic.
    Returns (seed, fallback, metadata). If tissue is empty, returns (None, False, {}).
    """
    keys = list(active_tissue.keys())
    if not keys:
        return None, False, {}

    candidate_keys = _select_eviction_candidates(keys, rng, config)
    grace_period = getattr(config, "grace_period", 0)
    candidates = []
    
    # Analyze Candidates
    mature_count = 0
    total_candidates = len(candidate_keys)
    
    for seed in candidate_keys:
        neuron = active_tissue[seed]
        if neuron.age > grace_period:
            candidates.append((seed, neuron))
            mature_count += 1

    if not candidates:
        # If no mature neurons were found in the candidate pool,
        # evict first sampled key to guarantee progress under pressure.
        # Fallback Triggered
        chosen = candidate_keys[0] if candidate_keys else keys[0]
        return chosen, True, {
            "reason": "fallback_no_mature",
            "candidates_scanned": total_candidates,
            "mature_found": 0
        }

    decay_rate = getattr(config, "passive_decay_rate", 0.00001)
    
    # Use deterministic time if provided, else fallback (though caller should provide it)
    now = time_fn() if time_fn else time.time()
    
    # Evaluate Scores
    scored_candidates = []
    for seed, neuron in candidates:
        score = metabolic_score(neuron, now, decay_rate)
        scored_candidates.append((score, seed))
    
    # Find minimum score
    min_score, target_seed = min(scored_candidates, key=lambda x: x[0])
    
    return target_seed, False, {
        "reason": "metabolic_score",
        "candidates_scanned": total_candidates,
        "mature_found": mature_count,
        "lowest_score": min_score
    }
