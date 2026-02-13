from .policy import EvictionPolicy
from .standard import StandardEvictionPolicy

# Global Default Instance (for backward compat if needed)
_default_policy = StandardEvictionPolicy()

def pick_eviction_candidate(active_tissue, rng, config, time_fn=None):
    """
    Legacy wrapper for backward compatibility.
    """
    return _default_policy.pick_candidate(active_tissue, rng, config, time_fn)

def get_policy(name: str) -> EvictionPolicy:
    """
    Factory for eviction policies.
    """
    if name == "standard":
        return StandardEvictionPolicy()
    # Add future policies here
    return StandardEvictionPolicy()
