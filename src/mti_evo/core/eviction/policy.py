from typing import Protocol, Any, Tuple, Dict, Optional

class EvictionPolicy(Protocol):
    """
    Protocol for Eviction Strategies.
    """
    def pick_candidate(self, active_tissue: Dict[int, Any], rng: Any, config: Any, time_fn: Any) -> Tuple[Optional[int], bool, Dict[str, Any]]:
        """
        Selects a seed to evict.
        Returns: (seed, fallback_used, metadata)
        """
        ...
