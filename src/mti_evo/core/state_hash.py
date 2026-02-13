
import json
import hashlib
from typing import Dict, Any

def hash_neuron_state(state: Dict[str, Any]) -> str:
    """
    Generates a deterministic SHA256 hash of a neuron state dictionary.
    Used for detecting drift, verification, and audit trails.
    
    Args:
        state: Dictionary roughly matching NeuronStateV1 schema.
        
    Returns:
        Hex string of SHA256 hash.
    """
    # Defensive: Filter out transient keys if any (though V1 schema should be pure)
    # Ensure float stability is not an issue by formatting or trusting json.dumps default
    # For strict determinism, we might want to round floats, but for now we trust exact bits.
    
    # Sort keys for consistent ordering
    state_str = json.dumps(state, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

def hash_lattice_state(states: list) -> str:
    """
    Generates a merkle-like root hash for a list of neuron hashes.
    """
    # Sort by seed to ensure order independence
    # Assuming states have 'seed' key
    sorted_states = sorted(states, key=lambda x: x.get('seed', 0))
    hashes = [hash_neuron_state(s) for s in sorted_states]
    combined = "".join(hashes)
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()
