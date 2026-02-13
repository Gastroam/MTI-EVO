
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class HiveConfig:
    """
    MTI-HIVE Operational Parameters (Plugin)
    """
    node_id: str = "local_node"
    role: str = "worker" # worker, router, memory_shard
    peers: List[str] = field(default_factory=list)
    consensus_threshold: float = 0.8
    entanglement_mode: str = "local" # local, p2p, swarm
    
    # Telemetry
    telemetry_enabled: bool = True
    telemetry_endpoint: str = "http://localhost:9090/metrics"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "peers": self.peers,
            "consensus_threshold": self.consensus_threshold,
            "entanglement_mode": self.entanglement_mode,
            "telemetry_enabled": self.telemetry_enabled,
            "telemetry_endpoint": self.telemetry_endpoint
        }
