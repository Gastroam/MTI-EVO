"""
MTI-EVO Runtime
===============
The Composition Root for the MTI System.
Wires together Config, Persistence, Cortex, and Engines.
Exposes the Public API.
"""
import os
import time
from typing import Dict, Any, Optional

from mti_evo.core.config import MTIConfig
from mti_evo.core.logger import get_logger
from mti_evo.telemetry import TelemetrySystem, log_event
from mti_evo.cortex.broca import BrocaAdapter
from mti_evo.cortex.memory import CortexMemory
from mti_evo.bootstrap.attractors import ensure_proven_attractors

logger = get_logger("MTI-Runtime")

class SubstrateRuntime:
    """
    The Single Source of Truth for MTI-EVO execution.
    """
    def __init__(self, config: MTIConfig = None, persistence_id: str = "default", read_only: bool = False):
        self.start_time = time.time()
        self.config = config or MTIConfig()
        self.persistence_id = persistence_id
        
        logger.info(f"Initializing SubstrateRuntime ({persistence_id}) [Read-Only: {read_only}]")
        
        # 1. Telemetry
        self.telemetry = TelemetrySystem()
        
        # 2. Persistence (CortexMemory)
        # Runtime manages the lifecycle
        # We need to pass the config implicitly or explicitly? 
        # CortexMemory(backend="mmap", read_only=True/False)
        self.hippocampus = CortexMemory(backend="mmap", read_only=read_only)
        
        # 3. Cognitive Adapter (Broca)
        # Injects Config and Memory
        self.broca = BrocaAdapter(config=self.config, hippocampus=self.hippocampus)
        self.cortex = self.broca.cortex
        
        # 4. Bootstrap Policy (Runtime Decision)
        # Only run attractors if not read-only? Or always?
        # Attractors require write access to Cortex/Memory.
        if not read_only:
             ensure_proven_attractors(self.cortex)
        
        # 3. Engine Registry (Lazy)
        self.engine = None
        
        logger.info("Runtime Ready.")

    def status(self) -> Dict[str, Any]:
        """System Health & Stats."""
        return {
            "uptime": time.time() - self.start_time,
            "neuron_count": len(self.cortex.active_tissue),
            "persistence_id": self.persistence_id,
            "backend": self.hippocampus.backend
        }

    def probe(self, seed: int) -> Dict[str, Any]:
        """Deep scan of a neuron."""
        from mti_evo.api import EvoAPI 
        # Reuse API logic or move logic here?
        # Ideally logic moves to Runtime or Core directly.
        # Broca has probe_neuron? No, API had it.
        # Let's grab it from API or reimplement clean.
        # For now, placeholder or delegate.
        return self.broca.probe_neuron(seed) if hasattr(self.broca, 'probe_neuron') else {}

    def graph(self) -> Dict[str, Any]:
        """Topology export."""
        # This logic was in EvoAPI.get_graph_topology.
        # Should be moved to Runtime or a GraphService.
        # For now, we leave it in API/Broca as we refactor.
        pass

    def resonate(self, text: str) -> Dict[str, Any]:
        """Inject text, get resonance."""
        return self.broca.process_thought(text)

    def reflex(self, prompt: str, max_tokens=1024, temperature=0.7):
        """
        LLM Inference.
        In Substrate mode, this might delegate to IPC if not holding the model.
        """
        raise NotImplementedError("Reflex must be handled by proper Engine or IPC strategy.")
        
    def shutdown(self):
        """Cleanup."""
        if self.hippocampus:
            self.hippocampus.close()
