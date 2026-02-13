"""
MTI-EVO Telemetry
=================
Metrics collection for the Holographic Cortex.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class MTIMetrics:
    # Gauges
    neuron_count: int = 0
    avg_resonance: float = 0.0
    
    # Counters
    neurogenesis_total: int = 0
    evictions_total: int = 0
    pulses_processed: int = 0
    
    # Latency
    last_pulse_ms: float = 0.0
    
    _start_time: float = field(default_factory=time.time)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "uptime": time.time() - self._start_time,
            "neuron_count": self.neuron_count,
            "neurogenesis": self.neurogenesis_total,
            "evictions": self.evictions_total,
            "avg_resonance": self.avg_resonance,
            "latency_ms": self.last_pulse_ms,
            "pulses_processed": self.pulses_processed
        }

class TelemetrySystem:
    def __init__(self):
        self.metrics = MTIMetrics()
        
    def record_pulse(self, active_count, resonance, duration_ms):
        self.metrics.neuron_count = active_count
        self.metrics.avg_resonance = resonance
        self.metrics.last_pulse_ms = duration_ms
        self.metrics.pulses_processed += 1
        
    def record_neurogenesis(self):
        self.metrics.neurogenesis_total += 1
        
    def record_eviction(self):
        self.metrics.evictions_total += 1
