# MTI-EVO Source Documentation

The **Machine Thought Interface (MTI-EVO)** is a bio-mimetic cognitive architecture.

## Core Components

### 1. The Holographic Lattice (`mti_core.py`)
The central processing unit. A sparse, infinite-capacity associative memory system.
- **HolographicLattice**: Manages the lifecycle of memories (Neurogenesis, Adaptation, Eviction).
- **MTINeuron**: The fundamental unit, implementing the "Resonance Logic".

Internal module layout (safe modularization):
- `core/neuron.py`: `MTINeuron` and shared input validation.
- `core/eviction.py`: eviction scoring and candidate selection.
- `core/lattice.py`: `HolographicLattice` orchestration.
- `mti_core.py`: compatibility facade re-exporting the public classes.

### 2. Configuration (`mti_config.py`)
Centralized management of "Biological Laws".
```python
from src.mti_config import MTIConfig

config = MTIConfig(
    gravity=25.0,           # Pain / Error Magnitude
    capacity_limit=10000,   # Max Neurons
    random_seed=1337,       # Deterministic replay seed
    deterministic=True,     # Reproducible RNG behavior
    eviction_mode="deterministic_sample",
    eviction_sample_size=50,
    log_level="DEBUG"
)
```

### 3. Telemetry (`mti_telemetry.py`)
Real-time metrics for system health.
- **Neuron Count**: Total active nodes.
- **Avg Resonance**: Coherence of the current thought.
- **Evictions**: Rate of memory recycling.

### 4. Logging (`mti_logger.py`)
Structured logging powered by `rich`.
- `INFO`: Standard operations (Neurogenesis).
- `DEBUG`: Detailed mathematical traces.
- `WARNING`: System stress (Capacity reached).

## Usage Example

```python
from src.mti_core import HolographicLattice
from src.mti_config import MTIConfig

# 1. Configure the Brain
config = MTIConfig(capacity_limit=100)
lattice = HolographicLattice(config=config)

# 2. Stimulate (Input -> Resonance)
# input_signal can be scalar (presence) or vector (pattern)
resonance = lattice.stimulate(seed_stream=[101, 102], input_signal=[1.0, 0.5])

# 3. Check Stats
print(lattice.telemetry.metrics.snapshot())
```

## Hardening Test Entrypoint

```bash
python -m pytest
```
