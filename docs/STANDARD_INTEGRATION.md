# Standard MTI-EVO Integration Guide (v2.3)

This guide defines the **canonical** method for integrating with the MTI-EVO architecture. It standardizes how scripts should instantiate Intelligence (Engines) and Memory (Substrate).

> **Axiom**: Do not invent new ways to load models. Use the factory methods.

---

## 1. Intelligence (The Engine)

Direct instantiation of engine classes (e.g., `GGUFEngine(...)`) is **DEPRECATED**.
All scripts must use the `LLMAdapter` factory, which respects the central `mti_config.json`.

### Standard Pattern

```python
from mti_evo.config import load_config
from mti_evo.llm_adapter import LLMAdapter

# 1. Load Universal Config (Env > JSON > Defaults)
config = load_config()

# 2. Instantiate Intelligence
# Auto-detects backend (Quantum, Native, GGUF, API) based on config
adapter = LLMAdapter(config=config, auto_load=True)

# 3. Inference
response = adapter.infer("Define resonance.", max_tokens=128)
print(f"Response: {response.text}")
print(f"Latency: {response.latency_ms}ms")
```

### Why?
- **Portability**: Your script will run on a high-end rig (Quantum) or a laptop (GGUF) without code changes.
- **Resource Management**: The Adapter handles VRAM loading, unloading, and context window management.
- **Monitoring**: Centralized logging and telemetry hook into the Adapter.

---

## 2. Memory (The Substrate)

Direct loading of JSON dumps (`json.load(open('cortex_dump.json'))`) is **DEPRECATED** for core logic.
All memory access must go through `MTIHippocampus`, which defaults to the high-performance `mmap` backend.

### Standard Pattern

```python
from mti_evo.mti_hippocampus import MTIHippocampus

# 1. Connect to the Substrate
# "auto" prefers mmap (72x faster) if available, falls back to JSON for legacy data
hippocampus = MTIHippocampus(backend="auto")

# 2. Recall (Rehydrate the Lattice)
# Returns a dictionary of active neurons {seed: NeuronObject}
memory_tissue = hippocampus.recall()

print(f"Connected to Substrate: {len(memory_tissue)} neurons active.")

# 3. Consolidation (Save State)
# Only necessary if you modified weights/topology
hippocampus.consolidate(memory_tissue)
hippocampus.close()
```

### Direct Substrate Access (Advanced)
For tools needing real-time manipulation without loading the whole brain:

```python
# Force mmap backend for O(1) random access
substrate = MTIHippocampus(backend="mmap")
neuron = substrate.get_neuron(seed=123456)

if neuron:
    print(f"Neuron 123456 Gravity: {neuron.gravity}")
```

---

## 3. Configuration

Do not hardcode paths. Use `src/mti_evo/config.py`.

- **Reading**: `config = load_config()`
- **Writing**: Configuration is managed via the standard `mti_config.json` file in the root directory.

### Environment Variables override JSON
- `MTI_MODEL_PATH` -> overrides `model_path`
- `MTI_MODEL_TYPE` -> overrides `model_type`

---

**Effective Date**: January 31, 2026
**Compliance**: Mandatory for all `playground/` and `src/` scripts.
