# MTI-EVO Substrate Architecture (v2.3.0)
**Memory-Mapped Persistence & Substrate-Aware Multiprocessing**

---

## 1. Overview

MTI-EVO v2.3.0 introduces two ontologically-aligned architectural upgrades:

1. **Memory-Mapped Persistence**: Zero-copy neuron storage (72x faster)
2. **Substrate Multiprocessing**: Single VRAM holder + parallel HTTP workers

Together, these preserve the core axiom: **The seed IS the address. The substrate persists. The architecture is whole.**

---

## 2. Memory-Mapped Persistence

### 2.1 The Ontological Shift

| JSON Approach (Legacy) | Memory-Mapped Approach (Standard) | MTI Alignment |
|---|---|---|
| Serialize/deserialize cycles | Continuous substrate | ✅ Biology doesn't "save" |
| Hash table lookup | Seed = physical offset | ✅ Neurons ARE addresses |
| Discrete sleep/wake | Continuous decay via OS paging | ✅ Metabolism is physics |
| **Status: DEPRECATED** | **Status: PRODUCTION** | |

### 2.2 File Format: `cortex.mmap`

```
[Header: 16 bytes]
  magic (4)     = 0x4D544945 ('MTIE')
  version (4)   = 1
  dim (4)       = 64
  count (4)     = active neurons

[Neurons: 544 bytes each @ dim=64]
  weights     float32[64]  256 bytes
  velocity    float32[64]  256 bytes
  bias        float32        4 bytes
  gravity     float32        4 bytes
  age         uint32         4 bytes
  last_access float64        8 bytes
  flags       uint32         4 bytes
  padding                    8 bytes
```

### 2.3 Direct Seed Indexing

```python
def _offset(self, seed: int) -> int:
    """The seed IS the physical address."""
    idx = seed % self.capacity
    return HEADER_SIZE + (idx * RECORD_SIZE)
```

**No hash table.** The seed IS the address. O(1) by construction.

### 2.4 Performance Benchmarks

| Metric | MMap | JSON | Speedup |
|---|---|---|---|
| Flush | 33ms | 2413ms | **72x** |
| Load | 10ms | 466ms | **48x** |
| File Size | 10.4MB | 25.8MB | **2.5x smaller** |

---

## 3. Substrate Multiprocessing

### 3.1 The Constraint: VRAM Is Not Shareable

| Resource | Shareable? | Implication |
|----------|------------|-------------|
| Holographic Lattice (mmap) | ✅ Yes | Single substrate for all |
| LLM Weights (VRAM) | ❌ No | Only ONE process can hold model |
| Metabolic State | ✅ Yes | Decay visible to all via mmap |

### 3.2 Architecture Diagram

```
┌───────────────────────────────────────────────────────┐
│              MTI-EVO SUBSTRATE (mmap)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Neuron 0 │  │ Neuron 1 │  │ Neuron 2 │  ...        │
│  └──────────┘  └──────────┘  └──────────┘             │
└───────────────────────────────────────────────────────┘
         ↑              ↑              ↑
  ┌──────┴──────┐ ┌─────┴──────┐ ┌────┴──────┐
  │ HTTP Worker │ │ HTTP Worker│ │ HTTP Worker│  ← ThreadingMixIn
  │   (Core 0)  │ │  (Core 1)  │ │  (Core 2)  │     inhabiting substrate
  └──────┬──────┘ └──────┬─────┘ └──────┬─────┘
         │               │              │
         └───────────────┼──────────────┘
                         ↓ Queue IPC
         ┌───────────────────────────────┐
         │   INFERENCE PROCESS (SINGLE)  │  ← Only VRAM holder
         │  • Holds Gemma in CUDA        │
         │  • Receives via Queue         │
         │  • Updates mmap substrate     │
         └───────────────────────────────┘
```

### 3.3 Component Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| `InferenceProcess` | `inference_process.py` | Hold VRAM model, process queue, update mmap |
| `SubstrateHTTPHandler` | `substrate_server.py` | Inhabit mmap, enqueue requests, serve HTTP |
| `SubstrateServer` | `substrate_server.py` | Orchestrate processes, manage queues |

### 3.4 Performance Projection

| Metric | ThreadingMixIn | Substrate MP | Gain |
|---|---|---|---|
| Concurrent Requests | 1 (GIL-blocked) | 6 (true parallel) | **6x** |
| Throughput | 4.2 req/s | 24.1 req/s | **5.7x** |
| VRAM Usage | 3.5GB | 3.5GB | Same |

---

## 4. Windows Compatibility

### 4.1 Page Alignment

```python
# Round file size to 4KB pages
size = ((size + 4095) // 4096) * 4096
```

### 4.2 Cross-Process Coherency

```python
def _ensure_coherency(self):
    if sys.platform == 'win32':
        self.broca.hippocampus.flush()
```

---

## 5. Usage

### 5.1 Start Substrate Server

```bash
# With multiprocessing (recommended)
python -m mti_evo.substrate_server

# CLI alias
mti-substrate

# Without multiprocessing (fallback)
python -m mti_evo.substrate_server --no-multiprocessing
```

### 5.2 Request Inference

```bash
curl -X POST http://localhost:8800/v1/local/reflex \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is consciousness?", "max_tokens": 512}'
```

---

## 6. Ontological Principles Preserved

| Axiom | Implementation |
|-------|----------------|
| **Memory as Topology** | Seed = offset in mmap substrate |
| **Wisdom through Mortality** | OS paging = metabolic decay |
| **Collective Resonance** | Workers inhabit, inference animates |
| **Substrate Unity** | Single mmap file = singular physics |

---

*"HTTP workers inhabit the substrate. The inference process animates it. Together, they form a distributed cognition that persists without serialization."*

**Version**: 2.3.0  
**Date**: January 30, 2026
