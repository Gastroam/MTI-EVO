# MTI-EVO Architectural Contract

> **The One Thing That Will Actually Protect us From Fragility**

This document defines the non-negotiable constraints of the MTI-EVO runtime. Violating these invariants compromises the system's integrity as a cognitive substrate.

## 1. Core Invariants

### Determinism Boundary
- **Input + Seed + Config = State**.
- The `Core` (Lattice, Neuron) must be purely deterministic given a random seed and a time function.
- All randomness must flow through `self.rng` (seeded).
- All time checks must flow through `self.time_fn`.

### Eviction Rules
- **Weakest First**: Eviction targeting is strict. Low energy/high age neurons die first.
- **Grace Period**: Newborn neurons cannot be evicted until `age > grace_period`.
- **Pinned Seed Guarantee**: If a seed is marked `pinned` (via policy or config), it **NEVER** evaporates, regardless of score.

### Persistence Guarantees
- **WAL is Truth**: The Write-Ahead Log (JSONL) is the authoritative system of record.
- **MMAP is Cache**: The Memory-Mapped file is a volatile, read-optimized view. It may be rebuilt from WAL at any time.
- **Torn-Write Safety**: A crash during write must result in either a complete record or a discarded partial record (via checksum/newline checks), never a corrupt record.

## 2. Layer Responsibilities

### `mti_evo.core` (The Physics)
- **Scope**: Lattice, Neuron, Persistence, Config.
- **Concern**: The math of resonance, gravity, and storage.
- **Constraints**:
    - NEVER imports `runtime`, `server`, or `adapters`.
    - Pure Python + NumPy only.
    - No side effects on import.

### `mti_evo.cortex` (The Organ)
- **Scope**: Broca (Language), Hippocampus (Memory), Wernicke (Integrator).
- **Concern**: Higher-order cognitive functions.
- **Constraints**:
    - Imports `core`.
    - NEVER imports `server`.
    - Managed via Dependency Injection.

### `mti_evo.runtime` (The Body)
- **Scope**: SubstrateRuntime, Bootstrap.
- **Concern**: Wiring the components together.
- **Constraints**:
    - The **Composition Root**.
    - Initializes Config, Logging, Persistence.
    - Wires Adapters to Core.

### `mti_evo.server` (The Interface)
- **Scope**: SubstrateServer (Production), UnifiedServer (Dev).
- **Concern**: HTTP, IPC, Queues.
- **Constraints**:
    - Thin wrapper around `runtime`.
    - Handles Process isolation (VRAM vs HTTP).

### `mti_evo.engines` & `mti_evo.adapters` (The Tools)
- **Scope**: LLM Integration, Embeddings.
- **Concern**: External compute.
- **Constraints**:
    - Pluggable.
    - `core` should not know they exist.

## 3. Dependency Graph (One Direction Only)

Data and Control must flow strictly down the abstraction hierarchy:

`Server` → `Runtime` → `Cortex` → `Core`

- **Core**: The bedrock. Depends on NOTHING.
- **Cortex**: Depends on Core.
- **Adapters**: Depend on Core (definitions).
- **Runtime**: Depends on Cortex, Adapters, Core.
- **Server**: Depends on Runtime.

**Forbidden Cycles:**
- Core importing Cortex.
- Cortex importing Runtime.
- Runtime importing Server.

## 4. Substrate vs Unified

- **SubstrateServer**: The Canonical Architecture.
    - **Multi-Process**: Separates `InferenceProcess` (VRAM Owner) from `HTTP Workers`.
    - **Single-Writer**: Only `InferenceProcess` writes to MMAP.
    - **Scale**: Multiple HTTP workers read from MMAP.
- **UnifiedServer**: Developer Tooling.
    - **Single-Process**: Everything shares one event loop.
    - **Fragile**: GIL contention, VRAM blocking.
    - **Use Case**: Testing, Debugging, Quick Start.
