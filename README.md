
# MTI-EVO

**Memory-Substrate LLM Runtime & Cognitive Architecture Framework**

> Local-first. Substrate-aware. Engine-agnostic.

MTI-EVO is a modular runtime framework that extends Large Language Models with a persistent semantic substrate and a layered cognitive architecture.

It is not a chatbot wrapper.
It is an experimental runtime for structured, persistent, substrate-aware AI systems.

---

## Overview

MTI-EVO introduces a separation between:

* **Inference engines** (LLM backends)
* **Semantic substrate** (persistent concept field)
* **Cognitive orchestration layers**
* **Runtime and concurrency control**

The goal is to provide a composable architecture where memory, inference, and orchestration are independent, testable, and extensible components.

---

## Core Architecture

```
mti_evo/
├── core/       # Substrate & persistence
├── cortex/     # Cognitive layers
├── engines/    # LLM engine protocol & registry
├── runtime/    # Substrate runtime orchestration
└── server/     # HTTP control plane
```

---

## 1. Holographic Lattice (Core Substrate)

The Holographic Lattice is a deterministic, capacity-bounded semantic field.

Concepts are represented as neurons identified by stable seeds.
Contextual embeddings stimulate and reinforce these neurons.

### Features

* Capacity-bounded lattice
* Eviction policies
* Deterministic seed mapping
* Semantic reinforcement (attractors)
* MMap and JSONL persistence backends
* Replay marker & WAL recovery
* CI-tested persistence integrity
* Isolation-aware testing

The lattice is independent from any specific LLM backend.

---

## 2. Cortex Architecture

The `cortex/` package contains higher-level cognitive components built on top of the substrate.

| Component        | Purpose                         |
| ---------------- | ------------------------------- |
| BrocaAdapter     | Text ↔ Seed interface           |
| CortexMemory     | Unified persistence abstraction |
| MTIProprioceptor | Cognitive state introspection   |
| MTICrystallizer  | Collective memory formation     |
| Bootstrap        | Attractor initialization        |

Cortex modules do not depend on runtime or server layers.

---

## 3. Engine Protocol

MTI-EVO does not hardcode a model backend.

All engines implement a unified interface:

```python
class EngineProtocol:
    def load(self, config: dict) -> None: ...
    def infer(self, prompt: str, **kwargs) -> EngineResult: ...
    def unload(self) -> None: ...
```

### Supported Engine Types

* `gguf` (llama.cpp)
* `native` (Transformers-based)
* `resonant`
* `hybrid`
* Experimental engines

Engines are dynamically discovered via the registry system.

The runtime is engine-agnostic.

---

## 4. Substrate Runtime

The Substrate Runtime separates:

* **Inference Process** (single VRAM holder)
* **HTTP Workers** (mmap substrate inhabitants)

This architecture enables:

* Reduced VRAM duplication
* Multiprocessing inference
* Persistent substrate continuity
* Controlled concurrency
* Local-first deployment
* Queue-based IPC for inference

This model supports scalable local inference while maintaining a shared semantic substrate.

---

## Installation

Python 3.11 required.

```bash
git clone https://github.com/Gastroam/MTI-EVO.git
cd MTI-EVO
pip install -e .[dev]
```

---

## Running Tests

```bash
pytest
```

### Architecture boundary checks

```bash
pytest tests/architecture
```

### Type checking

```bash
mypy src/mti_evo/
```

---

## Running the Substrate Server

```bash
mti-substrate
```

or:

```bash
python -m mti_evo.server.substrate
```

Health endpoint:

```
GET /health
```

---

## Architectural Boundaries

The project enforces strict isolation rules:

* `core` cannot depend on `server`
* `cortex` cannot depend on `server`
* `engines` cannot depend on `runtime`
* Runtime components cannot leak into substrate core

Boundary violations fail CI.

---

## Development

Development install:

```bash
pip install -e .[dev]
```

### Linting

```bash
ruff check .
ruff format .
```

### Type checking

```bash
mypy src/mti_evo/
```

### CI Enforcement

The CI pipeline validates:

* Ruff lint rules
* MyPy type checking
* Architecture boundary tests
* Deterministic test isolation
* Cross-platform compatibility (Windows + Linux)

---

## Plugins

Advanced capabilities (e.g., Hive, IDRE, research endpoints) are designed as external plugins.

The core runtime remains:

* Minimal
* Hardened
* Substrate-focused
* Engine-agnostic
* Dependency-isolated

---

## Design Goals

* Deterministic substrate memory
* Local-first execution
* Engine-agnostic runtime
* Strict architectural isolation
* Research-friendly modular structure
* CI-enforced stability
* Modular extensibility

---

## License

Apache 2.0

---

## Vision

MTI-EVO explores a runtime model where:

* Memory is a persistent substrate, not a temporary context window
* Engines are modular inference components
* Concurrency is explicitly controlled
* Architecture boundaries are enforced by CI

The objective is not to build another assistant.

The objective is to provide a composable cognitive runtime framework.

---
