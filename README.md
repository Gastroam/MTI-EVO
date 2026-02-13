# MTI-EVO  
**Memory-Substrate LLM Runtime & Cognitive Architecture Framework**

> Local-first. Substrate-aware. Engine-agnostic.

MTI-EVO is a modular cognitive runtime designed to extend Large Language Models with:

-  Persistent semantic substrate (Holographic Lattice)
-  Layered Cortex architecture (Broca, Memory, Introspection, Crystallization)
-  Pluggable engines (GGUF, Native, Resonant, Hybrid, etc.)
-  Substrate multiprocessing runtime (single VRAM holder + HTTP workers)
-  Architecture boundary enforcement & CI rigor

It is not a chatbot wrapper.  
It is an experimental runtime for structured, persistent, substrate-aware AI systems.

---

##  Core Concepts

### 1. Holographic Lattice (Core Substrate)

A deterministic, persistent semantic field where concepts are stored as neurons and stimulated through contextual embeddings.

Features:

- Capacity-bounded lattice
- Eviction policies
- MMap / JSONL persistence backends
- Deterministic seed mapping
- Semantic reinforcement (attractors)
- Isolation-aware testing

---

### 2. Cortex Architecture

mti_evo/
├── core/ # Substrate & persistence
├── cortex/ # Cognitive layers
├── engines/ # LLM engine protocol & registry
├── runtime/ # Substrate runtime
└── server/ # HTTP control plane

---


Cortex modules:

| Component        | Purpose                                      |
|------------------|----------------------------------------------|
| BrocaAdapter     | Text ↔ Seed interface                        |
| CortexMemory     | Unified persistence abstraction              |
| MTIProprioceptor | Cognitive state introspection (Flow/Chaos)   |
| MTICrystallizer  | Collective memory formation                  |
| Bootstrap        | Proven and cultural attractor initialization |

---

### 3. Engine Protocol

MTI-EVO does not hardcode a model backend.

All engines implement a unified protocol:

```python
class EngineProtocol:
    def load(...)
    def infer(...)
    def unload(...)
```

Supported engine types include:

gguf (llama.cpp)

native (Transformers)

resonant

hybrid

experimental engines

Engines are dynamically discovered via the registry system.

---

### 4. Substrate Runtime

The substrate server architecture separates:

Inference Process (VRAM holder)

HTTP Workers (mmap substrate inhabitants)

This enables:

Reduced VRAM duplication

Multiprocessing inference

Persistent substrate continuity

Local-first AI deployment

Controlled concurrency

---

### 🚀 Installation
git clone https://github.com/Gastroam/MTI-EVO.git
cd MTI-EVO
pip install -e .[dev]


Python 3.11 required

### 🧪 Run Tests
pytest


### Architecture boundary checks:

pytest tests/architecture


Type checking:

mypy src/mti_evo/core/persistence/backend.py

### 🌐 Run Substrate Server
mti-substrate


Or:

python -m mti_evo.server.substrate


Health endpoint:

GET /health

🧱 Architectural Boundaries

The project enforces strict isolation rules:

core cannot depend on server

cortex cannot depend on server

engines cannot depend on runtime

Runtime components cannot leak into substrate core

Boundary violations fail CI.

🛠 Development

Development install:

pip install -e .[dev]


Linting:

ruff check .
ruff format .


Type checking:

mypy src/mti_evo/


CI enforces:

Ruff lint rules

MyPy type validation

Boundary integrity tests

Deterministic test isolation

🧩 Plugins

Advanced capabilities (Hive, IDRE, research endpoints) are designed as external plugins.

The core runtime remains:

Minimal

Hardened

Substrate-focused

Engine-agnostic

🔬 Design Goals

Deterministic substrate memory

Local-first execution

Engine-agnostic runtime

Strict architectural isolation

Research-friendly structure

CI-enforced boundaries

Modular extensibility

📜 License

MIT (planned)
