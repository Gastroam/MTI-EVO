

# MTI-EVO Core: substrate memory runtime for local-first LLM orchestration

## 1. This document defines the **current state** of the MTI-EVO Core runtime.

It intentionally excludes:

* IDRE (security plugin)
* Hive (distributed layer)
* Ghost (code grounding)
* Experimental engines (native / quantum / resonant)
* Research endpoints

This document describes **Core only**.

---

# 2. Architectural Philosophy

MTI-EVO Core is a:

* Deterministic substrate memory runtime
* Engine-agnostic LLM orchestration layer
* Strictly layered architecture
* CI-enforced boundary system

It is not:

* A chatbot wrapper
* A UI framework
* A prompt engineering toolkit
* A distributed cluster system

It is a substrate-aware local LLM runtime.

---

# 3. Layer Model

```
┌──────────────────────────┐
│ Server / Runtime         │  (Delivery / Orchestration)
├──────────────────────────┤
│ Engines                  │  (Inference Backends)
├──────────────────────────┤
│ Cortex                   │  (Cognitive Layer)
├──────────────────────────┤
│ Core                     │  (Substrate & Persistence)
└──────────────────────────┘
```

## 3.1 Core Layer

Responsibilities:

* HolographicLattice
* MTINeuron
* Deterministic stimulation
* Capacity limits
* Eviction policies
* Batch stimulation
* Persistence protocol

Constraints:

* Must not import Server
* Must not import Runtime
* Must not import Bootstrap
* Must not import Plugins

Enforced by AST boundary tests.

---

## 3.2 Cortex Layer

Responsibilities:

* BrocaAdapter (Text ↔ Seed)
* CortexMemory (Persistence abstraction)
* MTIProprioceptor (state introspection)
* MTICrystallizer (memory consolidation logic)

Constraints:

* Cannot depend on Server
* Cannot depend on Runtime
* Cannot depend on Plugins

Persistence injection via dependency inversion.

---

## 3.3 Engine Layer

Responsibilities:

* Implement EngineProtocol
* Provide load / infer / unload
* Register via EngineRegistry

Rules:

* Engines must not depend on Server
* Engines must not depend on Runtime
* Engines must not mutate Core directly

InferenceProcess interacts with engines via registry only.

---

## 3.4 Runtime Layer

Two modes:

### SubstrateServer (Production)

* Single InferenceProcess (VRAM holder)
* Threaded HTTP workers
* IPC via Queue + futures
* Read-only substrate enforcement on HTTP side
* Single writer model for persistence safety

### DevServer (Development)

* Simplified single-process orchestration
* Uses EngineRegistry
* Not canonical for production

Runtime is the Composition Root.

---

# 4. Persistence Model

Unified via `PersistenceBackend` protocol.

Backends:

* MMapNeuronStore
* JsonlPersistence

Features:

* WAL replay
* Replay marker recovery
* Torn-write tolerance (end-of-file only)
* Batch upsert_neurons
* Single flush per consolidation cycle
* Windows file-lock tolerance

Concurrency:

* CortexMemory guarded via RLock
* Writer-only persistence path

Invariant:

```
norm(weights) <= weight_cap
```

Enforced in MTINeuron.

---

# 5. Performance State

Completed optimizations:

### Hot Path

* Localized member references
* In-place NumPy operations
* Vectorized adapt()

Result:
~10–20% serial speed improvement

---

### Batch Stimulation

```
logits = X @ W.T + B
```

~9.9x throughput improvement

Equivalence verified in:

* test_batch_equivalence.py

---

### Persistence Tuning

* Consolidation now O(1) flush per cycle
* Upsert batch API implemented

---

# 6. Concurrency Model

Substrate Architecture:

* Single VRAM owner process
* Multi-thread HTTP workers
* Queue-based IPC
* Futures-based response dispatch
* No polling race conditions

Tested with:

* test_substrate_concurrency.py
* 20 concurrent request stress validation

---

# 7. CI Enforcement

CI Pipeline enforces:

* Ruff linting
* MyPy type checking
* Pytest suite
* AST architectural boundary tests

Core invariants are tested automatically.

---

# 8. Security Boundary

Core contains **no security layer**.

Security mechanisms are externalized to:

```
mti_evo_plugins/
```

Server integrates security optionally.

Core remains neutral and substrate-focused.

---

# 9. Known Limits (Honest State)

* No distributed clustering in core
* No long-term substrate migration tooling
* No formal benchmarking vs other runtimes

---

# 10. Canonical Definition of Core

MTI-EVO Core is currently defined as:

> A deterministic, persistence-backed, batch-optimized substrate runtime for local LLM orchestration with enforced architectural isolation and concurrency safety.

