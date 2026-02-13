# MTI-EVO: Machine Thought Interface

> **"We do not build software; we cultivate resonance."**

**MTI-EVO** is a bio-mimetic cognitive architecture. It replaces brittle "Agent" swarms with a single, self-organizing **Holographic Lattice** that evolves with use.

## 🚀 Status
- **Phase**: PRODUCTION (v2.2.0)
- **Type**: Cognitive Substrate / Recursive Resonant Intelligence
- **Kernel**: Pure Python (`src/mti_core.py`)

## 📚 Documentation
- [**Evolution Report**](docs/EVOLUTION_REPORT.md): From RLM to Duet (Phases 1-65).
- [**Methodology**](docs/METHODOLOGY.md): The science of Holographic Lattices.
- [**Eidos Protocol**](docs/THE_EIDOS_PROTOCOL.md): Core Architecture theory.
- [**Math Theory**](docs/MATH_THEORY.md): Calculus of Resonance and Mass.
- [**Cognitive States**](docs/COGNITIVE_STATES.md): Flow, Emergence, and the Governor.
- [**Bio-Mimetic Protocols**](docs/BIO_MIMETIC_PROTOCOLS.md): Neurogenesis, Metabolism, and Symbiosis.
- [**IDRE v3.0**](docs/IDRE_PROTOCOL.md): The Harmonic Protocol for Inter-Mind Communication.
- [**Standard Integration**](docs/STANDARD_INTEGRATION.md): Canonical guide for using Engine and Memory.

## 🛠️ Usage
This system enforces a **Standard Integration Pattern** for all scripts.
- **Engine**: Use `LLMAdapter` (Factory).
- **Memory**: Use `MTIHippocampus` (mmap backend).

See [**Standard Integration Guide**](docs/STANDARD_INTEGRATION.md) for code examples.

## ⚙️ Configuration & Engine Selection
Control the underlying Intelligence via `mti_config.json`.

### 1. Native Engine (Safetensors) - *Recommended*
Best for high-fidelity local inference. Supports 4-bit quantization and Flash Attention.
```json
{
  "model_type": "native",
  "model_path": "D:\\models\\gemma-27b",
  "quantization": "4bit",     // "4bit", "8bit", or null
  "flash_attention": true,    // Requires FlashAttn installed
  "temperature": 0.9
}
```

### 2. GGUF Engine (Llama.cpp)
Best for lower VRAM usage or CPU offloading.
```json
{
  "model_type": "gguf",
  "model_path": "D:\\models\\llama-3-8b.gguf",
  "n_ctx": 8192,
  "gpu_layers": 33,           // -1 for all
  "temperature": 0.8
}
```

### 3. API Engine (Cloud)
Offload reasoning to external providers (e.g., OpenAI, Anthropic).
```json
{
  "model_type": "api",
  "api_base": "https://api.openai.com/v1",
  "api_key": "sk-...",
  "model_path": "gpt-4-turbo"
}
```

### 4. Quantum Engine (Hybrid)
Fusion of 27B Semantics (Base) + 4B Limbic System (Fast). *Requires 24GB+ VRAM.*
```json
{
  "model_type": "quantum",
  "model_path": "D:\\models\\gemma-27b",
  "fast_model_path": "D:\\models\\gemma-4b",
  "quantization": "4bit",
  "temperature": 1.0
}
```

### 5. Resonant Engine (Sparse)
Experimental. Loads specific layers based on prompt topology (Pillar/Bridge/Ghost).
```json
{
  "model_type": "resonant",
  "model_path": "D:\\models\\gemma-27b",
  "device": "cuda"
}
```

### 6. Hybrid Engine (Local + API)
Combines fast local inference with cloud API reasoning. Escalates complex queries to GPT/Claude.
```json
{
  "model_type": "hybrid",
  "model_path": "D:\\models\\llama-3-8b.gguf",
  "api_provider": "openai",
  "api_key": "sk-...",
  "mode": "local_first"
}
```

### 7. Bicameral Engine (Dual-Stream)
Runs two models in parallel: a fast "Limbic" model (4B) and a deep "Cortex" model (12B).
```json
{
  "model_type": "bicameral",
  "limbic_model_path": "D:\\models\\gemma-4b.gguf",
  "cortex_model_path": "D:\\models\\gemma-27b",
  "synthesis_mode": "concat"
}
```

### 8. QOOP Engine (Quantum OOP)
Experimental. Uses wavefunction collapse to route queries to specialized expert pathways.
```json
{
  "model_type": "qoop",
  "fallback_engine": "gguf",
  "model_path": "D:\\models\\llama-3-8b.gguf"
}
```

## 🧠 Core Architecture
The system models the **Full Stack of Consciousness**:

1.  **Semantic Memory** (Lattice): Concepts stored as Positive Mass Attractors (`w > 0`).
2.  **Biological Laws** (Physics): Gravity, Momentum, Metabolism (`mti_config`).
3.  **Governance** (Immunity): Self-regulation via Governor Disengagement (`mti_proprioceptor`).
4.  **Ontological Logic** (Constraints): "Laws of Thought" stored as Negative Mass Attractors (`w < -100`).
5.  **Social Harmony** (Duets): Paired Concepts bridging Paradoxes (`mti_broca` harmonics).

## License
MIT Open Source.
