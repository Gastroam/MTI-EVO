# MTI-EVO Engine Architecture

This directory contains the LLM backend engines for MTI-EVO.

## Engine Interface

All engines inherit from `BaseEngine` and must implement:

```python
class BaseEngine(ABC):
    def load_model() -> None
    def infer(prompt: str, max_tokens: int, stop: list, **kwargs) -> LLMResponse
    def embed(text: str) -> List[float]
    def unload() -> None
```

## Available Engines

| Engine | File | Description |
|---|---|---|
| `gguf` | `gguf_engine.py` | GGUF models via `llama-cpp-python`. **Default.** |
| `native` | `native_engine.py` | Safetensors via `transformers` (requires `torch`). |
| `quantum` | `quantum_engine.py` | Hybrid Quantum architecture. |
| `resonant` | `resonant_engine.py` | **Metabolic Layer Activation** (cognitive sector sparse loading). |
| `bicameral` | `bi_camera_engine.py` | **Dual-Model** (4B Limbic + 12B Cortex parallel streams). |
| `qoop` | `qoop_engine.py` | **Quantum OOP** (probabilistic routing via wavefunction collapse). |
| `hybrid` | `hybrid_engine.py` | **Local+API Fusion** (GGUF + cloud API reasoning). |
| `api` | `api_engine.py` | External API calls (OpenAI, Anthropic, etc.). |

## Adding a Custom Engine

1. Create `my_engine.py` in this directory.
2. Inherit from `BaseEngine`.
3. Implement required methods.
4. Register in `llm_adapter.py` (`_select_backend`).

## Configuration

Engines read from `mti_config.json` or the config dict passed at init:
- `model_path`: Path to model file.
- `n_ctx`: Context window size.
- `temperature`: Sampling temperature.
- `gpu_layers`: Layers to offload to GPU (-1 = all).
