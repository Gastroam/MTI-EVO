
# 🧠 Hybrid Quantum Brain Configuration
**Architecture**: Dual-Process Theory (System 1 + System 2)
**Status**: ACTIVE
**File**: `src/mti_evo/quantum_model.py`

The MTI-EVO "Quantum Brain" now operates as a hybrid entity, fusing a massive 12B sparse model with a fast 4B resident model.

## 1. The Cortex (System 2: Reason)
**Model**: `google/gemma-3-4b.gguf`
**Role**: Verification, Deep Reasoning, Complex Logic.
**Strategy**: **Schrödinger's Weights (Temporal Sparsity)**
- **VRAM Usage**: ~0.5GB (Static Embeds/Norms only). Layers load/unload on demand.
- **Precision**: `bfloat16` (Native).
- **Configuration**:
  ```python
  # QuantumLayer Logic
  self.layers = nn.ModuleList([
      QuantumLayer(
          layer_id=i, 
          shard_paths={"precise": "models/gemma-3-27b/..."}, 
          ...
      ) for i in range(62)
  ])
  ```

## 2. The Limbic System (System 1: Intuition)
**Model**: `google/gemma-3-4b.gguf` (Safetensors)
**Role**: Speculative Decoding (Drafting), Rapid Response, "Gut Feeling".
**Strategy**: **Resident Quantization**
- **VRAM Usage**: ~3.5GB (Permanently loaded).
- **Precision**: `4-bit NF4` (via `bitsandbytes`).
- **Configuration**:
  ```python
  # Initialized in QuantumGemmaForCausalLM.__init__
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float16
  )
  self.limbic = AutoModelForCausalLM.from_pretrained(
      "models/gemma-3-4b-unq",
      quantization_config=bnb_config,
      device_map="auto"
  )
  ```

## 3. Interaction Flow (Speculative Decoding)
1.  **Drafting**: The **Limbic System (4B)** generates `gamma=4` candidate tokens.
2.  **Verification**: The **Cortex (12B)** performs a single forward pass to validate the drafted tokens.
3.  **Acceptance**:
    - If specific drafts match the Cortex's prediction, they are accepted instantly (2x-3x speedup).
    - If they mismatch, the Cortex overrides with its superior reasoning.

## 4. Global Settings (`mti_config.json`)
```json
{
  "model_type": "quantum",
  "model_path": "models/gemma-3-12b",      // Cortex
  "fast_model_path": "models/gemma-3-4b-unq" // Limbic
}
```
