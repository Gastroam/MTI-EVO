# 🌌 QUANTUM LAYER ARCHITECTURE
**Component**: `quantum_layer.py`
**Type**: Hypermorphic Neural Module
**Status**: ACTIVE (Native Protocol)

## 1. The Schrödinger's Weights Principle
The `QuantumLayer` implements a neural network layer that exists in a state of **superposition** until the exact moment of execution (Observation).

### The Problem
Running a 27B parameter model (Gemma-3-27B) requires ~50GB+ of VRAM, which is impossible on consumer hardware (e.g., RTX 3070 6GB).

### The Solution: Temporal Sparsity
Instead of loading the entire model into VRAM, we treat each layer as a probability cloud.
- **Superposition**: The layer holds paths to multiple potential implementations (shards) but loads none.
- **Observation (Forward Pass)**: When a token arrives, the layer **collapses** to a concrete implementation.
- **Materialization**: Weights are loaded from disk ("Fast" NVMe or "Precise" RAM cache) into VRAM on-demand.
- **Reset**: After the token leaves, the layer dissolves back into the ether (`reset_reality`), freeing VRAM for the next layer.

## 2. Technical Implementation

### Class: `QuantumLayer(nn.Module)`
Wraps the standard PyTorch `nn.Module` lifecycle with quantum semantics.

#### Key Methods

1.  **`__init__`**:
    - Accepts `shard_paths` (dict of filenames) and `weights` (probability distribution).
    - Stores standard HuggingFace `config` for on-the-fly reconstruction.

2.  **`collapse() -> str`**:
    - Performs a weighted random selection of the implementation key (e.g., "precise").
    - **Determinism**: Once collapsed, the layer stays in that state until explicitly reset.

3.  **`forward(x)`**:
    - **The Act of Observation**.
    - If the layer is not materialized (`collapsed_impl is None`), it triggers the load sequence.
    - **Dynamic Loading**:
        - Uses `safetensors.torch.load_file` to fetch the specific shard.
        - **Smart Prefix Search**: Scans the state dict for `layers.{id}` to find the correct tensors, prioritizing `language_model` keys for Gemma-3.
        - **Reconstruction**: Instantiates a native `Gemma3DecoderLayer` and loads the weights.
    - Executes the forward pass and returns the hidden state.

4.  **`reset_reality()`**:
    - The most critical method for VRAM management.
    - Moves the materialized layer to CPU and deletes it.
    - Calls `gc.collect()` and `torch.cuda.empty_cache()` to ensure the GPU is clean for the next layer.

## 3. Usage Example

```python
# 1. Initialize the Quantum Layer
layer = QuantumLayer(
    layer_id=5,
    shard_paths={"precise": "path/to/model-00005-of-00012.safetensors"},
    weights={"precise": 1.0},
    config=gemma_config
)

# 2. The Forward Pass (Triggers Load -> Compute)
output = layer(hidden_states)

# 3. Collapse & Clean (Frees VRAM)
layer.reset_reality()
```

## 4. Performance Notes
- **Latency**: The primary bottleneck is disk I/O. Using `load_file` (as currently implemented) loads the full shard.
- **Optimization Route**: Future versions could use `safetensors.safe_open` with slicing to load *only* the specific tensors for the layer, reducing I/O by ~90% (currently disabled for stability).
- **Precision**: Weights are loaded in `bfloat16` to match the native model typology.
