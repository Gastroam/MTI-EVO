# The Holographic Lattice: Internal Logic

The Core (`src/mti_core.py`) is a pure Python implementation of a sparse biological network.

```mermaid
classDiagram
    class MTIConfig {
        +float gravity = 20.0
        +float momentum = 0.9
        +float decay_rate = 0.15
        +int capacity_limit = 5000
    }

    class MTINeuron {
        +ndarray weights
        +float bias
        +float velocity
        +int age
        +forward(input_signal)
        +backward(error)
        +prune() bool
    }

    class HolographicLattice {
        +Dict[int, MTINeuron] active_tissue
        +stimulate(seed_stream, signal) float
        +neurogenesis(seed)
        +prune_tissue()
    }

    HolographicLattice *-- MTINeuron : Contains
    HolographicLattice ..> MTIConfig : Configured By
```

## The Learning Cycle
1.  **Stimulation**: `stimulate(seeds, 1.0)` is called.
2.  **Lookup**: Lattice checks `active_tissue` (Hash Map).
3.  **Neurogenesis**: If seed is missing, `MTINeuron` is created using `MTIConfig` params.
4.  **Resonance**: `Neuron.forward()` calculates activation.
5.  **Reflex**: If `learn=True`, `Neuron.backward()` applies gradients (Momentum).
6.  **Entropy**: `prune_tissue()` helps the weak die (Gravity).
