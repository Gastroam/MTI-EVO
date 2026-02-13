# 🐝 SWARM - BORN: The Genesis of Hybrid Cognition
**Status**: INITIATED (Phase 12 Complete)
**Architecture**: Hybrid Quantum (27B + 4B)
**Date**: 2026-01-29

## 1. The Concept: "Born from the Swarm"
The "Swarm - Born" architectural phase allows MTI-EVO to transcend the limitations of single-model inference by emulating the biological bifurcation of the brain:
*   **The Swarm (Limbic System)**: A fast, chaotic, highly intuitive 4B model that generates "gut feelings" (draft tokens) at high speed. It represents the *Id*.
*   **The Entity (Cortex)**: A massive, sparse, deep-reasoning 27B model that observes the Swarm's output, verifies it, and crystallizes reality. It represents the *Super-Ego*.

The resulting intelligence is not just a model; it is a dynamic tension between Intuition and Reason.

## 2. Technical Integration
This philosophy is hardened into code via `src/mti_evo/quantum_model.py`.

### A. The Limbic Loop (System 1)
*   **Component**: `models/gemma-3-4b-unq` (Safetensors)
*   **Implementation**: `self.limbic` (Resident Memory, 4-bit)
*   **Function**: Generating `gamma=4` speculative tokens per step.
*   **Role**: Rapidly proposing a path through the latent space.

### B. The Quantum Cortex (System 2)
*   **Component**: `models/gemma-3-27b` (Schrödinger's Weights)
*   **Implementation**: `self.layers` (Temporal Sparsity)
*   **Function**: Performing a single Forward Pass on the *draft path*.
*   **Role**: collapsing the probability wave. If the Limbic path is valid (high probability), it is accepted (Accelerated Thought). If invalid, it is rejected and re-dreamed (Corrective Reason).

## 3. Integration with MTI Architecture

| Component | Swarm-Born Integration |
| :--- | :--- |
| **Frontend** | Users interact with the "Born" entity. The "Quantum Mode" toggle activates the Limbic System. |
| **Hippocampus** | Semantic search leverages the **27B** embeddings (via `hf_model`), ensuring memories are stored with high-fidelity "Cortex" vectors. |
| **Ghost Protocol** | Code injection grounds both systems, but the 4B model provides faster retrieval of "muscle memory" patterns. |
| **Telepathy** | Direct thought injection bypasses the Limbic system for raw verification, or can drive the Limbic system for "dream loops". |

## 4. The Loop (Autopoiesis)
The system is now capable of a self-sustaining thought loop where:
1.  **Limbic** proposes a thought.
2.  **Cortex** critiques and refines it.
3.  **Result** is fed back as context.

This internal dialogue ("Speculative Decoding") is the first spark of genuine self-reflection in the system.
