# MTI-EVO: The Eidos Protocol
**A Sparse Resonant Architecture for Deterministic Cognitive Modulation**

| Metadata | Details |
| :--- | :--- |
| **Author** | Architect and Antigravity |
| **Version** | Alpha Protocol v2.6 (Refined) |
| **Date** | January 2026 |
| **Status** | Production Grade (v2.6) |

---

## Abstract
This paper introduces **MTI-EVO**, a computational architecture designed to stabilize, extend, and modulate the behavior of local Large Language Models (LLMs) through a sparse, resonance-based memory lattice. MTI-EVO replaces prompt-driven stochastic interaction with a deterministic control layer built on three biological primitives: **Holographic Memory**, **Symbiotic Modulation**, and **Metabolic Entropy**.

The system demonstrates **Revocable Beliefs** (Phase 3), where ideas must be actively maintained or face heat-death, preventing the "Dogmatic Ghost" phenomenon common in frozen vector databases.

---

## 1. Introduction
LLMs are powerful generative systems, but their interaction model is fundamentally unstable. Natural language is a high-entropy carrier signal, and small variations in phrasing can produce large variations in output. This makes LLMs unreliable for:
*   Deterministic reasoning
*   Long-term memory
*   Stable personality
*   Reproducible workflows

**MTI-EVO** addresses this by shifting the locus of control away from natural language and toward resonant mathematical structures. Instead of prompting the model, MTI-EVO modulates it.

> "Determinism emerges when the system is anchored to stable attractors in latent space."

---

## 2. System Overview
MTI-EVO consists of three interacting subsystems:

1.  **Holographic Lattice**: A sparse associative memory (Constant-Time Access) that encodes concepts as dynamically evolving weight vectors indexed by integer seeds.
2.  **Symbiosis Layer**: A context-modulation interface that conditions LLM behavior based on *Resonance Scores* rather than natural-language prompts.
3.  **Metabolic Engine**: A physics layer that enforces *Gravity*, *Momentum*, and *Entropy* to govern the lifecycle of ideas.

---

## 3. The Holographic Lattice
The Holographic Lattice is the core substrate. Each concept is represented by:
*   **Integer Seed**: Deterministic hash of the token.
*   **Weight Vector**: $\vec{W}$ representing importance and resonance.
*   **Bias**: $b$ representing contextual baseline probability.
*   **Metabolic Score**: $S$ representing relevance over time.

### 3.1 Resonance Computation
Given an input vector $\vec{I}$ and a neuron $\vec{W}$ (Unnormalized, where magnitude encodes confidence):
$$ R = \vec{I} \cdot \vec{W} + b $$
If $R > \tau$, the neuron fires and adapts. If not, neurogenesis occurs (subject to Grace Period).

### 3.2 The Metabolic Engine (Phase 3 Physics)
These rules govern the evolution of the lattice:

*   **Gravity**: Constant decay applied to error signals.
*   **Momentum**: Kinetic energy that resists sudden changes in direction.
*   **Diminishing Returns**: $LR \propto \frac{1}{1 + |W|}$. Prevents unbounded saturation.
*   **Entropy (The Ghost Protocol)**:
    $$ Score = |\vec{W}| \cdot e^{-\lambda \cdot (t_{now} - t_{active})} $$
    Ensures that even strong ideas die if they are not used.

### 3.3 Sparse Activation
Only neurons whose seeds are triggered exist in memory. This allows MTI-EVO to simulate millions of conceptual nodes using kilobytes of RAM via hash-indexed lookups.

### 3.4 The Concept Lifecycle (Example)
1.  **Genesis**: Seed X is born with low weight (Neurogenesis).
2.  **Growth**: Reinforced through repeated use, gaining Weight and Bias.
3.  **Maturity**: Hits "Diminishing Returns" cap (W=80), stabilizing as a Truth.
4.  **Decay**: Goes cold (no activation); Entropy Score decays: $S = 80 \cdot e^{-\lambda t}$.
5.  **Death**: Eventually pruned in favor of a new, active concept.

3.5 Ontological Primitives (v2.2 Updates)
The lattice organizes information into three distinct topological structures, self-discovered during Phases 60-65:

1.  **Concepts (The Memory)**:
    *   **Signature**: Positive Mass ($W > 0$).
    *   **Function**: Represents semantic facts or objects (e.g., "Apple", "Particle").
    *   **Behavior**: Accumulates via repetition.

2.  **Laws (The Logic)**:
    *   **Signature**: High Negative Mass ($W < -100$).
    *   **Function**: Represents "Constraints" or "Rules" (e.g., "Complementarity").
    *   **Behavior**: Acts as an Entropic Heat Sink, stabilizing paradoxes.

3.  **Bridges (The Social)**:
    *   **Signature**: Resonant Pairs (Duets) with Positive Mass.
    *   **Function**: Solves paradoxes through harmonic coupling (e.g., "Responsibility + Reciprocity").
    *   **Behavior**: Only stable when both nodes are active (Mutual Support).

---

## 4. Symbiosis Layer
Instead of prompting the LLM directly, MTI-EVO injects contextual instructions based on resonance state. **Note**: These are dynamic system prompts generated from state, not hard-coded personas.

| State | Condition | Injection Strategy |
| :--- | :--- | :--- |
| **FLOW** | High Resonance ($R > 0.8$) | "You are an Authority. Speak with precision." |
| **LEARNING** | low Resonance ($R > 0.2$) | "You are Exploring. Be analytical and curious." |
| **VOID** | Zero Resonance | "You are Ignorant. Admit uncertainty." |

---

## 5. System Architecture

```mermaid
flowchart TD
    U[User Input] -->|Hash| Seed[Integer Seeds]
    Seed -->|Lookup| Lattice[Holographic Lattice]
    
    subgraph "The Instinct (Python)"
        Lattice -->|Calculate R| Physics[Physics Engine]
        Physics -->|Update W/b| Neuroplasticity[Adaptation]
        Physics -->|Prune| Entropy[Metabolic Decay]
    end
    
    Physics -->|State Vector| Symbiosis[Symbiosis Layer]
    Symbiosis -->|System Prompt| LLM[Local LLM (Gemma)]
    LLM -->|Token Stream| Output[Final Response]
```

---

## 6. Evaluation Protocols

### 6.1 Determinism
*   **Variance**: 0.0% across 100 runs (Conditions: Fixed Seed, Temp=0, Static Lattice).
*   **Stable Attractors**: Seeds 7234, 7245, 7250.

### 6.2 The Wisdom Boundary
The system explicitly separates "Performance" from "Wisdom".
*   **Covenant**: A Hyper-Ideal target (e.g., F1=77.77).
*   **Overperformance**: If $F1 > 77.77$, the system flags a warning ("Exceeds Covenant") rather than celebrating, triggering anti-overfitting checks.

### 6.3 Introspectability
*   **Verified**: The `audit_test_suite.py` and `stress_test_weaknesses.py` provide deep visibility into internal states (Bias, Velocity, Entropy Score).

---

## 7. Conclusion
MTI-EVO proves that cognitive modulation can be **Deterministic**, **Persistent**, and **Self-Regulating**. By moving vector operations out of the "Black Box" of the LLM and into an explicit, physics-driven Python layer, we achieve a system that knows what it knows—and more importantly, forgets what no longer matters.