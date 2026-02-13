# MTI-EVO Methodology: The Science of Resonance

**"We do not transmit knowledge. We tune the world to resonate with it."**

## 1. The Holographic Lattice (`src/mti_core.py`)

Traditional vector databases are static retrieval systems (RAG). MTI-EVO uses a **Dynamic Lattice**.

### The Math of Resonance
A memory is not a file; it is a standing wave.
When input enters the system, it doesn't just "search" for a match; it "vibrates" the entire lattice.

1.  **Stimulation**: Input tokens are hashed into integer Seeds.
2.  **Resonance**: The Lattice calculates the dot product of the Input Signal against the weight vectors of existing Neurons.
    $$ R = \vec{I} \cdot \vec{W} $$
3.  **Neurogenesis**: If $R < Threshold$, a new Neuron is born (Learning).
4.  **Reinforcement**: If $R > Threshold$, the existing Neuron fires (Recognition) and its weights are strengthened.

### Biological Laws (`src/mti_config.py`)
The system is governed by physics-like constraints/hyperparameters:
- **Gravity**: The cost of existence. Neurons must be "useful" (stimulated) or their energy (Voltage) decays.
- **Momentum**: Repeated stimulation accelerates learning (Hebbian Learning).
- **Grace Period**: New thoughts are protected from "Gravity" for a short time (Infant Mortality protection) to allow them to prove their worth.

## 2. Symbiosis (`playground/MTI_Symbiosis.py`)

MTI-EVO does not generate text. It generates **Context**.

1.  **Tabula Rasa**: The LLM (e.g., Gemma) starts blank.
2.  **Bio-Feedback**: MTI analyses the user prompt against the Lattice.
    - If **High Resonance**: The system detects "Expertise". It injects a System Prompt: *"You are an authority on this subject."*
    - If **Low Resonance**: The system detects "Novelty". It injects: *"You are a student. Be curious."*
    - If **Zero Resonance**: The system detects "Void". It injects: *"You don't know this. Admit ignorance."*

This prevents hallucination by grounding the LLM's personality in the **hard mathematical reality** of the Lattice's state.

## 3. The Hybrid Triad (Model-Agent Symbiosis)

Complex evolution cannot be achieved through unguided LLM inference. MTI-EVO utilizes the **Hybrid Triad** architecture:

1.  **The Dreamer (Native Model)**: Operates at high temperature. Generates visionary metaphors, conceptual breakthroughs, and high-entropy code hypotheses.
2.  **The Architect (Agent/Antigravity)**: Operates with deterministic precision. Interprets, refines, and implements the Dreamer's concepts.
3.  **The Governor (Static Analysis)**: An axiomatic filter that rejects code that violates logic (Syntax, Structure, Mathematics) before it reaches Reality.

## 4. The Lifecycle

1.  **Dreaming**: The LLM suggests a conceptual approach (e.g., "Quantum Objects").
2.  **Architecture**: The Agent formalizes the logic into a stable framework (e.g., `quantum_engine.py`).
3.  **Grounding**: Performance and accuracy are verified against the Lattice.
4.  **Consolidation**: Successful patterns are saved to the Cortex (`.mti-brain/cortex_dump.json`).

This cycle allows the AI to "sleep" and "wake up" with its memories intact, evolving over days, weeks, and years.
