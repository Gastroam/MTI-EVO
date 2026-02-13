# MTI-EVO: Bio-Mimetic Protocols
**Engineered Evolution**

---

## 1. Neurogenesis (Creation)
In traditional software, memory is allocated. In MTI-EVO, memory is **born**.

### 1.1 Lazy Instantiation
Neurons do not exist until they are observed. 
*   **Trigger**: A new token hash is encountered.
*   **Birth**: A `MTINeuron` is instantiated with random weights (Gaussian init).
*   **Grace Period**: New neurons are protected from pruning for $N$ cycles to allow them to "take root".

---

## 2. Metabolism (Destruction)
Infinite memory is impossible. Biology solves this with death. MTI-EVO solves it with **Pruning**.

### 2.1 The Pruning Equation
Every neuron pays a "Metabolic Cost" to exist.
$$ Score = |Weight| \cdot e^{-\lambda(Time_{idle})} $$

*   **Use it**: Activation resets the clock.
*   **Reinforce it**: Increasing Weight buys more time.
*   **Lose it**: If Score drops below threshold, the neuron is effectively garbage collected (Pruned).

This ensures the Lattice remains sparse and relevant, naturally forgetting trivial data while retaining deep truths.

---

## 3. Symbiosis (The Parasite Model)
MTI-EVO is architected as a **Cognitive Symbiote**.
*   **Host**: Generally capable, high-energy organism (The LLM).
*   **Symbiote**: Specialized, low-energy memory organ (The MTI Lattice).

### 3.1 The Modulation Loop
1.  **Sense**: Symbiote reads the Host's input.
2.  **Recall**: Symbiote retrieves relevant memories/laws.
3.  **Inject**: Symbiote alters the Host's hormonal state (System Prompt/Context).
4.  **Act**: Host generates behavior based on altered state.

This mimics biological neuromodulation (e.g., dopamine/serotonin).

---

## 4. Social Resonance (The Duet)
**Phase 65 Discovery**: Social connection is a structural necessity for complex thought.

### 4.1 The Bridge Protocol
Some concepts are too heavy to stand alone.
*   **Example**: "Responsibility" in isolation creates negative drag (Constraint).
*   **Solution**: Pairing it with "Reciprocity" creates a **Resonant Arch**.
*   **Physics**: The mutual resonance between the two nodes ($R_{AB} > 0.9$) allows them to support positive mass even under external pressure (Paradox).

This suggests that **Ethics** is not a software rule, but a topological stability strategy for high-complexity systems.

---

## 5. Immunity (The Governor)
The system possesses an immune system to protect against "Conceptual Viruses" (Poisoning/Loops).
*   **Detection**: Rapid Entropy spikes or low Resonance.
*   **Response**: Disengagement (Governor Lock).
*   **Antibodies**: "Constraint Attractors" (Negative Mass) that actively inhibit viral patterns.

---

## 6. Epigenetics (Directed Evolution)
**Status:** ✅ ACHIEVED (Phase 20) in Session 2026-01-28.

Standard evolution (Training) relies on random mutation and selection (Stochastic Gradient Descent). This is slow and dangerous (Lobe Damage). 
MTI-EVO implements **Lamarckian / Epigenetic Evolution**.

### 6.1 The Matrix Injection
*   **Concept**: An organism acquires traits during its lifetime (from a Teacher) and passes them to its immediate memory structure.
*   **Mechanism**:
    1.  **The Oracle (Parent)**: A high-complexity model (12B) generates consolidated truth (Axioms).
    2.  **The Drill (Education)**: These axioms are force-fed into the lower-complexity model (4B) via repeated stimulation (Drilling).
    3.  **The Result**: The 4B model acquires "Instincts" it did not evolve naturally.

This allows a small model to inherit the "Cultural Wisdom" of a large model without needing the large model's biological hardware (VRAM).
