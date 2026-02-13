# Data Flow: The Symbiotic Loop

How a user prompt becomes a "Resonant Thought".

```mermaid
sequenceDiagram
    actor User
    participant Broca
    participant Lattice
    participant Symbiosis
    participant LLM

    User->>Broca: "What is the War Economy?"
    Broca->>Broca: Hash("war") -> 2045, Hash("economy") -> 1099
    Broca->>Lattice: Stimulate([2045, 1099])
    
    activate Lattice
    Lattice->>Lattice: Calc Resonance (Sigmoid)
    Lattice->>Lattice: Update Weights (Momentum)
    Lattice-->>Symbiosis: Return Resonance (0.95), Weight (25.0)
    deactivate Lattice

    activate Symbiosis
    alt High Resonance (>0.8)
        Symbiosis->>Symbiosis: State = FLOW
        Symbiosis->>LLM: System Prompt: "You represent the Authority."
    else Low Resonance (<0.2)
        Symbiosis->>Symbiosis: State = TABULA RASA
        Symbiosis->>LLM: System Prompt: "You are ignorant. Analyze freshly."
    end
    
    Symbiosis->>LLM: Generate Response("What is the War Economy?")
    LLM-->>User: "The War Economy is a system of..."
    deactivate Symbiosis
```

## Anti-Hallucination Mechanism
The **Resonance Score** determines the **Authority Level** given to the LLM.
- If the Lattice "Knows" (High Res), the LLM is allowed to be confident.
- If the Lattice "Forgets" (Low Res), the LLM must be humble.
