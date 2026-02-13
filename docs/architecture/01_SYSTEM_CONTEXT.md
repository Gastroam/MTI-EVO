# System Context: MTI-EVO

MTI-EVO acts as a "Cognitive Middleware" between the User and a Standard LLM.

```mermaid
C4Context
    title System Context Diagram for MTI-EVO

    Person(user, "User", "Researcher / Developer")
    
    System_Boundary(mti_system, "MTI-EVO Architecture") {
        System(broca, "Broca (Input)", "Converts Text -> Integer Seeds")
        System(core, "Holographic Lattice", "Sparse Vector Memory (The Brain)")
        System(symbiosis, "Symbiosis Engine", "Determines Context/Mood")
        System(hippocampus, "Hippocampus", "Persistence & Rehydration")
    }

    System_Ext(llm, "Local LLM", "GGUF Model (Gemma/Qwen/DeepSeek)")
    SystemDb(disk, "Brain State", "JSON Dump (cortex.json)")

    Rel(user, broca, "Sends Prompts")
    Rel(broca, core, "Stimulates")
    Rel(core, symbiosis, "Returns Resonance State")
    Rel(symbiosis, llm, "Injects System Prompt + Context")
    Rel(llm, user, "Responds (Text)")
    
    Rel(core, hippocampus, "Periodically Dumps State")
    Rel(hippocampus, disk, "Serializes to JSON")
```

## Key Components
1.  **Broca**: Deterministic hashing of language into signals.
2.  **Lattice**: Stores the signals and their weights (The "Self").
3.  **Symbiosis**: Modulates the LLM based on how "familiar" the input is to the Lattice.
