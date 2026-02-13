# The HIVE: Multi-Expert MTI-EVO Architecture

**Goal:** Transform MTI-EVO from a monolithic agent into a neuro-symbolic ensemble of specialized experts (The Hive), governed by strict IDRE protocols and orchestrated by an Intent Router.

## 1. Expert Profiles (The Council)

Each expert is a specialized configuration of the underlying model (Gemma-3), optimized for specific tasks.

| Expert | Profile / Engine | Port | Input/Output | Capabilities | Telepathy Channel |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemma-Math** | **Ramanujan Engine** | 8766 | Raw Logic -> Proofs | `infer`, `no_context` | **Yes** (Strict Caps) |
| **Gemma-Dreams** | **Oneiric Analyzer** | 8800 | Narrative -> Clusters | `embed`, `dream` | **Opcional** (Batch) |
| **Gemma-Code** | **Ghost Auditor** | 8800 | Codebase -> Diagnosis | `read_fs`, `analyze` | **Parcial** (Read-Only) |
| **Gemma-Consensus**| **Hive Node** | 8766 | Election -> Verdict | `vote`, `aggregator` | **Yes** (Meta-Infer) |
| **Gemma-Director** | **Visual Auteur** | 8766 | Script -> Visuals/FFMPEG | `ffmpeg`, `edit` | **Yes** (Strict) |
| **Gemma-Scribe** | **Master Storyteller** | 8766 | Idea -> Screenplay | `write`, `format` | **Yes** (Creative) |

## 2. Orchestration Layer

### A. Intent Router (L1)
*   **Function**: Intercepts prompts at the entry point.
*   **Logic**: Parses `intent` block from client or infers from prompt content.
*   **Routing**: Dispatches request to the appropriate Expert Profile.
    *   *Example*: `purpose="math"` -> Routes to **Gemma-Math**.

### B. IDRE Governance (L2)
*   **Identity**: Validates "Who is asking?" (Local Process, Nonce).
*   **Intent**: Validates "What do they want?" (Operation, Scope).
*   **Risk**: Dynamic scoring specific to the Expert.
    *   *Math Profile*: High tolerance for symbols, Zero tolerance for system paths.
    *   *Code Profile*: High tolerance for paths, Zero tolerance for `exec`.
*   **Evidence**: Cryptographic log of the transaction.

### C. Consensus Layer (L3 - Optional)
*   **Function**: Resolution of ambiguity.
*   **Mechanism**: If `Confidence < Threshold`, query 2+ experts.
*   **Aggregation**: Weighted voting based on expert domain relevance.

## 3. Implementation Strategy

### Resource Management (VRAM Constraints)
*   **Virtual Experts**: Instead of multiple heavy processes, we use **Logical Profiles**.
*   **Context Switching**: The `LLMAdapter` dynamically swaps:
    *   **System Prompts**: "You are a pure logic engine..." vs "You are a dream weaver..."
    *   **Parameter Sets**: Temperature (0.0 for Math, 0.8 for Dreams), Top-K, etc.
    *   **Tool Access**: Restricted via software gates.
*   **Reasoning**: Running 4x 4B models requires ~24GB+ VRAM. Logical switching allows running on consumer hardware (8GB-12GB) with minimal latency (kv-cache swapping).

### Shared Memory (Hippocampus)
*   **Embeddings DB**: A unified vector store accessed by all experts.
*   **Transient Cache**: Hash-based caching of prompt/response pairs to speed up Consensus checks.

## 4. Evaluation Metrics
*   **Math**: Logical output validity, Step-by-step coherence.
*   **Dreams**: Cluster purity (HDBScan noise ratio), UMAP stability.
*   **Code**: Reference accuracy (Ghost Index match rate).
*   **Consensus**: Inter-annotator agreement score.
