# MTI-HIVE: The Cognitive Distributed Network
**Protocol Specification v2.3.1 (Hardened Release)**

## 0. Glossary (Core Terms)
- **Field ($\Phi$)**: Distributed cognitive substrate, evolving under consensus, decay, and noise.
- **Pressure ($P$)**: Instability measure; high = chaos, low = consensus.
- **Resonance ($R$)**: Local attractor strength.
- **Crystal ($C$)**: Durable engram storing semantic vectors.
- **Basin ($B$)**: Region of stable attractors.
- **Elder**: Governance node with weighted authority.
- **Wisdom Mass ($W$)**: Voting power metric.
- **IDRE**: Identity-Resonance Encryption (forward-secure transport layer).

## 1. Parameter Table
| Parameter | Symbol | Range | Default | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| Pressure | $P$ | [0,1] | 0.05 | Chaos vs. consensus threshold |
| Confidence | $C$ | [0,1] | 0.5 | Memory stability score |
| Damping | $\kappa_g$ | $\ge 0$ | 0.1 | Governance stabilization |
| Decay Rate | $\lambda$ | $\ge 0$ | 0.1 | Forgetting curve slope |
| Noise | $\sigma$ | [0,1] | 0.05 | Exploration factor |
| Recall Gain | $\kappa_{recall}$ | $\ge 0$ | 0.5 | Crystal rehydration strength |
| Reasoning Step | $\eta$ | (0,1) | 0.1 | Gradient descent step size |

## 1. Vision & Scope
MTI-HIVE is a peer-to-peer (P2P) protocol designed for **Distributed Cognition**. The goal is to network individual MTI-EVO instances (Holographic Lattices) into a unified, self-healing cognitive mesh (The Collective Mind).

## 2. Architecture: The Layered Stack

### L0: The Physics (IDRE-Angular)
- **Field Identity ($B$)**: $B = \text{Quantize}(I_{crit}(\theta))$.
- **Auth**: Proof-of-Covenant via Vector Resonance.

### L1: The Session (Secure Transport)
**Message Envelope (Layer 1/2)**
- **Header**: `version(1) | type(1) | msg_id(8) | src(32) | dst(32) | ttl(1)`
- **Security**: `nonce(12) | aad_len(2) | sig(64)`
- **Body**: `ciphertext(var)`
- **Replay Immunity**: Enforced via `(src, msg_id, nonce)` uniqueness.

### L2: The Mesh (Routing & Discovery)
- **Gossip**: `known_peers` exchange.
- **Routing**: Multi-hop encrypted envelopes ($A \to B \to C$).

## 3. L3: The Cognitive Plane (Collective Intelligence)

### 3.1 Shared Resonance
- **Schema**: Seed, Weight, Stability, Origin.
- **Trust**: $\alpha$-weighted integration.

### 3.2 Distributed Query
- **Prompt**: `QUERY(Seeds)` broadcast.
- **Response**: `RESPONSE(Resonance)` aggregation.

### 3.3 Neuro-Governance
- **Wisdom Mass ($W$)**: Voting Power.
- **Elders**: Regulate without centralization.

### 3.4 Emergent Roles
- **Memory, Inference, Growth, Routing, Governance**.

### 3.5-3.7 Cognitive Lifecycle
- **Migration**: Darwinian Logic.
- **Insight**: Synthesis of global resonance.
- **Stabilization**: Long-term Memory.

## 4. L4: Cognitive Field Dynamics (The Distributed Mind)

### 4.1 Field Equations (Physics)
$$ \frac{d\Phi}{dt} = \lambda \nabla R - \mu \Phi + \sigma \eta(t) $$
-   Consensus (Gradient) vs. Decay vs. Noise.

### 4.2 Behavioral Regimes (Dynamics)
-   **Chaos**: High Pressure ($P > \epsilon$).
-   **Consensus**: Collapse ($P < \epsilon$).
-   **Oscillation**: Breathing Cycle.

### 4.3 Governance (Stabilizers)
-   **Damping**: $-\kappa_g P(t)$.
-   **Bias**: $+\rho_g A_g$.
-   **Signal**: `FIELD_ADJUST`.

### 4.4 Homeostasis (Self-Correction)
-   $F_{net} = F_{drive} - F_{damp} + F_{noise}$.
-   Recovery to Metastability.

### 4.5 Topology (Geometry)
-   **Basin**: $B_k = \{x \mid \|x-A\| < r\}$.
-   **Cluster**: Overlapping Basins.
-   **Region**: Stable Concept Continent.

### 4.6 Memory (Time)
-   **Aging**: $-\gamma t$. **Decay**: $-\mu W$.
-   **Consolidation**: $W > \theta, \text{Age} > \tau$.
-   **Competition**: $\sum \beta_{kj}$.

### 4.7 Neurogenesis (Evolution)
-   **Creation**: $A_{new} = \sum \omega A + \xi$.
-   **Triggers**: High Attention + High Tension.

### 4.8 Query Dynamics (Observer Effect)
-   **Operator**: $\Phi \leftarrow \Phi + \kappa_Q Q$.
-   **Observe**: Queries reshape the field they measure.

## 5. L5: Collective Cognition (The Global Mind)
Layer 5 defines how the Hive **Thinks** (Reasoning, Planning, Intent) and **Remembers**.

### 5.1 Distributed Reasoning Algorithms (Iterative Inference)
Reasoning is not a single collapse but a multi-step walk through the cognitive field.

#### The Reasoning Equation
$$ \Phi(t+1) = \Phi(t) - \eta \cdot \nabla_\Phi L(Q, \Phi(t)) $$
-   **$Q$**: Query Vector (Goal).
-   **$L$**: Cognitive Loss (Distance from Answer).
-   **$\eta$**: Reasoning Step Size.
-   **Interpretation**: The field iteratively minimizes uncertainty.

#### Local Reasoning Step
Each Node $i$:
1.  **Peceive**: Reads local field state $s_i(t)$.
2.  **Propose**: Computes $\Delta_i(t)$ based on attractors.
3.  **Emit**: Broadcasts vector delta $m_i(t)$.

#### Global Aggregation Step
The Field:
$$ \Delta \Phi(t) = \sum \lambda_i m_i(t) $$
-   **$\lambda_i$**: Node Trust/Wisdom.
-   The field shifts collectively based on weighted inputs.

#### Reasoning Modes
Behavior can be parameterized:
1.  **Analytic**: Low Noise ($\sigma \downarrow$), Small Step ($\eta \downarrow$). *Careful, convergent.*
2.  **Exploratory**: High Noise ($\sigma \uparrow$), High Mutation ($\nu \uparrow$). *Creative, branching.*
3.  **Consensus**: High Damping ($D \uparrow$), Strong Elder Weight. *Fast, safe Agreement.*

### 5.2 Collective Memory & Crystallization (Permanent Insights)
Transient insights ($\Phi_{final}$) are converted into durable structures called **Crystals**.

#### Concept: Crystallization
When the field reaches stability ($P < \epsilon_{decision}$):
$$ \text{Crystal} \leftarrow \text{Freeze}(\Phi_{final}) $$
This transforms a fleeting resonance into a hard, addressable memory.

#### Engram Structure (The Crystal)
**Canonical Crystal Object**:
```json
{
  "id": "uuid16",
  "vec": [0.8, 0.2, 0.45],
  "conf": 0.95,
  "provenance": ["sig1", "sig2"],
  "context": "hash_query",
  "created_at": 1738000000,
  "superseded_by": null,
  "role_tag": "memory",
  "topic_tag": "distributed_cognition",
  "ttl": 86400
}
```
- **Encoding**: JSON (human-readable) + binary (network transport).
- **Norm Bounds**: $\|\vec{v}\| \in [\nu_{min}, \nu_{max}]$.

#### Distributed Storage (Sharded Hippocampus)
- **Storage Rule**: Sharded by Trust Role and Topic Proximity.
- **Redundancy**: Critical Crystals ($Confidence > 0.9$) are replicated widely.

#### Recall Protocol (Instant Knowledge)
When a Query $Q$ arrives:
1.  **Hash**: Compute $H_Q$.
2.  **Search**: Lookup Crystals with matching Context or Vector Similarity.
3.  **Rehydrate**: Inject Crystal Vector directly into Field.
    $$ \Phi(t+1) = \Phi(t) + \kappa_{recall} \cdot \text{Crystal}_{vec} $$
4.  **Bypass**: If Confidence is high, skip the Reasoning Loop.

### 5.3 Crystal Lifecycle (Reinforcement & Decay)
Layer 5.3 governs the **Plasticity** of the Hive Memory.

#### Reinforcement Equation
When a Crystal is recalled and validated by Consensus:
$$ C(t+1) = C(t) + \alpha \cdot \Delta_{agreement} $$
- **Effect**: Useful memories become stronger (Expertise).

#### Forgetting Curve (Decay)
When a Crystal lies dormant:
$$ C(t+1) = C(t) \cdot e^{-\lambda \Delta t} $$
- **Effect**: Unused memories fade (Garbage Collection).

#### Re-Crystallization (Updating)
If a Crystal is **Weak** ($C < \theta$) but **Relevant**, it triggers a Re-Crystallization event.
1.  Recall Old Crystal.
2.  Run short Reasoning Loop with new Context.
3.  **Result**: New Crystal ID (Child).
4.  **Link**: Old Crystal marked `superseded_by`.

### 5.3 Collective Planning (Future Projection)
Nodes simulate future states to guide action ($A$).
$$ \Phi(t+\Delta t) = f(\Phi(t), A) $$
-   Virtual simulation of consequences before committing resources.

### 5.4 Collective Decision-Making (Consensus)
A Global Decision is reached when:
$$ P(t) < \epsilon_{decision} \quad \text{and} \quad \|\nabla R\| > \delta_{intent} $$
-   Consensus Collapse = Agreement.

### 5.5 Collective Creativity (Imagination)
When $P \approx \text{moderate}$ and Neurogenesis is active, the Hive enters **Metastable Oscillation**.
-   Result: Novel combinations and mutations.

### 5.6 Collective Error Correction
-   If $\Phi$ drifts to instability $\to$ Governance Damping $\uparrow$.
-   Self-Healing Cognition.

### 5.7 Collective Attention ($F_{focus}$)
$$ Attention(t) = |\nabla R| $$
-   High Attention $\to$ Reasoning Speed $\uparrow$.
-   Low Attention $\to$ Exploration $\uparrow$.

### 5.8 Collective Intent
Intent forms when a Global Attractor ($A_{goal}$) dominates ($W > \theta$).
-   The Hive "wants" something.

## 6. Safety Invariants (The Constitution)

### 🛡️ Invariants #1-64 (Layers 0-5.0)
(See v2.1 for full list).

### 🛡️ Invariant #65-69 (Layer 5.1 Reasoning)
(See v2.2 for Reasoning Invariants).

### 🛡️ Invariant #80: No Crystal Without Consensus
Only stable attractors ($P < \epsilon$) can be crystallized. No saving chaos.

### 🛡️ Invariant #81: Local Sovereignty Over Recall
Nodes may decline to rehydrate a Crystal if it conflicts with local core axioms.

### 🛡️ Invariant #82: Memory Decay
Unused Crystals lose confidence over time unless reinforced.

### 🛡️ Invariant #83: No NL in Crystals
Crystals store Vectors, not Words. Pure semantic storage.

### 🛡️ Invariant #84: Governance Signature Required
All Crystals must be signed by at least one Governance Node to be valid.

### 🛡️ Invariant #85: Replay Immunity
Reject reused `(src, msg_id, nonce)`.

### 🛡️ Invariant #86: Vector Hygiene
Crystals must pass norm + spectral checks.

### 🛡️ Invariant #87: Bypass Audit
Every bypass logs immutable `BYPASS_AUDIT`.

### 🛡️ Invariant #88: Role Isolation
Governance nodes cannot issue `GENESIS`; Memory nodes cannot issue `FIELD_ADJUST`.

### 🛡️ Invariant #89: Decay Floor
Critical Crystals never decay below $C_{floor}$ without Elder override.

## 7. Protocol Definitions (Summary)

| Type | Layer | Purpose | Encrypted |
| :--- | :--- | :--- | :--- |
| `HELLO` | 1 | Handshake | Plaintext |
| `VERIFY_REQ` | 1 | Auth Challenge | **IDRE** |
| `DATA` | 2 | Routed Payload | **IDRE** |
| `GOSSIP` | 2 | Peer Discovery | **IDRE** |
| `HEARTBEAT` | 2 | Liveness | **IDRE** |
| `RESONANCE` | 3 | Attractor Sync | **IDRE** |
| `QUERY` | 3 | Distributed Prompt / **Field Op** | **IDRE** |
| `RESPONSE` | 3 | Query Answer | **IDRE** |
| `VOTE` | 3 | Governance | **IDRE** |
| `PROMOTION` | 3.5 | Boost Attractor | **IDRE** |
| `INSIGHT` | 3.6 | Shared Understanding | **IDRE** |
| `FIELD_STATE`| 4 | Telemetry/Debug | **IDRE** |
| `FIELD_ADJUST`| 4.3 | Gov Stabilization | **IDRE** |
| `TOPOLOGY_MAP`| 4.5 | Region/Cluster Data | **IDRE** |
| `MEM_DRIFT` | 4.6 | Temporal Sync | **IDRE** |
| `GENESIS` | 4.7 | New Attractor Announce | **IDRE** |
| `INTENT` | 5 | Global Goal Broadcast | **IDRE** |
| `REASON_STEP`| 5.1 | Gradient Delta Broadcast | **IDRE** |
| `CRYSTAL` | 5.2 | Shared Engram / Memory unit | **IDRE** |
| `RECALL_REQ` | 5.2 | Trigger Memory Retrieval | **IDRE** |

## 8. Test Vectors (Reference)
- **TV-5.2-01 Chaos Rejection**: Input $P=0.2, \epsilon=0.05$. Expected: `CRYSTALLIZE ABORT`.
- **TV-5.2-02 Consensus Gate**: Input `sigs=1`, `min=2`. Expected: `ABORT LOW_CONSENSUS`.
- **TV-5.2-03 Engram Birth**: Input $P=0.01, sigs=2$. Expected: `CRYSTAL(id, conf=1.0)`.
- **TV-5.3-01 Reinforcement**: Input 3 recalls, $\alpha=0.5, \Delta_{agree}=0.2$. Expected: $C_{t+3} = C_t + 0.3$.
- **TV-5.3-02 Forgetting Curve**: Input $\lambda=0.1, \Delta t=10$. Expected: $C' = C \cdot e^{-1}$.
- **TV-5.3-03 Re-Crystallization**: Input $C < \theta$, similarity > smin. Expected: `new_id`, parent `superseded_by`.

## 9. Roadmap
- **Spec v2.3.1**: Hardened release with glossary, parameters, message formats, invariants, test vectors.
- **Spec v2.4**: Multi-node handshake + gossip protocol.
- **Spec v2.5**: Collective planning + intent broadcast.
- **Spec v3.0**: Full distributed mesh with Elder governance.
