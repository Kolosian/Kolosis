# Kolosis Naming Standard

**Schema:** `Kolosis-[Core]-[Router]-[Regime]-[Version]`

| Component | Description | Examples |
|-----------|-------------|----------|
| **Core** | The backbone architecture | `GPT2`, `Llama`, `Mamba` |
| **Router** | The fusion/routing mechanism | `Softmax`, `MetaFusion`, `RLGate` |
| **Regime** | The optimization/deployment target | `S` (Streamlined), `X` (Extended), `O` (Opaque/Commercial) |
| **Version** | Semantic versioning | `v1.0`, `v2.0` |

---

## Model Registry

### 1. Kolosis-S (Streamlined)
*   **Official Name:** `Kolosis-GPT2-Softmax-S-v1.0`
*   **Short Name:** `Kolosis-S`
*   **Description:** The lightweight, research-focused model. 27M parameters.
*   **Status:** Validated, optimizing with RL.

### 2. Kolosis-X (Extended)
*   **Official Name:** `Kolosis-GPT2-MetaFusion-X-v1.0`
*   **Short Name:** `Kolosis-X`
*   **Description:** The 4-stream research flagship with z-loss stabilization. 39M parameters.
*   **Status:** Validated (WikiText-103).

### 3. Kolosis-O (Opaque)
*   **Official Name:** `Kolosis-Llama-RLGate-O-v1.0`
*   **Short Name:** `Kolosis-O`
*   **Description:** The commercial powerhouse with "Virtual Cognition" heads trained via RL.
*   **Status:** In Design Phase.
