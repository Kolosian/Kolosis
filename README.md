# Kolosis: Multi-Stream Interpretable Language Models

> **⚠️ Status**: Kolosis-X validated (2% PPL overhead). RL Optimization phase in progress.

**Kolosis** is a novel neural architecture that explicitly separates language processing into specialized "cognitive streams"—**Temporal, Semantic, Conceptual, and Causal**—to create smaller, more interpretable, and adaptable language models.

Unlike standard Transformers that are monolithic "black boxes," Kolosis routes every token to a specialized expert stream. This provides:
1.  **Interpretability:** We know *why* the model predicted a token (e.g., "Temporal stream active" = relying on sequence history).
2.  **Modularity:** Fine-tune just one stream (e.g., "Medical Concepts") without catastrophic forgetting.
3.  **Efficiency:** Competitive perplexity (within 2% of baseline) with 30-40% fewer active parameters per token.

---

## 🚀 Key Innovations

### 1. The 4-Stream Cognitive Architecture
Instead of one massive feed-forward network, Kolosis uses 4 specialized paths:
*   **Temporal Stream:** Multi-scale attention for sequence history (like "memory").
*   **Semantic Stream:** Token-token relationship modeling.
*   **Concept Stream:** Hierarchical embeddings (Symbol → Concept → Law).
*   **Causal Stream:** Explicit cause-effect modeling.

### 2. Z-Loss Router Stabilization
We solved the "MoE Collapse" problem using **Z-Loss Regularization** (inspired by ST-MoE), which keeps the router stable without forcing artificial uniformity. This allows streams to naturally specialize based on their strengths.

### 3. RL-Based Self-Correction (New!)
We are currently pioneering a **Reinforcement Learning** approach to train the router. Instead of just minimizing prediction error, the router "learns from experience" to pick the stream that *would have* minimized loss, effectively self-optimizing its own brain.

---

## 📊 Current Benchmarks (WikiText-103)

| Model | Parameters | Val Perplexity | Overhead vs Baseline | Status |
|-------|------------|----------------|----------------------|--------|
| **Baseline GPT** | 30.6M | 59.32 | — | Reference |
| **Kolosis-X** | 39.5M | **60.64** | +2.2% | ✅ Validated |
| **Kolosis-S** | 27.4M | ~65.00* | +9.5% | 🔄 Optimizing |

*\*Kolosis-S is the smaller, streamlined variant currently under RL optimization.*

---

## 🛠️ Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/kolosis.git
cd kolosis
pip install -r requirements.txt
```

### 2. Train Kolosis-X (The Flagship)
```bash
python experiments/wikitext/train_kolosis_x_colab.py
```
*Note: This script includes Z-loss regularization and dual-optimizer setup by default.*

### 3. Run RL Router Optimization (Kolosis-S)
We have a specialized kit to improve the router's decision making:
1.  **Diagnose:** `python experiments/wikitext/measure_router_optimality_s.py`
2.  **Optimize:** `python experiments/wikitext/train_router_rl_s.py`

---

## 📚 Documentation & Journey

*   **[Development Journey](docs/Updates/kolosis_development_journey.md):** The full story of failures, pivots, and breakthroughs (Read this first!).
*   **[Kolosis-X Design](docs/Kolosis-X/Kolosis-X.md):** Deep dive into the architecture.
*   **[Optimization Plan](rl_router_optimization_plan.md):** The roadmap for our current RL experiments.

---

## 🤝 Contributing

We are looking for collaborators to help with:
1.  **Scaling:** Test Kolosis on OpenWebText or Pile subsets.
2.  **Visualization:** Build tools to visualize stream activation in real-time.
3.  **Downstream Tasks:** Validate performance on GLUE/SuperGLUE.

## 📄 License
MIT License.

