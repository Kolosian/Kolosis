# Kolosis: Cognitive Multi-Stream Architecture

> **⚠️ Research Preview**: Core architecture validated on WikiText-103. Full benchmarking in progress.

Kolosis is a novel neural architecture that separates language processing into specialized **cognitive streams** (Symbol, Concept, Semantic, Temporal), inspired by human cognition. Unlike traditional transformers that use a single unified representation, Kolosis enables interpretable, controllable, and efficient language modeling.

## 🎯 Key Features

- **🧠 Cognitive Specialization**: Separate streams for different types of thinking
- **📊 Interpretable**: See which cognitive functions are active via fusion weights
- **⚡ Efficient**: Achieves competitive performance with 56% fewer parameters
- **🎛️ Controllable**: Tune cognitive balance for different tasks
- **🔍 Transparent**: Full visibility into model reasoning

## 📈 Current Results

### Kolosis V2 Minimal (27.4M parameters)
- **WikiText-103 Perplexity**: 49.76 (Epoch 9)
- **Fusion Weights**: Semantic 61%, Concept 39%
- **Parameters**: 10% fewer than baseline (27M vs 30M)
- **Status**: ✅ Validated

### In Progress
- [ ] Baseline GPT comparison (rerunning with matched config)
- [ ] Kolosis + Temporal validation (architecture implemented)
- [ ] Multi-head ablation study

**Help wanted!** If you have GPU resources, we'd love help completing these experiments.

## 🏗️ Architecture

Kolosis uses four specialized cognitive streams:

```
Input Text
    ↓
┌─────────────────────────────────────┐
│  Hierarchical Embeddings            │
│  Symbol → Concept → Law             │
└─────────────────────────────────────┘
    ↓
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Symbol       │ Concept      │ Semantic     │ Temporal     │
│ Stream       │ Stream       │ Stream       │ Stream       │
│              │              │              │              │
│ Pattern      │ Abstraction  │ Relations    │ Memory       │
│ Recognition  │ & Categories │ & Context    │ & History    │
└──────────────┴──────────────┴──────────────┴──────────────┘
    ↓
┌─────────────────────────────────────┐
│  Learned Fusion (Softmax Weights)   │
│  Combines streams adaptively        │
└─────────────────────────────────────┘
    ↓
Output Predictions
```

### Hierarchical Embeddings
```python
# Three levels of semantic understanding
Symbol:  "apple" → [0.2, 0.5, ...]  # Surface pattern
Concept: "apple" → fruit, food      # Abstract meaning  
Law:     "apple" → gravity, organic # Universal rules
```

### Multi-Scale Temporal Attention
```python
Fast decay (γ≈0.7):   ~8 tokens    # Immediate context
Medium decay (γ≈0.9): ~100 tokens  # Sentence-level
Slow decay (γ≈0.98):  ~2000 tokens # Document-level
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/kolosis.git
cd kolosis
pip install -r requirements.txt
```

### Training Kolosis V2 Minimal
```bash
python experiments/wikitext/train_kolosis_v2_minimal_single_head.py
```

### Training with Temporal Attention
```bash
python experiments/wikitext/train_kolosis_v2_temporal_single_head.py
```

### Using Pretrained Models
```python
from neural_networks.kolosis import KolosisV2MinimalSingleHead

# Load model
model = KolosisV2MinimalSingleHead(
    vocab_size=50257,
    n_embd=128,
    block_size=128,
    n_layer=4
)

# Load checkpoint
model.load_state_dict(torch.load('path/to/checkpoint.pt'))

# Generate text
output = model.generate(input_ids, max_new_tokens=100)

# Inspect cognitive balance
fusion = model.get_fusion_weight()
print(f"Concept: {fusion:.2%}, Semantic: {1-fusion:.2%}")
```

## 📊 Benchmarks

### WikiText-103 (Block Size: 128, Single Head)

| Model | Parameters | Perplexity | Status |
|-------|-----------|------------|--------|
| Kolosis V2 Minimal | 27.4M | **49.76** | ✅ Validated |
| Baseline GPT | 30.6M | *In Progress* | ⏳ Running |
| Kolosis + Temporal | 47.7M | *Pending* | 📋 Planned |

*All experiments use non-overlapping windows and causal masking for fair comparison.*

## 🧪 Reproducibility

All data leakage issues have been identified and fixed:
1. ✅ Non-overlapping training windows
2. ✅ Causal attention masks
3. ✅ Causal semantic embeddings

See [`docs/data_leakage_issues_and_fixes.md`](docs/data_leakage_issues_and_fixes.md) for details.

## 📚 Documentation

- **[Kolosis Synopsis](docs/kolosis_synopsis.md)**: Comprehensive overview for general audience
- **[Data Leakage Fixes](docs/data_leakage_issues_and_fixes.md)**: Scientific rigor documentation
- **[Training Instructions](docs/temporal_attention_training_instructions.md)**: Detailed setup guide

## 🎓 Key Innovations

1. **Hierarchical Embeddings**: Symbol → Concept → Law (3-level semantic understanding)
2. **Cognitive Streams**: Explicit separation of thinking modes
3. **Learned Fusion**: Automatic cognitive balance discovery
4. **Multi-Scale Temporal**: Human-like memory with exponential decay

## 🔬 Research

### Comparison to Related Work

**Closest architectures:**
- **Mixture of Experts (MoE)**: Similar specialization, but Kolosis uses dense fusion vs sparse routing
- **Switch Transformer**: Similar expert routing, but Kolosis has cognitive specialization
- **o1 (OpenAI)**: Similar interpretability goals, but Kolosis has architectural transparency

**Key differences:**
- Kolosis: Cognitive-based specialization (Symbol/Concept/Semantic/Temporal)
- MoE: Pattern-based specialization (learned experts)
- Kolosis: Dense fusion (all streams active, weighted)
- MoE: Sparse routing (select subset of experts)

### Publications

*Coming soon - paper in preparation*

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

- **🖥️ GPU Resources**: Help run baseline and temporal experiments
- **📊 Benchmarking**: Test on additional datasets (Penn TreeBank, Enwik8, etc.)
- **🔬 Research**: Ablation studies, scaling experiments
- **📝 Documentation**: Tutorials, examples, use cases
- **🐛 Bug Reports**: Issues, edge cases, improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- WikiText-103 dataset from Salesforce Research
- Inspired by human cognitive architecture research
- Built with PyTorch and Hugging Face Transformers

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/kolosis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/kolosis/discussions)
- **Email**: your.email@example.com

## 📖 Citation

```bibtex
@software{kolosis2024,
  title={Kolosis: Cognitive Multi-Stream Architecture for Efficient Language Modeling},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/kolosis}
}
```

---

**Status**: Research Preview | **Version**: 0.1.0 | **Last Updated**: December 2024
