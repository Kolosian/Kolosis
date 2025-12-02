# Kolosis V2 Minimal

**⚠️ Research Preview**: This model is part of ongoing research. Full benchmarking in progress.

## Model Description

Kolosis V2 Minimal is a cognitive multi-stream language model that separates language processing into specialized streams (Concept and Semantic), inspired by human cognition.

**Key Features:**
- 🧠 Dual cognitive streams (Concept + Semantic)
- 📊 Interpretable fusion weights
- ⚡ Efficient architecture (27M parameters)
- 🎛️ Controllable cognitive balance

## Model Details

- **Developed by**: [Your Name]
- **Model type**: Causal Language Model
- **Architecture**: Cognitive Multi-Stream Transformer
- **Parameters**: 27,417,812
- **License**: MIT
- **Language**: English
- **Training Data**: WikiText-103

## Architecture

```
Input → Hierarchical Embeddings (Symbol→Concept→Law)
    ↓
┌──────────────┬──────────────┐
│ Concept      │ Semantic     │
│ Stream       │ Stream       │
│ (Abstraction)│ (Relations)  │
└──────────────┴──────────────┘
    ↓
Learned Fusion (Weighted Combination)
    ↓
Output Predictions
```

### Streams

1. **Concept Stream**: Handles abstraction and categorization
2. **Semantic Stream**: Processes relationships and context

### Fusion Mechanism

The model learns to weight each stream's contribution:
- **Concept**: 39% (abstraction)
- **Semantic**: 61% (relationships)

*These weights were learned during training, showing that relationship understanding is more important than abstraction for language modeling.*

## Performance

### WikiText-103

| Metric | Value |
|--------|-------|
| **Perplexity** | 49.76 |
| **Validation Loss** | 3.91 |
| **Parameters** | 27.4M |
| **Training Epochs** | 9 |

**Status**: ✅ Validated on WikiText-103

**In Progress**:
- Baseline comparison (rerunning with matched config)
- Temporal attention variant
- Multi-dataset validation

## Usage

### Installation

```bash
pip install torch transformers
```

### Basic Usage

```python
import torch
from transformers import GPT2Tokenizer

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load model (you'll need to download the architecture)
# See: https://github.com/yourusername/kolosis
from kolosis import KolosisV2MinimalSingleHead

model = KolosisV2MinimalSingleHead.from_pretrained('yourusername/kolosis-v2-minimal')

# Generate text
input_text = "The future of artificial intelligence"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

print(tokenizer.decode(output[0]))
```

### Inspect Cognitive Balance

```python
# Get fusion weights
fusion_weight = model.get_fusion_weight()
print(f"Concept: {fusion_weight:.2%}")
print(f"Semantic: {1-fusion_weight:.2%}")
```

## Training Details

### Training Data

- **Dataset**: WikiText-103
- **Tokenizer**: GPT-2 (50,257 vocab)
- **Context Length**: 128 tokens
- **Training Examples**: 504,546
- **Validation Examples**: 1,030

### Training Procedure

- **Optimizer**: AdamW
- **Learning Rate**: 3e-4
- **Batch Size**: 8
- **Epochs**: 9
- **Hardware**: NVIDIA GPU (CUDA)
- **Training Time**: ~11 hours

### Preprocessing

- Non-overlapping windows (block_size=128)
- Causal masking (no future token leakage)
- Causal semantic embeddings

## Limitations

- **Language**: English only
- **Context**: 128 tokens (relatively short)
- **Scale**: 27M parameters (small by modern standards)
- **Training Data**: WikiText-103 only (limited domain)
- **Research Preview**: Full validation in progress

## Bias and Ethical Considerations

This model was trained on WikiText-103, which is derived from Wikipedia. It may reflect biases present in the training data, including but not limited to:
- Gender bias
- Cultural bias
- Historical bias
- Representation bias

**Use responsibly** and be aware of potential biases in generated text.

## Citation

```bibtex
@software{kolosis2024,
  title={Kolosis: Cognitive Multi-Stream Architecture for Efficient Language Modeling},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/kolosis}
}
```

## More Information

- **GitHub**: https://github.com/yourusername/kolosis
- **Paper**: *In preparation*
- **Documentation**: See GitHub repository

## Contact

- **Issues**: https://github.com/yourusername/kolosis/issues
- **Email**: your.email@example.com

---

**Model Version**: v0.1.0 (Research Preview)  
**Last Updated**: December 2024
