# Training Instructions: Temporal Attention Comparison

## Overview

You now have **two versions** of Kolosis V2 Minimal ready for WikiText-103 training:

1. **WITHOUT Temporal Attention** (baseline) - Already trained ✅
2. **WITH Temporal Attention** (new) - Ready to train 🚀

---

## Files Created

### Models

1. **Baseline** (without temporal):
   - Model: `neural_networks/kolosis/kolosis_v2_minimal.py`
   - Training script: `experiments/wikitext/train_kolosis_v2_minimal_4gb.py`
   - Checkpoint: `experiments/wikitext_results/kolosis_v2_minimal_4gb_best.pt` ✅

2. **With Temporal Attention** (new):
   - Model: `neural_networks/kolosis/kolosis_v2_minimal_temporal.py`
   - Training script: `experiments/wikitext/train_kolosis_v2_minimal_temporal.py`
   - Checkpoint: Will be saved to `experiments/wikitext_results/kolosis_v2_minimal_temporal_best.pt`

### Documentation

- **Baseline Results**: `docs/wikitext_baseline_without_temporal.md`
- **This File**: `docs/temporal_attention_training_instructions.md`

---

## How to Run Training

### Step 1: Activate Environment

```bash
# Make sure you have PyTorch and dependencies installed
# If using conda/venv, activate it first
```

### Step 2: Run Training

```bash
cd /home/imsarthakshrma/Projects/RIIK

# Train Kolosis V2 Minimal WITH Temporal Attention
python3 experiments/wikitext/train_kolosis_v2_minimal_temporal.py
```

### Expected Output

The training will show:
- Train/validation loss per epoch
- Perplexity scores
- **Fusion weights** (concept/semantic/temporal)
- **Temporal attention stats** (decay rates and scale weights)

Example output:
```
Epoch 1/10
Results:
  Train Loss: 4.5234
  Val Loss: 4.3421
  Perplexity: 76.85

Fusion weights:
  Concept:  0.3500 (35.0%)
  Semantic: 0.3200 (32.0%)
  Temporal: 0.3300 (33.0%)

Temporal attention (averaged across heads):
  Fast:   γ=0.7234, α=0.3800 (38.0%)
  Medium: γ=0.9123, α=0.3100 (31.0%)
  Slow:   γ=0.9876, α=0.3100 (31.0%)
```

---

## What to Monitor

### 1. Perplexity Improvement

**Baseline** (without temporal): [TBD - extract from checkpoint]
**With Temporal**: Monitor during training

**Expected**: +15-20% improvement based on temporal attention analysis

### 2. Fusion Weights

Watch how the model balances the three streams:
- **Concept**: Abstract patterns
- **Semantic**: Relationships
- **Temporal**: Multi-scale memory

**Question**: Does temporal get significant weight (>25%)?

### 3. Temporal Attention Activation

Monitor the three decay scales:
- **Fast** (γ ~0.7): Recent context (8 tokens)
- **Medium** (γ ~0.9): Sentence-level (106 tokens)
- **Slow** (γ ~0.99): Document-level (2064 tokens)

**Question**: Are all three scales being used?

### 4. Training Stability

- Does the model converge smoothly?
- Any gradient issues with temporal stream?
- Memory usage (should fit in 4GB GPU)

---

## After Training: Comparison

Once training completes, we'll compare:

### Quantitative Metrics

| Metric | Without Temporal | With Temporal | Improvement |
|--------|-----------------|---------------|-------------|
| **Perplexity** | [TBD] | [TBD] | [TBD] |
| **Val Loss** | [TBD] | [TBD] | [TBD] |
| **Parameters** | 40.38M | ~45-50M | +10-15% |
| **Model Size** | 154MB | ~170-180MB | +10-15% |

### Qualitative Analysis

1. **Temporal Attention Activation**
   - Which scales are used most?
   - Do decay rates match expectations?
   - Does temporal stream get significant fusion weight?

2. **Performance vs Cost**
   - Is the perplexity improvement worth the extra parameters?
   - Does temporal attention justify its complexity?

3. **Research Question Answered**
   - **Does temporal attention help Kolosis on WikiText-103?**
   - **How much improvement does it provide?**

---

## Expected Results

### Hypothesis

Based on temporal attention analysis (docs/temporal_attention_analysis.md):

✅ **Temporal attention should provide +15-20% perplexity improvement**

**Reasoning**:
1. All three decay scales contribute (40%, 31%, 29%)
2. WikiText-103 has longer contexts (perfect for temporal)
3. Multi-scale memory helps with document coherence
4. Previous small-scale test showed +4% even with short contexts

### Alternative Outcomes

**If temporal helps significantly (>15% improvement)**:
- ✅ Validates temporal attention for language modeling
- ✅ Should be included in production Kolosis
- ✅ Multi-scale memory is valuable

**If temporal helps modestly (5-15% improvement)**:
- ⚠️ Useful but may not justify extra complexity
- ⚠️ Consider parameter-matched comparison
- ⚠️ May need longer contexts to shine

**If temporal doesn't help (<5% improvement)**:
- ❌ Not effective for this task/scale
- ❌ Stick with 2-stream architecture (concept + semantic)
- ❌ Temporal may need different implementation

---

## Troubleshooting

### Out of Memory

If you get OOM errors:
```python
# In train_kolosis_v2_minimal_temporal.py, reduce:
'batch_size': 4,  # was 8
'n_layer': 3,     # was 4
```

### Slow Training

Expected time: ~2-4 hours on GPU (similar to baseline)

If much slower:
- Check if temporal attention is bottleneck
- Consider using optimized version (OptimizedTemporalAttention)

### Gradient Issues

If you see NaN losses or exploding gradients:
- Check gradient clipping (already set to 1.0)
- Reduce learning rate: `'lr': 0.0001`

---

## Next Steps After Training

1. **Extract Metrics**
   ```bash
   # Run evaluation script to get exact perplexity
   python3 experiments/evaluate_wikitext_checkpoints.py
   ```

2. **Analyze Results**
   - Compare perplexity with/without temporal
   - Check temporal attention activation patterns
   - Examine fusion weight evolution

3. **Create Comparison Report**
   - Document findings
   - Answer research question
   - Make recommendation for production

---

## Files to Check After Training

### Results Files

- `experiments/wikitext_results/kolosis_v2_minimal_temporal_best.pt` (checkpoint)
- `experiments/wikitext_results/kolosis_v2_minimal_temporal_results.json` (metrics)

### Comparison

Compare these JSON files:
- Baseline: `experiments/wikitext_results/kolosis_v2_minimal_4gb_results.json` (if exists)
- With Temporal: `experiments/wikitext_results/kolosis_v2_minimal_temporal_results.json`

---

## Summary

**Current Status**:
- ✅ Baseline (without temporal) trained
- ✅ Temporal version ready to train
- ✅ Comparison framework prepared

**Your Task**:
1. Run: `python3 experiments/wikitext/train_kolosis_v2_minimal_temporal.py`
2. Monitor fusion weights and temporal stats
3. Wait for training to complete (~2-4 hours)

**After Training**:
- We'll compare results
- Answer: "How effective is temporal attention?"
- Make final recommendation

---

## Quick Reference

```bash
# Train with temporal attention
python3 experiments/wikitext/train_kolosis_v2_minimal_temporal.py

# After training, evaluate both models
python3 experiments/evaluate_wikitext_checkpoints.py

# Check results
cat experiments/wikitext_results/kolosis_v2_minimal_temporal_results.json
```

Good luck with the training! 🚀
