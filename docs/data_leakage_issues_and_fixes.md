# Data Leakage Issues and Fixes: WikiText-103 Training

## Overview

During the setup of fair comparison experiments for Kolosis V2 Minimal on WikiText-103, we discovered **three critical data leakage issues** that were artificially inflating model performance. This document details each issue, how it was detected, and the fixes applied.

---

## Issue 1: Overlapping Training Windows

### Problem

**File**: `experiments/wikitext/train_baseline_gpt.py`, `train_kolosis_v2_minimal_4gb.py`, `train_kolosis_v2_minimal_temporal.py`

**Symptom**: Suspiciously low validation loss (~1.0) and high number of training examples (733k instead of expected ~250k).

**Root Cause**:
```python
# WRONG: Creates overlapping windows
for i in range(0, len(tokens) - block_size, block_size // 2):
    chunk = tokens[i:i + block_size + 1]
```

This created windows with 50% overlap:
- Window 1: tokens [0:128]
- Window 2: tokens [64:192]  ← 64 tokens overlap with Window 1
- Window 3: tokens [128:256] ← 64 tokens overlap with Window 2

**Impact**:
- Training examples contained duplicate subsequences
- Validation set had sequences seen during training
- Model could "memorize" overlaps instead of learning patterns
- Artificially low perplexity (~2.7 instead of realistic ~50-150)

### Fix

Changed to non-overlapping windows:
```python
# CORRECT: Non-overlapping windows
for i in range(0, len(tokens) - block_size, block_size):
    chunk = tokens[i:i + block_size + 1]
```

**Results**:
- Training examples reduced from 733k → 504k (31% reduction, as expected)
- Validation examples reduced from 1,485 → 1,030 (31% reduction)
- Clean, non-overlapping data splits

**Files Fixed**:
- `experiments/wikitext/train_baseline_gpt.py` (line 90)
- `experiments/wikitext/train_kolosis_v2_minimal_4gb.py` (line 41)
- `experiments/wikitext/train_kolosis_v2_minimal_temporal.py` (line 41)

---

## Issue 2: Bidirectional Attention (Future Token Leakage)

### Problem

**Files**: `experiments/wikitext/train_baseline_gpt.py`, `neural_networks/kolosis/kolosis_v2_minimal_single_head.py`, `neural_networks/kolosis/kolosis_v2_minimal_temporal_single_head.py`

**Symptom**: Loss of 1.41 at 27% of first epoch (perplexity ~4.1) - impossibly good for causal language modeling.

**Root Cause**:
```python
# WRONG: Bidirectional attention (like BERT)
x = self.blocks(x)  # No mask provided
```

`nn.TransformerEncoderLayer` defaults to **bidirectional attention** when no mask is provided. This means:
- Token at position `t` could attend to tokens at positions `t+1`, `t+2`, etc.
- The model could "see the future" during training
- This is appropriate for BERT-style models, but **NOT** for causal language models like GPT

**Impact**:
- Model was cheating by looking at future tokens
- Loss was artificially low (~1.4 instead of realistic ~4-5 in early training)
- Results were invalid for language modeling evaluation

### Fix

Added explicit causal mask to all attention layers:

```python
# CORRECT: Causal attention mask
mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)

# Pass mask to each transformer layer
for block in self.blocks:
    x = block(x, src_mask=mask)
```

**Mask Structure**:
```
Position:  0    1    2
    0:   [ 0  -inf -inf]  ← Position 0 can only see itself
    1:   [ 0    0  -inf]  ← Position 1 can see 0 and 1
    2:   [ 0    0    0 ]  ← Position 2 can see 0, 1, and 2
```

**Files Fixed**:
- `experiments/wikitext/train_baseline_gpt.py` (lines 58-65)
- `neural_networks/kolosis/kolosis_v2_minimal_single_head.py` (lines 133-143)
- `neural_networks/kolosis/kolosis_v2_minimal_temporal_single_head.py` (lines 153-163)

**Note**: The temporal attention module (`MultiScaleTemporalAttention`) already had correct causal masking via `self.tril` buffer, so it did not need fixing.

---

## Issue 3: Semantic Stream Lookahead

### Problem

**Files**: `neural_networks/kolosis/kolosis_v2_minimal_single_head.py`, `neural_networks/kolosis/kolosis_v2_minimal_temporal_single_head.py`

**Symptom**: Loss of 1.13 at 3% of first epoch (perplexity ~3.1) - even lower than Issue 2, indicating additional leakage.

**Root Cause**:
```python
# WRONG: Position t receives information about t+1
if T > 1:
    pairs = torch.cat([base_emb[:, :-1], base_emb[:, 1:]], dim=-1)
    relation_features = self.relation_encoder(pairs)
    relation_features = F.pad(relation_features, (0, 0, 0, 1))  # Pad at END
    return base_emb + relation_features
```

This created pairs `(emb[t], emb[t+1])` and assigned them to position `t`:
- Position 0 received `rel(emb[0], emb[1])` ← knows about token 1!
- Position 1 received `rel(emb[1], emb[2])` ← knows about token 2!
- Position `t` received `rel(emb[t], emb[t+1])` ← knows about token t+1!

**Impact**:
- Even with causal attention masks, the **embeddings themselves** contained future information
- The model could predict token `t+1` by simply reading its own embedding at position `t`
- This is a subtle but critical form of data leakage
- Loss was impossibly low (1.13) because the answer was "baked into" the input

### Fix

Shifted relationship features to be strictly causal:

```python
# CORRECT: Position t receives information about (t-1, t)
if T > 1:
    pairs = torch.cat([base_emb[:, :-1], base_emb[:, 1:]], dim=-1)
    relation_features = self.relation_encoder(pairs)
    # Pad at START to shift features: pos[1] gets rel(0,1)
    relation_features = F.pad(relation_features, (0, 0, 1, 0))  # Pad at START
    return base_emb + relation_features
```

Now the assignments are:
- Position 0 receives `0` (padding) ← no relationship info
- Position 1 receives `rel(emb[0], emb[1])` ← knows about 0→1 transition
- Position `t` receives `rel(emb[t-1], emb[t])` ← knows about (t-1)→t transition

**Key Insight**: `F.pad(tensor, (0, 0, 1, 0))` pads **before** the sequence (top padding), while `F.pad(tensor, (0, 0, 0, 1))` pads **after** (bottom padding).

**Files Fixed**:
- `neural_networks/kolosis/kolosis_v2_minimal_single_head.py` (lines 110-113)
- `neural_networks/kolosis/kolosis_v2_minimal_temporal_single_head.py` (lines 135-138)

---

## Verification

### Before Fixes (Contaminated Data)
- **Training examples**: 733,444 (overlapping)
- **Validation examples**: 1,485 (overlapping)
- **Loss at 27% Epoch 1**: 1.41 (bidirectional attention)
- **Loss at 3% Epoch 1**: 1.13 (semantic lookahead)
- **Perplexity**: 3-4 (impossibly low)

### After All Fixes (Clean Data)
- **Training examples**: 504,546 (non-overlapping) ✅
- **Validation examples**: 1,030 (non-overlapping) ✅
- **Loss at 39% Epoch 1**: 4.72 (realistic) ✅
- **Perplexity**: ~112 (realistic for early training) ✅

---

## Lessons Learned

### 1. Overlapping Windows Are Subtle
The `block_size // 2` step seemed reasonable for "data augmentation" but created train/val contamination. Always use non-overlapping windows for clean evaluation.

### 2. Default Attention Is Bidirectional
`nn.TransformerEncoderLayer` is designed for BERT-style models (bidirectional). For GPT-style models (causal), you **must** provide an explicit causal mask.

### 3. Embeddings Can Leak Information
Even with perfect attention masking, if embeddings contain future information (via lookahead relationships), the model can still cheat. **All** components must respect causality.

### 4. Suspiciously Good Results Are Red Flags
- Loss < 2.0 in early training on WikiText-103 → likely leakage
- Perplexity < 10 in first epoch → likely leakage
- Always sanity-check against published baselines

### 5. Multiple Leaks Can Compound
We had **three independent leaks**:
1. Overlapping windows (data-level)
2. Bidirectional attention (architecture-level)
3. Semantic lookahead (embedding-level)

Each needed to be fixed independently. The final loss (4.72) is realistic only after **all three** were addressed.

---

## Impact on Results

### Invalid Results (Before Fixes)
Any results from training runs before 2025-11-26 13:00 UTC are **invalid** and should be discarded:
- `baseline_gpt_best.pt` (if trained before fixes)
- `kolosis_v2_minimal_best.pt` (if trained before fixes)
- `kolosis_v2_minimal_temporal_best.pt` (if trained before fixes)

### Valid Results (After Fixes)
Training runs started after 2025-11-26 13:00 UTC with the fixed code are valid for publication and comparison.

---

## Files Modified

### Training Scripts
1. `experiments/wikitext/train_baseline_gpt.py`
   - Line 90: Non-overlapping windows
   - Lines 58-65: Causal attention mask

2. `experiments/wikitext/train_kolosis_v2_minimal_4gb.py`
   - Line 41: Non-overlapping windows

3. `experiments/wikitext/train_kolosis_v2_minimal_temporal.py`
   - Line 41: Non-overlapping windows

4. `experiments/wikitext/train_kolosis_v2_minimal_single_head.py`
   - Imports fixed model (no direct changes)

5. `experiments/wikitext/train_kolosis_v2_temporal_single_head.py`
   - Imports fixed model (no direct changes)

### Model Files
1. `neural_networks/kolosis/kolosis_v2_minimal_single_head.py`
   - Lines 133-143: Causal attention mask for concept/semantic streams
   - Lines 110-113: Causal semantic embedding (pad at start)

2. `neural_networks/kolosis/kolosis_v2_minimal_temporal_single_head.py`
   - Lines 153-163: Causal attention mask for concept/semantic streams
   - Lines 135-138: Causal semantic embedding (pad at start)

### Documentation
1. `docs/data_leakage_fix.md` (previous version, now superseded)
2. `docs/data_leakage_issues_and_fixes.md` (this document)

---

## Checklist for Future Experiments

When setting up new language modeling experiments, verify:

- [ ] **Data**: Non-overlapping windows (step = block_size, not block_size // 2)
- [ ] **Attention**: Causal mask provided to all `TransformerEncoderLayer` instances
- [ ] **Embeddings**: No lookahead in positional, hierarchical, or relational embeddings
- [ ] **Sanity Check**: Initial loss should be ~log(vocab_size) ≈ 10.8 for GPT-2 vocab
- [ ] **Validation**: Perplexity should be realistic (50-150 for WikiText-103 in early epochs)
- [ ] **Comparison**: Results should be comparable to published baselines

---

## References

- PyTorch `nn.TransformerEncoderLayer` docs: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
- Causal masking in transformers: https://arxiv.org/abs/1706.03762 (Attention Is All You Need)
- WikiText-103 dataset: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-26  
**Status**: All issues resolved, training validated
