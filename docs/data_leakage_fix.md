# Data Leakage Fix - WikiText-103 Training Scripts

## Issue Found

**Problem**: Overlapping windows in dataset creation caused data leakage.

**Original Code** (WRONG):
```python
for i in range(0, len(tokens) - block_size, block_size // 2):
    chunk = tokens[i:i + block_size + 1]
    if len(chunk) == block_size + 1:
        self.examples.append(chunk)
```

**Why This Is Wrong**:
- `block_size // 2` creates 50% overlap between consecutive windows
- Window 1: tokens [0:128]
- Window 2: tokens [64:192] ← overlaps with Window 1
- Window 3: tokens [128:256] ← overlaps with Window 2

**Impact**:
1. **Training contamination**: Model sees same tokens multiple times
2. **Validation contamination**: Validation examples overlap
3. **Inflated performance**: Artificially low loss due to memorization
4. **Invalid results**: Cannot compare to published baselines

---

## Fix Applied

**Corrected Code**:
```python
# Use non-overlapping windows to prevent data leakage
for i in range(0, len(tokens) - block_size, block_size):
    chunk = tokens[i:i + block_size + 1]
    if len(chunk) == block_size + 1:
        self.examples.append(chunk)
```

**Changes**:
- `block_size // 2` → `block_size`
- Creates non-overlapping, independent windows
- Window 1: tokens [0:128]
- Window 2: tokens [128:256] ← no overlap
- Window 3: tokens [256:384] ← no overlap

---

## Files Fixed

1. ✅ `experiments/wikitext/train_kolosis_v2_minimal_temporal.py`
2. ✅ `experiments/wikitext/train_kolosis_v2_minimal_4gb.py`
3. ✅ `experiments/wikitext/train_baseline_gpt.py`

---

## Expected Impact

### Before Fix (Overlapping):
- Training examples: ~733,444 (with 50% overlap)
- Validation examples: ~1,485 (with 50% overlap)
- **Artificially low loss** due to data leakage

### After Fix (Non-overlapping):
- Training examples: ~366,722 (50% fewer, but clean)
- Validation examples: ~742 (50% fewer, but clean)
- **Higher but honest loss** - true model performance

### Loss Expectations:

**With Overlap** (contaminated):
- Training loss: ~1.0
- Validation loss: ~1.5-2.0
- **INVALID** - not comparable to published results

**Without Overlap** (clean):
- Training loss: ~3.5-4.5
- Validation loss: ~4.0-5.0
- Perplexity: ~50-150 (typical for WikiText-103)
- **VALID** - comparable to published results

---

## What To Do Now

1. **Stop current training** - Results are contaminated
2. **Delete contaminated checkpoints**:
   ```bash
   rm experiments/wikitext_results/kolosis_v2_minimal_temporal_best.pt
   rm experiments/wikitext_results/kolosis_v2_minimal_4gb_best.pt
   rm experiments/wikitext_results/baseline_gpt_best.pt
   ```

3. **Re-run training with fixed scripts**:
   ```bash
   # Train with temporal attention (fixed)
   python3 experiments/wikitext/train_kolosis_v2_minimal_temporal.py
   ```

4. **Expect different results**:
   - Higher loss (but honest)
   - Slower convergence
   - Valid comparison to baselines

---

## Verification

To verify the fix worked, check the number of training examples:

**Before** (overlapping):
```
Created 733444 training examples  # Too many!
Created 1485 validation examples
```

**After** (non-overlapping):
```
Created ~366722 training examples  # About half
Created ~742 validation examples
```

If you see approximately **half the examples**, the fix is working correctly.

---

## Credit

Thanks to ChatGPT for catching this data leakage issue! This is a common mistake in language modeling experiments.

---

## Summary

✅ **Fixed**: All three training scripts now use non-overlapping windows
✅ **Impact**: Results will be honest but loss will be higher
✅ **Action**: Re-run training with clean data
✅ **Benefit**: Results will be comparable to published baselines
