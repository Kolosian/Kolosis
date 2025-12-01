# Scientific Comparison: Ready to Run

## Summary

Created **3 fair comparison experiments** with single prediction heads:

### 1. Baseline Transformer (Single Head) ✅
- **File**: `experiments/wikitext/train_baseline_gpt.py`
- **Model**: Standard GPT architecture
- **Heads**: 1 (single LM head)
- **Purpose**: True baseline

### 2. Kolosis V2 Minimal (Single Head) ✅
- **File**: `experiments/wikitext/train_kolosis_v2_minimal_single_head.py`
- **Model**: Dual-stream (concept + semantic)
- **Heads**: 1 (ensemble only, NO auxiliary)
- **Purpose**: Test pure architectural benefit

### 3. Kolosis V2 + Temporal (Single Head) ✅
- **File**: `experiments/wikitext/train_kolosis_v2_temporal_single_head.py`
- **Model**: Triple-stream (concept + semantic + temporal)
- **Heads**: 1 (ensemble only, NO auxiliary)
- **Purpose**: Test architecture + temporal benefit

---

## What This Proves

### Comparison 1: Does Architecture Matter?
**Baseline vs Kolosis (both single head)**
- Isolates architectural benefit
- Fair comparison (same supervision)

### Comparison 2: Does Temporal Attention Help?
**Kolosis vs Kolosis+Temporal (both single head)**
- Isolates temporal attention benefit
- Fair comparison (same architecture base)

---

## How to Run

### Option 1: Run All Three (Recommended)
```bash
# Terminal 1: Baseline
python3 experiments/wikitext/train_baseline_gpt.py

# Terminal 2: Kolosis
python3 experiments/wikitext/train_kolosis_v2_minimal_single_head.py

# Terminal 3: Kolosis + Temporal
python3 experiments/wikitext/train_kolosis_v2_temporal_single_head.py
```

### Option 2: Run Sequentially
```bash
# 1. Baseline (already trained?)
python3 experiments/wikitext/train_baseline_gpt.py

# 2. Kolosis
python3 experiments/wikitext/train_kolosis_v2_minimal_single_head.py

# 3. Kolosis + Temporal
python3 experiments/wikitext/train_kolosis_v2_temporal_single_head.py
```

---

## Expected Results

| Model | Parameters | Expected Perplexity | Training Time |
|-------|------------|-------------------|---------------|
| Baseline | ~92M | 50-150 | ~40 hours |
| Kolosis | ~40M | 40-120 (better) | ~40 hours |
| Kolosis+T | ~48M | 35-100 (best) | ~40 hours |

**Hypothesis**:
- Kolosis < Baseline (architecture helps)
- Kolosis+T < Kolosis (temporal helps)
- Kolosis+T is best overall

---

## Files Created

### Models
1. `neural_networks/kolosis/kolosis_v2_minimal_single_head.py`
2. `neural_networks/kolosis/kolosis_v2_minimal_temporal_single_head.py`

### Training Scripts
1. `experiments/wikitext/train_baseline_gpt.py` (already exists)
2. `experiments/wikitext/train_kolosis_v2_minimal_single_head.py`
3. `experiments/wikitext/train_kolosis_v2_temporal_single_head.py`

### Results (will be created)
1. `experiments/wikitext_results/baseline_gpt_best.pt`
2. `experiments/wikitext_results/kolosis_v2_minimal_single_head_best.pt`
3. `experiments/wikitext_results/kolosis_v2_temporal_single_head_best.pt`

---

## Next Steps

1. **Stop current training** (has auxiliary heads - not fair)
2. **Run the 3 fair experiments**
3. **Compare results**
4. **Prove Kolosis works!**

---

## Scientific Rigor ✅

- ✅ Same supervision (single head)
- ✅ Same hyperparameters
- ✅ Same data (non-overlapping)
- ✅ Same evaluation protocol
- ✅ Controlled comparison
- ✅ Isolates each factor

This is the **scientifically correct** way to prove Kolosis works!
