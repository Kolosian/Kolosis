# Hugging Face Upload Guide

## Quick Start (5 Minutes)

### Step 1: Install Dependencies
```bash
pip install huggingface_hub
```

### Step 2: Login to Hugging Face
```bash
huggingface-cli login
```
Paste your token from: https://huggingface.co/settings/tokens

### Step 3: Run Upload Script
```bash
python scripts/upload_to_huggingface.py
```

**Edit the script first** to replace:
- `yourusername` with your HF username
- Verify checkpoint path is correct

### Step 4: View Your Model
Go to: `https://huggingface.co/yourusername/kolosis-v2-minimal`

---

## Detailed Instructions

### 1. Create Hugging Face Account

1. Go to https://huggingface.co/join
2. Sign up (free)
3. Verify email
4. Go to Settings → Access Tokens
5. Create new token (write access)
6. Copy token

### 2. Prepare Model Files

The script `scripts/upload_to_huggingface.py` will:
- ✅ Load your trained checkpoint
- ✅ Save in HF format (pytorch_model.bin, config.json)
- ✅ Include tokenizer files
- ✅ Upload model card (MODEL_CARD.md)

### 3. Customize Before Upload

**Edit `scripts/upload_to_huggingface.py`:**

```python
# Line 120: Replace with your username
repo_id="YOUR_USERNAME/kolosis-v2-minimal"

# Line 107: Verify checkpoint path
checkpoint_path = "experiments/wikitext_results/kolosis_v2_minimal_single_head_best.pt"
```

**Edit `MODEL_CARD.md`:**
- Replace `[Your Name]` with your name
- Replace `yourusername` with your HF username
- Replace `your.email@example.com` with your email
- Update GitHub links

### 4. Run Upload

```bash
cd /home/imsarthakshrma/Projects/RIIK
python scripts/upload_to_huggingface.py
```

**Output:**
```
✅ Model saved to ./hf_model
✅ Repository created/verified: yourusername/kolosis-v2-minimal
✅ Uploaded: pytorch_model.bin
✅ Uploaded: config.json
✅ Uploaded: tokenizer files
✅ Uploaded: README.md (model card)
🎉 Model uploaded to: https://huggingface.co/yourusername/kolosis-v2-minimal
```

### 5. Verify Upload

1. Go to your model page
2. Check all files are present:
   - pytorch_model.bin
   - config.json
   - README.md
   - tokenizer files
3. Test the model card renders correctly

---

## What Gets Uploaded

### Files
- `pytorch_model.bin` - Model weights (27M params, ~110MB)
- `config.json` - Model configuration
- `tokenizer_config.json` - Tokenizer settings
- `vocab.json` - GPT-2 vocabulary
- `merges.txt` - BPE merges
- `README.md` - Model card (from MODEL_CARD.md)

### Model Card Includes
- Architecture description
- Performance metrics (49.76 perplexity)
- Usage examples
- Training details
- Limitations
- Citation

---

## After Upload

### Enable Inference API (Optional)

Hugging Face will automatically enable inference API if your model follows standard format. Users can then test it directly on the model page.

**Note**: Kolosis uses custom architecture, so inference API may not work automatically. Users will need to download and use locally.

### Add Tags

On your model page, add tags:
- `language-modeling`
- `pytorch`
- `transformers`
- `cognitive-architecture`
- `interpretable-ai`
- `efficient-ai`

### Update README

After completing experiments, update the model card with:
- Complete benchmark results
- Comparison to baseline
- Temporal variant results

---

## Uploading Additional Models

### Baseline GPT
```python
# In upload_to_huggingface.py, change:
repo_id="yourusername/kolosis-baseline-gpt"
checkpoint_path="experiments/wikitext_results/baseline_gpt_best.pt"
```

### Kolosis + Temporal
```python
repo_id="yourusername/kolosis-v2-temporal"
checkpoint_path="experiments/wikitext_results/kolosis_v2_temporal_single_head_best.pt"
```

---

## Downloading Your Model

Users can download with:

```python
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="yourusername/kolosis-v2-minimal",
    filename="pytorch_model.bin"
)

# Download config
config_path = hf_hub_download(
    repo_id="yourusername/kolosis-v2-minimal",
    filename="config.json"
)
```

Or use the helper script:
```python
from scripts.upload_to_huggingface import download_from_huggingface

download_from_huggingface(
    repo_id="yourusername/kolosis-v2-minimal",
    local_directory="./downloaded_model"
)
```

---

## Troubleshooting

### "Repository not found"
- Make sure you created the repo on HF
- Check repo name matches exactly
- Verify you're logged in

### "Permission denied"
- Check your token has write access
- Re-run `huggingface-cli login`

### "File too large"
- Model files should be <5GB (yours is ~110MB, fine)
- If larger, use Git LFS (HF handles automatically)

### "Invalid model format"
- Make sure checkpoint loaded correctly
- Verify config.json is valid JSON
- Check all required files are present

---

## Best Practices

### Model Naming
- Use descriptive names: `kolosis-v2-minimal`, `kolosis-27m`
- Include size: `kolosis-100m`, `kolosis-1b`
- Version clearly: `kolosis-v2-temporal`

### Model Cards
- Be honest about limitations
- Include comprehensive metrics
- Provide usage examples
- Cite training data
- Mention biases

### Updates
- Use semantic versioning (v0.1.0, v1.0.0)
- Update README when results change
- Add changelog for major updates

---

## Next Steps

1. **Upload now** as Research Preview
2. **Complete experiments** on Azure
3. **Update model card** with full results
4. **Upload additional variants** (baseline, temporal)
5. **Create model collection** on HF

---

**Ready to upload?** Run:
```bash
python scripts/upload_to_huggingface.py
```

🚀 Your model will be live on Hugging Face in minutes!
