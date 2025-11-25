# KOLOSIS Deployment Guide for Cloud Training

## Overview

This guide explains how to train KOLOSIS models on cloud platforms (Google Colab, AWS, Azure, etc.) with larger GPUs and custom datasets for English conversations, scientific text, and mathematics.

---

## Quick Start: Current 4GB GPU Test

**Running now**:
```bash
./venv/bin/python experiments/wikitext/train_kolosis_v2_minimal_4gb.py
```

This validates the architecture on your local 4GB GPU. Once complete, you're ready for cloud deployment.

---

## Cloud Deployment Options

### Option 1: Google Colab (Recommended for Quick Tests)

**Pros**: Free GPU (T4 16GB), easy setup, Jupyter notebooks
**Cons**: Session limits (12h), may disconnect

**Setup**:
```python
# 1. Upload your project to Google Drive
# 2. In Colab notebook:

from google.colab import drive
drive.mount('/content/drive')

# Clone or copy your project
!git clone https://github.com/yourusername/RIIK.git
cd RIIK

# Install dependencies
!pip install torch transformers datasets tqdm

# Run training
!python experiments/wikitext/train_kolosis_v2_minimal.py
```

### Option 2: AWS SageMaker / EC2

**Pros**: Scalable, persistent, professional
**Cons**: Costs money, more setup

**Recommended Instance**: `p3.2xlarge` (V100 16GB, ~$3/hour)

### Option 3: Lambda Labs / RunPod

**Pros**: Cheap GPU rental ($0.50-1/hour), simple
**Cons**: Less integrated than AWS

---

## Preparing Your Custom Datasets

### 1. English Conversations Dataset

**Format**: JSON with conversation structure
```json
{
  "conversations": [
    {
      "messages": [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"}
      ]
    }
  ]
}
```

**Preprocessing Script**: `experiments/prepare_conversation_data.py`
```python
import json
from transformers import GPT2Tokenizer

def prepare_conversations(json_path, output_path, block_size=256):
    """Convert conversations to training format"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    examples = []
    for conv in data['conversations']:
        # Concatenate conversation
        text = ""
        for msg in conv['messages']:
            text += f"{msg['role']}: {msg['content']}\n"
        
        # Tokenize
        tokens = tokenizer.encode(text, max_length=2048, truncation=True)
        
        # Create windows
        for i in range(0, len(tokens) - block_size, block_size // 2):
            chunk = tokens[i:i + block_size + 1]
            if len(chunk) == block_size + 1:
                examples.append(chunk)
    
    # Save
    torch.save(examples, output_path)
    print(f"Created {len(examples)} training examples")

# Usage
prepare_conversations('data/conversations.json', 'data/conversations_train.pt')
```

### 2. Scientific Text Dataset

**Format**: Plain text files or JSON
```json
{
  "papers": [
    {
      "title": "Quantum Computing Basics",
      "abstract": "...",
      "content": "..."
    }
  ]
}
```

**Key**: Use longer context (512+ tokens) for scientific text to capture complex reasoning.

### 3. Mathematics Dataset

**Format**: Problem-solution pairs
```json
{
  "problems": [
    {
      "problem": "Solve: 2x + 5 = 13",
      "solution": "x = 4",
      "steps": ["2x = 8", "x = 4"]
    }
  ]
}
```

**Special consideration**: Math benefits from **temporal attention** (tracking multi-step reasoning).

---

## Training Configuration for Different Datasets

### For Conversations (16GB GPU)
```python
config = {
    'vocab_size': 50257,
    'n_embd': 256,
    'block_size': 256,      # Conversations are shorter
    'n_layer': 6,
    'dropout': 0.1,
    'batch_size': 32,
    'epochs': 10,
    'lr': 0.0003
}
```

### For Scientific Text (16GB GPU)
```python
config = {
    'vocab_size': 50257,
    'n_embd': 256,
    'block_size': 512,      # Longer context for complex reasoning
    'n_layer': 8,           # Deeper for scientific understanding
    'dropout': 0.1,
    'batch_size': 16,       # Smaller batch due to longer context
    'epochs': 15,
    'lr': 0.0002
}
```

### For Mathematics (16GB GPU)
```python
config = {
    'vocab_size': 50257,
    'n_embd': 256,
    'block_size': 384,      # Medium context for step-by-step
    'n_layer': 6,
    'dropout': 0.05,        # Less dropout for precise reasoning
    'batch_size': 24,
    'epochs': 20,
    'lr': 0.0001            # Lower LR for stability
}
```

### For Large GPU (A100 40GB)
```python
config = {
    'vocab_size': 50257,
    'n_embd': 512,          # 2x larger embeddings
    'block_size': 1024,     # 4x longer context
    'n_layer': 12,          # 2x deeper
    'dropout': 0.1,
    'batch_size': 64,       # 2x larger batches
    'epochs': 10,
    'lr': 0.0003
}
```

---

## Step-by-Step Cloud Training

### 1. Prepare Your Data

```bash
# On your local machine
# 1. Collect your datasets
mkdir -p data/custom
# - conversations.json
# - scientific_papers.json
# - math_problems.json

# 2. Upload to cloud storage
# Google Drive, S3, or include in git repo
```

### 2. Set Up Cloud Instance

**Google Colab**:
```python
# In Colab notebook
!git clone https://github.com/yourusername/RIIK.git
%cd RIIK
!pip install -r requirements.txt

# Mount Drive for data
from google.colab import drive
drive.mount('/content/drive')

# Copy data
!cp /content/drive/MyDrive/datasets/* data/custom/
```

**AWS EC2**:
```bash
# SSH into instance
ssh -i key.pem ubuntu@ec2-instance

# Clone repo
git clone https://github.com/yourusername/RIIK.git
cd RIIK

# Install dependencies
pip install -r requirements.txt

# Download data from S3
aws s3 cp s3://your-bucket/datasets/ data/custom/ --recursive
```

### 3. Create Custom Training Script

**`experiments/train_custom_dataset.py`**:
```python
import torch
from torch.utils.data import DataLoader
from neural_networks.kolosis.kolosis_v2_minimal import KolosisV2Minimal

# Load your preprocessed data
train_data = torch.load('data/custom/conversations_train.pt')
val_data = torch.load('data/custom/conversations_val.pt')

# Create dataloaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Create model
model = KolosisV2Minimal(
    vocab_size=50257,
    n_embd=256,
    block_size=256,
    n_layer=6,
    dropout=0.1
).cuda()

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)

for epoch in range(10):
    # Training loop
    model.train()
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.cuda(), y.cuda()
            _, loss = model(x, y)
            val_loss += loss.item()
    
    print(f"Epoch {epoch}: Val Loss = {val_loss/len(val_loader):.4f}")
    
    # Save checkpoint
    torch.save(model.state_dict(), f'checkpoints/epoch_{epoch}.pt')
```

### 4. Run Training

```bash
# In tmux/screen session (so it doesn't stop if you disconnect)
tmux new -s training

# Run training
python experiments/train_custom_dataset.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

### 5. Monitor Progress

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Tail logs
tail -f training.log

# Check checkpoints
ls -lh checkpoints/
```

### 6. Download Results

**From Colab**:
```python
from google.colab import files
files.download('checkpoints/best_model.pt')
```

**From AWS**:
```bash
# On your local machine
scp -i key.pem ubuntu@ec2-instance:~/RIIK/checkpoints/*.pt ./
```

---

## Expected Training Times

| Dataset | Size | GPU | Config | Time |
|---------|------|-----|--------|------|
| Conversations | 100K examples | T4 16GB | n_embd=256, n_layer=6 | ~4 hours |
| Scientific | 50K papers | V100 16GB | n_embd=256, n_layer=8 | ~8 hours |
| Mathematics | 200K problems | A100 40GB | n_embd=512, n_layer=12 | ~12 hours |

---

## Cost Estimates

| Platform | GPU | Cost/Hour | 10h Training |
|----------|-----|-----------|--------------|
| Google Colab Pro | T4 16GB | $10/month | Included |
| AWS p3.2xlarge | V100 16GB | $3.06 | $30.60 |
| Lambda Labs | A100 40GB | $1.10 | $11.00 |
| RunPod | RTX 4090 24GB | $0.69 | $6.90 |

**Recommendation**: Start with Colab Pro for testing, then use Lambda/RunPod for production.

---

## Checklist for Cloud Training

- [ ] Prepare datasets (conversations, scientific, math)
- [ ] Choose cloud platform (Colab/AWS/Lambda)
- [ ] Set up instance with GPU
- [ ] Clone RIIK repository
- [ ] Install dependencies
- [ ] Upload/download datasets
- [ ] Configure model size for GPU
- [ ] Run training in tmux/screen
- [ ] Monitor progress (nvidia-smi, logs)
- [ ] Save checkpoints regularly
- [ ] Download trained models
- [ ] Evaluate on test set

---

## Next Steps After Training

1. **Evaluate**: Test on held-out data
2. **Compare**: Baseline vs Hierarchical vs Kolosis V2 Minimal
3. **Analyze**: Check fusion weights, temporal attention patterns
4. **Deploy**: Use best model for inference
5. **Iterate**: Fine-tune based on results

---

## Files You'll Need

**Training Scripts** (already created):
- `experiments/wikitext/train_baseline_gpt.py`
- `experiments/wikitext/train_hierarchical.py`
- `experiments/wikitext/train_kolosis_v2_minimal.py`
- `experiments/wikitext/train_kolosis_v2_minimal_4gb.py`

**To Create** (when you have your datasets):
- `experiments/prepare_conversation_data.py`
- `experiments/prepare_scientific_data.py`
- `experiments/prepare_math_data.py`
- `experiments/train_custom_dataset.py`

---

## Questions to Answer Before Training

1. **Dataset size**: How many examples? (affects training time)
2. **GPU budget**: How much can you spend? (affects platform choice)
3. **Context length**: How long are your texts? (affects block_size)
4. **Priority**: Conversations, scientific, or math first?

Let me know your answers and I'll create the exact scripts you need!
