import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import json
import os
import sys

# Import model definition
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_kolosis_s_colab import KolosisS

class LabeledDataset(Dataset):
    def __init__(self, data_path, tokenizer, block_size=128):
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        self.samples = []
        # Mapping: category -> stream_index
        # 0: Symbol, 1: Temporal, 2: Semantic, 3: Concept
        self.target_map = {
            'temporal': 1,
            'causal': 2,
            'semantic': 2,
            'conceptual': 3
        }
        
        for cat, content in data.items():
            if cat not in self.target_map:
                continue
                
            target_idx = self.target_map[cat]
            for sentence in content['sentences']:
                tokens = tokenizer.encode(sentence, max_length=block_size, truncation=True, return_tensors='pt')[0]
                self.samples.append({
                    'tokens': tokens,
                    'target_stream': target_idx
                })
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    # Pad tokens
    max_len = max(len(x['tokens']) for x in batch)
    padded_tokens = []
    targets = []
    
    for x in batch:
        tokens = x['tokens']
        pad_len = max_len - len(tokens)
        # Pad with EOS token (50256)
        padded = F.pad(tokens, (0, pad_len), value=50256)
        padded_tokens.append(padded)
        targets.append(x['target_stream'])
        
    return torch.stack(padded_tokens), torch.tensor(targets)

def train_specialization():
    # Config
    config = {
        'vocab_size': 50257,
        'n_embd': 128,      # Match pre-trained
        'block_size': 128,
        'n_layer': 2,       # Match pre-trained
        'dropout': 0.1,
        'lr': 1e-4,
        'epochs': 5,
        'batch_size': 8,
        'alpha': 0.5        # Weight for specialization loss
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Load Dataset
    dataset_path = 'explainability_test_dataset.json'
    if not os.path.exists(dataset_path):
        dataset_path = 'experiments/wikitext/explainability_test_dataset.json'
        
    dataset = LabeledDataset(dataset_path, tokenizer, config['block_size'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    print(f"Loaded {len(dataset)} labeled samples")
    
    # Load Model
    model = KolosisS(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        block_size=config['block_size'],
        n_layer=config['n_layer'],
        dropout=config['dropout']
    ).to(device)
    
    # Load Checkpoint
    checkpoint_path = 'kolosis_s_best.pt'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'experiments/Kolosis_S_checkpoints/kolosis_s_best.pt'
        
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("⚠️ Warning: No checkpoint found! Training from scratch (not recommended).")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    model.train()
    
    print("\nStarting Alignment Tuning...")
    for epoch in range(config['epochs']):
        total_loss = 0
        total_spec_loss = 0
        
        for batch_idx, (tokens, target_streams) in enumerate(loader):
            tokens, target_streams = tokens.to(device), target_streams.to(device)
            
            # Forward pass
            # We don't have LM targets, so main_loss will be None if we don't pass targets
            # But we want to preserve LM capability, so let's use tokens as targets (shifted)
            lm_targets = tokens.clone()
            
            logits, lm_loss, info = model(tokens, targets=lm_targets, return_stream_outputs=True)
            
            # Specialization Loss
            # gate_weights: [B, T, 4]
            gate_weights = info['gate_weights']
            
            # Average weights over time: [B, 4]
            avg_weights = gate_weights.mean(dim=1)
            
            # Cross Entropy between avg_weights and target_stream
            # avg_weights are probabilities, so we use NLLLoss on log(probs)
            spec_loss = F.nll_loss(torch.log(avg_weights + 1e-8), target_streams)
            
            # Total Loss
            loss = lm_loss + config['alpha'] * spec_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_spec_loss += spec_loss.item()
            
        avg_loss = total_loss / len(loader)
        avg_spec = total_spec_loss / len(loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} (Spec={avg_spec:.4f})")
        
    # Save Aligned Model
    save_path = 'kolosis_s_aligned.pt'
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Saved aligned model to {save_path}")

if __name__ == "__main__":
    train_specialization()
