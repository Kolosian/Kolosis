"""
WikiText-103 Training: Kolosis V2 Minimal (Memory-Optimized)
Reduced model size to fit 4GB GPU
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
import json
from tqdm import tqdm

# Import the model from the main file
from neural_networks.kolosis.kolosis_v2_minimal import KolosisV2Minimal

class WikiTextDataset(Dataset):
    """WikiText-103 dataset"""
    
    def __init__(self, texts, tokenizer, block_size=128):  # Reduced from 256
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        
        print(f"Tokenizing {len(texts)} documents...")
        for text in tqdm(texts):
            if len(text.strip()) == 0:
                continue
            
            tokens = tokenizer.encode(
                text, 
                add_special_tokens=False,
                max_length=2048,
                truncation=True
            )
            
            for i in range(0, len(tokens) - block_size, block_size // 2):
                chunk = tokens[i:i + block_size + 1]
                if len(chunk) == block_size + 1:
                    self.examples.append(chunk)
        
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        chunk = self.examples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def main():
    print("="*60)
    print("WIKITEXT-103: KOLOSIS V2 MINIMAL (4GB GPU)")
    print("="*60)
    
    # Memory-optimized configuration for 4GB GPU
    config = {
        'vocab_size': 50257,
        'n_embd': 128,        # Reduced from 256
        'block_size': 128,    # Reduced from 256
        'n_layer': 4,         # Reduced from 6
        'dropout': 0.1,
        'batch_size': 8,      # Reduced from 32
        'epochs': 10,
        'lr': 0.0003
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"\n⚠️  Memory-optimized config for 4GB GPU:")
    print(f"  - Embedding size: {config['n_embd']} (was 256)")
    print(f"  - Context length: {config['block_size']} (was 256)")
    print(f"  - Layers: {config['n_layer']} (was 6)")
    print(f"  - Batch size: {config['batch_size']} (was 32)")
    
    # Load tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("\nLoading WikiText-103...")
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    train_dataset = WikiTextDataset(dataset['train']['text'], tokenizer, config['block_size'])
    val_dataset = WikiTextDataset(dataset['validation']['text'], tokenizer, config['block_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=2, pin_memory=True)
    
    # Create model
    print(f"\nCreating Kolosis V2 Minimal...")
    model = KolosisV2Minimal(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        block_size=config['block_size'],
        n_layer=config['n_layer'],
        dropout=config['dropout']
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    # Training
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    results = {'config': config, 'train_losses': [], 'val_losses': [], 'perplexities': [], 'fusion_weights': []}
    best_val_loss = float('inf')
    os.makedirs('experiments/wikitext_results', exist_ok=True)
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch+1)
        val_loss, perplexity = evaluate(model, val_loader, device)
        
        fusion_weight = torch.sigmoid(model.fusion_weight).item()
        
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['perplexities'].append(perplexity)
        results['fusion_weights'].append(fusion_weight)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Concept weight (α): {torch.sigmoid(model.alpha).item():.4f}")
        print(f"  Fusion weight: {fusion_weight:.4f} (concept: {fusion_weight:.2f}, semantic: {1-fusion_weight:.2f})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'experiments/wikitext_results/kolosis_v2_minimal_4gb_best.pt')
            print("  ✅ Saved best model")
    
    with open('experiments/wikitext_results/kolosis_v2_minimal_4gb_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best perplexity: {torch.exp(torch.tensor(best_val_loss)).item():.2f}")
    print(f"Final fusion weight: {results['fusion_weights'][-1]:.4f}")

if __name__ == "__main__":
    main()
