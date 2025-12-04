"""
WikiText-103 Training: Kolosis-S (Streamlined)
Optimized multi-stream architecture with shared backbone.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
import json
from tqdm import tqdm

from neural_networks.kolosis.kolosis_s import KolosisS

from experiments.wikitext.dataset import WikiTextDataset

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

def get_model_stats(model, device, sample_batch):
    """Get fusion weights and temporal stats"""
    model.eval()
    x, y = sample_batch
    x, y = x.to(device), y.to(device)
    
    with torch.no_grad():
        _, _, info = model(x, y, return_stream_outputs=True)
    
    # Fusion weights
    weights = info['gate_weights'].mean(dim=[0, 1]).cpu().tolist()
    fusion_stats = {
        'symbol': weights[0],
        'temporal': weights[1],
        'semantic': weights[2],
        'concept': weights[3]
    }
    
    # Temporal stats
    temporal_stats = model.temporal_adapter.attn.get_temporal_stats()
    
    return fusion_stats, temporal_stats

def main():
    print("="*60)
    print("WIKITEXT-103: KOLOSIS-S (STREAMLINED)")
    print("="*60)
    
    config = {
        'vocab_size': 50257,
        'n_embd': 384,
        'block_size': 128,
        'n_layer': 6,      # Shared backbone layers
        'dropout': 0.1,
        'batch_size': 32,  # Optimized model allows larger batch size!
        'epochs': 10,
        'lr': 0.0003
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\nLoading WikiText-103...")
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    train_dataset = WikiTextDataset(dataset['train']['text'], tokenizer, config['block_size'])
    val_dataset = WikiTextDataset(dataset['validation']['text'], tokenizer, config['block_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=2, pin_memory=True)
    
    # Keep a sample batch for stats
    sample_batch = next(iter(val_loader))
    
    print(f"\nCreating Kolosis-S...")
    model = KolosisS(
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
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    results = {'config': config, 'train_losses': [], 'val_losses': [], 'perplexities': [], 'fusion_weights': [], 'temporal_stats': []}
    best_val_loss = float('inf')
    os.makedirs('experiments/wikitext_results', exist_ok=True)
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch+1)
        val_loss, perplexity = evaluate(model, val_loader, device)
        
        fusion_stats, temporal_stats = get_model_stats(model, device, sample_batch)
        
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['perplexities'].append(perplexity)
        results['fusion_weights'].append(fusion_stats)
        results['temporal_stats'].append(temporal_stats)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"\nFusion weights:")
        print(f"  Symbol:   {fusion_stats['symbol']:.4f} ({fusion_stats['symbol']*100:.1f}%)")
        print(f"  Temporal: {fusion_stats['temporal']:.4f} ({fusion_stats['temporal']*100:.1f}%)")
        print(f"  Semantic: {fusion_stats['semantic']:.4f} ({fusion_stats['semantic']*100:.1f}%)")
        print(f"  Concept:  {fusion_stats['concept']:.4f} ({fusion_stats['concept']*100:.1f}%)")
        
        print(f"\nTemporal attention:")
        print(f"  Fast:   γ={temporal_stats['gamma_fast']:.4f}, α={temporal_stats['alpha_fast']:.4f}")
        print(f"  Medium: γ={temporal_stats['gamma_medium']:.4f}, α={temporal_stats['alpha_medium']:.4f}")
        print(f"  Slow:   γ={temporal_stats['gamma_slow']:.4f}, α={temporal_stats['alpha_slow']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'experiments/wikitext_results/kolosis_s_best.pt')
            print("  ✅ Saved best model")
    
    with open('experiments/wikitext_results/kolosis_s_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best perplexity: {torch.exp(torch.tensor(best_val_loss)).item():.2f}")

if __name__ == "__main__":
    main()
