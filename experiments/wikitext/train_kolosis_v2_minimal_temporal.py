"""
WikiText-103 Training: Kolosis V2 Minimal WITH Temporal Attention
This version adds temporal attention to test its effectiveness.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
import json
from tqdm import tqdm

# Import the model with temporal attention
from neural_networks.kolosis.kolosis_v2_minimal_temporal import KolosisV2MinimalWithTemporal

class WikiTextDataset(Dataset):
    """WikiText-103 dataset"""
    
    def __init__(self, texts, tokenizer, block_size=128):
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
            
            # Use non-overlapping windows to prevent data leakage
            for i in range(0, len(tokens) - block_size, block_size):
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
    losses = []
    grad_norms = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        
        # Track gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norms.append(grad_norm.item())
        
        optimizer.step()
        
        total_loss += loss.item()
        losses.append(loss.item())
        
        # More frequent updates with detailed info
        if batch_idx % 50 == 0:
            recent_loss = sum(losses[-50:]) / len(losses[-50:]) if losses else loss.item()
            recent_grad = sum(grad_norms[-50:]) / len(grad_norms[-50:]) if grad_norms else grad_norm.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{recent_loss:.4f}',
                'grad_norm': f'{recent_grad:.3f}'
            })
    
    avg_loss = total_loss / len(train_loader)
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
    
    return avg_loss, avg_grad_norm, losses

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
    print("WIKITEXT-103: KOLOSIS V2 MINIMAL WITH TEMPORAL ATTENTION")
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
    print(f"\n✨ NEW: Temporal attention with multi-scale decay!")
    
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
    
    # Create model WITH temporal attention
    print(f"\nCreating Kolosis V2 Minimal WITH Temporal Attention...")
    model = KolosisV2MinimalWithTemporal(
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
    
    results = {
        'config': config,
        'train_losses': [],
        'val_losses': [],
        'perplexities': [],
        'fusion_weights': [],
        'temporal_stats': [],
        'grad_norms': []
    }
    best_val_loss = float('inf')
    os.makedirs('experiments/wikitext_results', exist_ok=True)
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        train_loss, avg_grad_norm, train_losses = train_epoch(model, train_loader, optimizer, device, epoch+1)
        val_loss, perplexity = evaluate(model, val_loader, device)
        
        # Get fusion weights
        fusion = model.get_fusion_weights()
        
        # Get temporal attention stats
        temporal_stats = model.get_temporal_stats()
        if temporal_stats:
            avg_stats = {
                'gamma_fast': sum(s['gamma_fast'] for s in temporal_stats) / len(temporal_stats),
                'gamma_medium': sum(s['gamma_medium'] for s in temporal_stats) / len(temporal_stats),
                'gamma_slow': sum(s['gamma_slow'] for s in temporal_stats) / len(temporal_stats),
                'alpha_fast': sum(s['alpha_fast'] for s in temporal_stats) / len(temporal_stats),
                'alpha_medium': sum(s['alpha_medium'] for s in temporal_stats) / len(temporal_stats),
                'alpha_slow': sum(s['alpha_slow'] for s in temporal_stats) / len(temporal_stats),
            }
        else:
            avg_stats = {}
        
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        results['perplexities'].append(perplexity)
        results['fusion_weights'].append(fusion)
        results['temporal_stats'].append(avg_stats)
        results['grad_norms'].append(avg_grad_norm)
        
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1} RESULTS")
        print(f"{'='*60}")
        
        print(f"\n📊 Loss Metrics:")
        print(f"  Train Loss:     {train_loss:.4f}")
        print(f"  Val Loss:       {val_loss:.4f}")
        print(f"  Perplexity:     {perplexity:.2f}")
        print(f"  Avg Grad Norm:  {avg_grad_norm:.4f}")
        
        # Loss improvement tracking
        if epoch > 0:
            train_improvement = ((results['train_losses'][-1] - train_loss) / results['train_losses'][-1]) * 100
            val_improvement = ((results['val_losses'][-1] - val_loss) / results['val_losses'][-1]) * 100
            print(f"  Train Δ:        {train_improvement:+.2f}%")
            print(f"  Val Δ:          {val_improvement:+.2f}%")
        
        print(f"\n🔀 Fusion Weights:")
        print(f"  Concept:  {fusion['concept']:.4f} ({fusion['concept']*100:.1f}%)")
        print(f"  Semantic: {fusion['semantic']:.4f} ({fusion['semantic']*100:.1f}%)")
        print(f"  Temporal: {fusion['temporal']:.4f} ({fusion['temporal']*100:.1f}%)")
        
        # Show which stream is dominant
        dominant = max(fusion.items(), key=lambda x: x[1])
        print(f"  Dominant: {dominant[0].capitalize()} ({dominant[1]*100:.1f}%)")
        
        if avg_stats:
            print(f"\n⏱️  Temporal Attention (averaged across heads):")
            print(f"  Fast:   γ={avg_stats['gamma_fast']:.4f}, α={avg_stats['alpha_fast']:.4f} ({avg_stats['alpha_fast']*100:.1f}%)")
            print(f"  Medium: γ={avg_stats['gamma_medium']:.4f}, α={avg_stats['alpha_medium']:.4f} ({avg_stats['alpha_medium']*100:.1f}%)")
            print(f"  Slow:   γ={avg_stats['gamma_slow']:.4f}, α={avg_stats['alpha_slow']:.4f} ({avg_stats['alpha_slow']*100:.1f}%)")
            
            # Calculate effective memory spans
            fast_span = int(-10 / torch.log(torch.tensor(avg_stats['gamma_fast'])).item()) if avg_stats['gamma_fast'] < 1.0 else 999
            medium_span = int(-10 / torch.log(torch.tensor(avg_stats['gamma_medium'])).item()) if avg_stats['gamma_medium'] < 1.0 else 999
            slow_span = int(-10 / torch.log(torch.tensor(avg_stats['gamma_slow'])).item()) if avg_stats['gamma_slow'] < 1.0 else 999
            
            print(f"\n  Memory Spans (tokens until 10% decay):")
            print(f"    Fast:   ~{fast_span} tokens")
            print(f"    Medium: ~{medium_span} tokens")
            print(f"    Slow:   ~{slow_span} tokens")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'experiments/wikitext_results/kolosis_v2_minimal_temporal_best.pt')
            print("  ✅ Saved best model")
    
    # Save results
    with open('experiments/wikitext_results/kolosis_v2_minimal_temporal_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best perplexity: {torch.exp(torch.tensor(best_val_loss)).item():.2f}")
    print(f"Final fusion weights: Concept={results['fusion_weights'][-1]['concept']:.4f}, "
          f"Semantic={results['fusion_weights'][-1]['semantic']:.4f}, "
          f"Temporal={results['fusion_weights'][-1]['temporal']:.4f}")
    
    if results['temporal_stats'][-1]:
        final_temporal = results['temporal_stats'][-1]
        print(f"Final temporal scales: Fast={final_temporal['alpha_fast']:.4f}, "
              f"Medium={final_temporal['alpha_medium']:.4f}, "
              f"Slow={final_temporal['alpha_slow']:.4f}")

if __name__ == "__main__":
    main()
