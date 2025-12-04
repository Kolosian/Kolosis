"""
WikiText-103 Training: Kolosis-X (Experimental)
Self-discovering multi-stream architecture with meta-learning fusion.
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

from neural_networks.kolosis.kolosis_x import KolosisX

from experiments.wikitext.dataset import WikiTextDataset

def compute_router_entropy(info):
    """Compute average router entropy from gate weights"""
    if 'gate_weights' not in info:
        return 0.0
    weights = info['gate_weights']  # (B, T, n_streams)
    # Entropy per token: -sum(p * log(p))
    entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)
    return entropy.mean().item()

def check_gradient_norms(model):
    """Check for dead streams (grad norm < 1e-6)"""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    return norms

def train_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_main = 0
    total_aux = 0
    total_unsup = 0
    total_div = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        _, loss, info = model(x, y, return_stream_outputs=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_main += info['main_loss']
        total_aux += info['aux_loss']
        total_unsup += info['unsup_loss']
        total_div += info['diversity_loss']
        
        # Monitoring
        entropy = compute_router_entropy(info)
        grad_norms = check_gradient_norms(model)
        
        # Check for dead streams (simple heuristic: check stream heads)
        dead_streams = []
        for i, _head in enumerate(model.stream_heads):
            head_norm = grad_norms.get(f'stream_heads.{i}.weight', 1.0)
            if head_norm < 1e-6:
                dead_streams.append(i)
        
        if batch_idx % 100 == 0:
            status = {
                'loss': f'{loss.item():.4f}',
                'main': f'{info["main_loss"]:.4f}',
                'div': f'{info["diversity_loss"]:.4f}',
                'ent': f'{entropy:.4f}'
            }
            if dead_streams:
                status['DEAD'] = str(dead_streams)
            pbar.set_postfix(status)
    
    n = len(train_loader)
    return {
        'total': total_loss / n,
        'main': total_main / n,
        'aux': total_aux / n,
        'unsup': total_unsup / n,
        'diversity': total_div / n
    }

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, y, return_stream_outputs=True)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def get_model_stats(model, device, sample_batch):
    """Get fusion weights"""
    model.eval()
    x, y = sample_batch
    x, y = x.to(device), y.to(device)
    
    with torch.no_grad():
        _, _, info = model(x, y, return_stream_outputs=True)
    
    # Fusion weights (meta-router)
    weights = info['gate_weights'].mean(dim=[0, 1]).cpu().tolist()
    stream_names = ['temporal', 'semantic', 'concept']
    fusion_stats = {name: weights[i] for i, name in enumerate(stream_names) if i < len(weights)}
    
    return fusion_stats

def main():
    print("="*60)
    print("WIKITEXT-103: KOLOSIS-X (EXPERIMENTAL)")
    print("="*60)
    
    config = {
        'vocab_size': 50257,
        'n_embd': 384,
        'block_size': 128,
        'n_layer': 6,      # Shared backbone layers
        'dropout': 0.1,
        'batch_size': 32,
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
    
    print(f"\nCreating Kolosis-X...")
    model = KolosisX(
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
    
    results = {
        'config': config,
        'train_losses': [],
        'val_losses': [],
        'perplexities': [],
        'fusion_weights': [],
        'loss_breakdown': []
    }
    best_val_loss = float('inf')
    os.makedirs('experiments/wikitext_results', exist_ok=True)
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch+1)
        val_loss, perplexity = evaluate(model, val_loader, device)
        
        fusion_stats = get_model_stats(model, device, sample_batch)
        
        results['train_losses'].append(train_losses['total'])
        results['val_losses'].append(val_loss)
        results['perplexities'].append(perplexity)
        results['fusion_weights'].append(fusion_stats)
        results['loss_breakdown'].append(train_losses)
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"    Main Task: {train_losses['main']:.4f}")
        print(f"    Auxiliary: {train_losses['aux']:.4f}")
        print(f"    Unsupervised: {train_losses['unsup']:.4f}")
        print(f"    Diversity: {train_losses['diversity']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"\nMeta-Router Weights:")
        print(f"  Temporal: {fusion_stats['temporal']:.4f} ({fusion_stats['temporal']*100:.1f}%)")
        print(f"  Semantic: {fusion_stats['semantic']:.4f} ({fusion_stats['semantic']*100:.1f}%)")
        print(f"  Concept:  {fusion_stats['concept']:.4f} ({fusion_stats['concept']*100:.1f}%)")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'experiments/wikitext_results/kolosis_x_best.pt')
            print("  ✅ Saved best model")
    
    with open('experiments/wikitext_results/kolosis_x_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best perplexity: {torch.exp(torch.tensor(best_val_loss)).item():.2f}")

if __name__ == "__main__":
    main()
