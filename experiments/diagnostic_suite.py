"""
Diagnostic Suite for Kolosis Models
Runs sanity checks before full-scale training:
1. Gradient flow analysis
2. Router entropy monitoring
3. Stream specialization probes
4. Reproducibility checks
5. Ablation studies
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

from neural_networks.kolosis.kolosis_s import KolosisS
from neural_networks.kolosis.kolosis_x import KolosisX

# Import BaselineGPT from training script
import importlib.util
spec = importlib.util.spec_from_file_location("baseline", "experiments/wikitext/train_baseline_gpt.py")
baseline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_module)
BaselineGPT = baseline_module.BaselineGPT

class MiniDataset(Dataset):
    """Tiny dataset for quick diagnostics"""
    def __init__(self, texts, tokenizer, block_size=128, max_examples=100):
        self.examples = []
        for text in texts[:50]:  # Only use first 50 documents
            if len(text.strip()) == 0:
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True)
            for i in range(0, min(len(tokens) - block_size, 200), block_size):
                chunk = tokens[i:i + block_size + 1]
                if len(chunk) == block_size + 1:
                    self.examples.append(chunk)
                    if len(self.examples) >= max_examples:
                        return
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        chunk = self.examples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def check_gradient_flow(model, dataloader, device, model_name):
    """Check gradient norms per component"""
    print(f"\n{'='*60}")
    print(f"GRADIENT FLOW CHECK: {model_name}")
    print(f"{'='*60}")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    grad_stats = {}
    
    for batch_idx, (x, y) in enumerate(dataloader):
        if batch_idx >= 5:  # Only check first 5 batches
            break
            
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if model_name == "KolosisX":
            _, loss, _ = model(x, y, return_stream_outputs=True)
        else:
            _, loss = model(x, y)
        
        loss.backward()
        
        # Collect gradient norms
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if name not in grad_stats:
                    grad_stats[name] = []
                grad_stats[name].append(grad_norm)
    
    # Analyze gradients
    print("\nGradient Statistics (top 10 by avg norm):")
    avg_grads = {name: np.mean(norms) for name, norms in grad_stats.items()}
    sorted_grads = sorted(avg_grads.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for name, avg_norm in sorted_grads:
        print(f"  {name[:50]:50s}: {avg_norm:.6f}")
    
    # Check for dead gradients
    dead_params = [name for name, avg in avg_grads.items() if avg < 1e-8]
    if dead_params:
        print(f"\n⚠️  WARNING: {len(dead_params)} parameters have near-zero gradients!")
        for name in dead_params[:5]:
            print(f"    - {name}")
    else:
        print(f"\n✅ All parameters receiving gradients")
    
    return grad_stats

def check_router_entropy(model, dataloader, device, steps=20):
    """Monitor router entropy over early training"""
    print(f"\n{'='*60}")
    print(f"ROUTER ENTROPY CHECK")
    print(f"{'='*60}")
    
    if not hasattr(model, 'router'):
        print("⚠️  Model has no router (skipping)")
        return None
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    entropy_history = []
    
    for step, (x, y) in enumerate(dataloader):
        if step >= steps:
            break
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        _, loss, info = model(x, y, return_stream_outputs=True)
        loss.backward()
        optimizer.step()
        
        # Calculate router entropy
        weights = info['gate_weights']  # (B, T, n_streams)
        # Entropy per token
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1)  # (B, T)
        avg_entropy = entropy.mean().item()
        
        entropy_history.append(avg_entropy)
        
        if step % 5 == 0:
            print(f"  Step {step:3d}: Entropy = {avg_entropy:.4f}, Loss = {loss.item():.4f}")
    
    # Analyze trend
    early_entropy = np.mean(entropy_history[:5])
    late_entropy = np.mean(entropy_history[-5:])
    
    print(f"\nEntropy Analysis:")
    print(f"  Early (steps 0-4):  {early_entropy:.4f}")
    print(f"  Late (steps 15-19): {late_entropy:.4f}")
    print(f"  Change: {late_entropy - early_entropy:+.4f}")
    
    if late_entropy < 0.1:
        print(f"  ⚠️  WARNING: Router collapsed to single stream!")
    elif late_entropy > 0.9:
        print(f"  ✅ Router is exploring multiple streams")
    else:
        print(f"  ✅ Router showing specialization")
    
    return entropy_history

def check_reproducibility(model_class, config, dataloader, device, seeds=[42, 123, 456]):
    """Test reproducibility across random seeds"""
    print(f"\n{'='*60}")
    print(f"REPRODUCIBILITY CHECK")
    print(f"{'='*60}")
    
    results = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = model_class(**config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        losses = []
        for step, (x, y) in enumerate(dataloader):
            if step >= 10:
                break
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if hasattr(model, 'router'):
                _, loss, _ = model(x, y, return_stream_outputs=True)
            else:
                _, loss = model(x, y)
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        final_loss = np.mean(losses[-3:])
        results.append(final_loss)
        print(f"  Seed {seed}: Final loss = {final_loss:.4f}")
    
    variance = np.var(results)
    print(f"\nVariance across seeds: {variance:.6f}")
    
    if variance < 0.01:
        print(f"✅ Reproducible behavior")
    else:
        print(f"⚠️  High variance - check for non-determinism")
    
    return results

def ablation_study(model_class, config, dataloader, device, epochs=2):
    """Quick ablation: no diversity loss"""
    print(f"\n{'='*60}")
    print(f"ABLATION: No Diversity Loss")
    print(f"{'='*60}")
    
    if model_class.__name__ != "KolosisX":
        print("⚠️  Ablation only applies to KolosisX (skipping)")
        return None
    
    # Train with diversity loss
    torch.manual_seed(42)
    model_with = model_class(**config).to(device)
    optimizer = torch.optim.Adam(model_with.parameters(), lr=1e-4)
    
    print("\nTraining WITH diversity loss...")
    losses_with = []
    for epoch in range(epochs):
        for step, (x, y) in enumerate(dataloader):
            if step >= 20:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss, _ = model_with(x, y, return_stream_outputs=True)
            loss.backward()
            optimizer.step()
            losses_with.append(loss.item())
    
    # Train without diversity loss (hack: set weight to 0)
    torch.manual_seed(42)
    model_without = model_class(**config).to(device)
    optimizer = torch.optim.Adam(model_without.parameters(), lr=1e-4)
    
    print("Training WITHOUT diversity loss...")
    losses_without = []
    for epoch in range(epochs):
        for step, (x, y) in enumerate(dataloader):
            if step >= 20:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            _, loss, info = model_without(x, y, return_stream_outputs=True)
            # Manually remove diversity component
            loss_no_div = (
                0.4 * info['main_loss'] +
                0.3 * info['aux_loss'] +
                0.2 * info['unsup_loss']
            ) / 0.9  # Renormalize
            loss_no_div = torch.tensor(loss_no_div, requires_grad=True)
            loss_no_div.backward()
            optimizer.step()
            losses_without.append(loss_no_div.item())
    
    print(f"\nFinal loss WITH diversity:    {np.mean(losses_with[-5:]):.4f}")
    print(f"Final loss WITHOUT diversity: {np.mean(losses_without[-5:]):.4f}")
    
    return {'with': losses_with, 'without': losses_without}

def run_full_diagnostic(model_name, model_class, config, device='cuda'):
    """Run all diagnostics for a model"""
    print(f"\n{'#'*60}")
    print(f"# DIAGNOSTIC SUITE: {model_name}")
    print(f"{'#'*60}")
    
    # Load mini dataset
    print("\nLoading mini dataset...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    mini_dataset = MiniDataset(dataset['train']['text'], tokenizer, max_examples=100)
    dataloader = DataLoader(mini_dataset, batch_size=4, shuffle=True)
    
    print(f"Dataset size: {len(mini_dataset)} examples")
    
    # Create model
    model = model_class(**config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    results = {
        'model': model_name,
        'params': n_params,
        'config': config
    }
    
    # Run checks
    results['gradient_flow'] = check_gradient_flow(model, dataloader, device, model_name)
    
    if hasattr(model, 'router'):
        results['router_entropy'] = check_router_entropy(model, dataloader, device)
    
    results['reproducibility'] = check_reproducibility(model_class, config, dataloader, device)
    
    if model_name == "KolosisX":
        results['ablation'] = ablation_study(model_class, config, dataloader, device)
    
    return results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Shared config (small for diagnostics)
    config = {
        'vocab_size': 50257,
        'n_embd': 128,
        'block_size': 128,
        'n_layer': 2,
        'dropout': 0.1
    }
    
    all_results = {}
    
    # Test Baseline
    all_results['baseline'] = run_full_diagnostic('BaselineGPT', BaselineGPT, config, device)
    
    # Test Kolosis-S
    all_results['kolosis_s'] = run_full_diagnostic('KolosisS', KolosisS, config, device)
    
    # Test Kolosis-X
    all_results['kolosis_x'] = run_full_diagnostic('KolosisX', KolosisX, config, device)
    
    # Save results
    output_dir = Path('experiments/diagnostic_results')
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON (excluding non-serializable items)
    json_results = {
        k: {
            'model': v['model'],
            'params': v['params'],
            'reproducibility': v['reproducibility'],
            'router_entropy': v.get('router_entropy'),
        }
        for k, v in all_results.items()
    }
    
    with open(output_dir / 'diagnostic_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSTICS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir / 'diagnostic_results.json'}")

if __name__ == "__main__":
    main()
