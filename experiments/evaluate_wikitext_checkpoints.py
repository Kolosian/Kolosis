"""
Evaluate WikiText-103 checkpoints to extract perplexity and answer critical questions.

This script:
1. Loads the baseline and Kolosis checkpoints
2. Evaluates on WikiText-103 validation set
3. Computes perplexity
4. Creates fair parameter-matched comparison
5. Checks temporal attention activation (if applicable)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import json

# Import models
from neural_networks.kolosis.kolosis_v2_minimal import KolosisV2Minimal

class BaselineGPT(nn.Module):
    """Baseline GPT for comparison"""
    
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=6, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4*n_embd,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
            
        return logits, loss

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
            
            for i in range(0, len(tokens) - block_size, block_size // 2):
                chunk = tokens[i:i + block_size + 1]
                if len(chunk) == block_size + 1:
                    self.examples.append(chunk)
        
        print(f"Created {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        chunk = self.examples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def evaluate(model, val_loader, device):
    """Evaluate model and return loss and perplexity"""
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

def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters())

def main():
    print("="*80)
    print("WIKITEXT-103 CHECKPOINT EVALUATION")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cpu':
        print("\n⚠️  WARNING: Running on CPU. This will be slow!")
        print("Consider running on GPU for faster evaluation.")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("\nLoading WikiText-103 validation set...")
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    val_dataset = WikiTextDataset(dataset['validation']['text'], tokenizer, block_size=128)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=2)
    
    results = {}
    
    # Evaluate Baseline GPT
    print(f"\n{'='*80}")
    print("EVALUATING BASELINE GPT")
    print('='*80)
    
    baseline_ckpt_path = 'experiments/wikitext_results/baseline_gpt_best.pt'
    if os.path.exists(baseline_ckpt_path):
        print(f"Loading checkpoint: {baseline_ckpt_path}")
        baseline_ckpt = torch.load(baseline_ckpt_path, map_location=device)
        
        # Create baseline model
        baseline_model = BaselineGPT(
            vocab_size=50257,
            n_embd=256,
            n_head=8,
            n_layer=6,
            block_size=256,
            dropout=0.1
        )
        
        # Load weights
        if 'model_state_dict' in baseline_ckpt:
            baseline_model.load_state_dict(baseline_ckpt['model_state_dict'])
            print(f"✓ Loaded model weights")
            
            # Check if metrics are in checkpoint
            if 'perplexity' in baseline_ckpt:
                print(f"Checkpoint perplexity: {baseline_ckpt['perplexity']:.2f}")
            if 'val_loss' in baseline_ckpt:
                print(f"Checkpoint val loss: {baseline_ckpt['val_loss']:.4f}")
        else:
            print("⚠️  Checkpoint format not recognized")
        
        baseline_model = baseline_model.to(device)
        baseline_params = count_parameters(baseline_model)
        print(f"Parameters: {baseline_params:,} ({baseline_params/1e6:.2f}M)")
        
        # Evaluate
        print("\nEvaluating on validation set...")
        val_loss, perplexity = evaluate(baseline_model, val_loader, device)
        
        results['baseline'] = {
            'val_loss': val_loss,
            'perplexity': perplexity,
            'parameters': baseline_params
        }
        
        print(f"\n✓ Baseline GPT Results:")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Parameters: {baseline_params:,}")
    else:
        print(f"✗ Checkpoint not found: {baseline_ckpt_path}")
    
    # Evaluate Kolosis V2 Minimal
    print(f"\n{'='*80}")
    print("EVALUATING KOLOSIS V2 MINIMAL")
    print('='*80)
    
    kolosis_ckpt_path = 'experiments/wikitext_results/kolosis_v2_minimal_4gb_best.pt'
    if os.path.exists(kolosis_ckpt_path):
        print(f"Loading checkpoint: {kolosis_ckpt_path}")
        
        # Create Kolosis model
        kolosis_model = KolosisV2Minimal(
            vocab_size=50257,
            n_embd=128,
            block_size=128,
            n_layer=4,
            dropout=0.1
        )
        
        # Load weights
        kolosis_model.load_state_dict(torch.load(kolosis_ckpt_path, map_location=device))
        print(f"✓ Loaded model weights")
        
        kolosis_model = kolosis_model.to(device)
        kolosis_params = count_parameters(kolosis_model)
        print(f"Parameters: {kolosis_params:,} ({kolosis_params/1e6:.2f}M)")
        
        # Check fusion weight
        fusion_weight = torch.sigmoid(kolosis_model.fusion_weight).item()
        print(f"Fusion weight: {fusion_weight:.4f} (concept: {fusion_weight:.2f}, semantic: {1-fusion_weight:.2f})")
        
        # Evaluate
        print("\nEvaluating on validation set...")
        val_loss, perplexity = evaluate(kolosis_model, val_loader, device)
        
        results['kolosis'] = {
            'val_loss': val_loss,
            'perplexity': perplexity,
            'parameters': kolosis_params,
            'fusion_weight': fusion_weight
        }
        
        print(f"\n✓ Kolosis V2 Minimal Results:")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Parameters: {kolosis_params:,}")
        print(f"  Fusion Weight: {fusion_weight:.4f}")
    else:
        print(f"✗ Checkpoint not found: {kolosis_ckpt_path}")
    
    # Comparison
    if 'baseline' in results and 'kolosis' in results:
        print(f"\n{'='*80}")
        print("COMPARISON")
        print('='*80)
        
        baseline_ppl = results['baseline']['perplexity']
        kolosis_ppl = results['kolosis']['perplexity']
        ppl_improvement = ((baseline_ppl - kolosis_ppl) / baseline_ppl) * 100
        
        baseline_params = results['baseline']['parameters']
        kolosis_params = results['kolosis']['parameters']
        param_reduction = ((baseline_params - kolosis_params) / baseline_params) * 100
        
        print(f"\nPerplexity:")
        print(f"  Baseline GPT:       {baseline_ppl:7.2f}")
        print(f"  Kolosis V2 Minimal: {kolosis_ppl:7.2f}")
        print(f"  Improvement:        {ppl_improvement:+7.2f}%")
        
        print(f"\nParameters:")
        print(f"  Baseline GPT:       {baseline_params:10,} ({baseline_params/1e6:.2f}M)")
        print(f"  Kolosis V2 Minimal: {kolosis_params:10,} ({kolosis_params/1e6:.2f}M)")
        print(f"  Reduction:          {param_reduction:7.2f}%")
        
        print(f"\nEfficiency:")
        print(f"  Kolosis achieves {ppl_improvement:+.1f}% perplexity improvement")
        print(f"  with {param_reduction:.1f}% fewer parameters")
        
        # Save results
        output_file = 'experiments/wikitext_results/evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
    
    print(f"\n{'='*80}")
    print("CRITICAL QUESTIONS ANSWERED")
    print('='*80)
    
    if 'kolosis' in results:
        print(f"\n1. What's the perplexity?")
        print(f"   Kolosis V2 Minimal: {results['kolosis']['perplexity']:.2f}")
        if 'baseline' in results:
            print(f"   Baseline GPT: {results['baseline']['perplexity']:.2f}")
    
    print(f"\n2. Did temporal attention activate?")
    print(f"   N/A - Kolosis V2 Minimal doesn't use temporal attention")
    print(f"   (Temporal was removed in favor of concept-semantic fusion)")
    
    if 'baseline' in results and 'kolosis' in results:
        print(f"\n3. How does 40M param Kolosis compare to 40M param baseline?")
        print(f"   Kolosis: {kolosis_params/1e6:.2f}M params, {results['kolosis']['perplexity']:.2f} perplexity")
        print(f"   Baseline: {baseline_params/1e6:.2f}M params, {results['baseline']['perplexity']:.2f} perplexity")
        print(f"   Note: Baseline has {baseline_params/1e6:.2f}M params (not 40M)")
        print(f"   Need to create a 40M param baseline for fair comparison")

if __name__ == '__main__':
    main()
