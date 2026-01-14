import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import numpy as np
import os
import sys
import math

# Import model definition
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_kolosis_s_colab import KolosisS, WikiTextDataset

def measure_optimality():
    # Config
    config = {
        'vocab_size': 50257,
        'n_embd': 128,
        'block_size': 128,
        'n_layer': 2,
        'dropout': 0.1,
        'batch_size': 8,
        'eval_batches': 50  # Limit batches for speed
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
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
        print("❌ Checkpoint not found!")
        return
        
    model.eval()
    
    # Load Data
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    dataset = WikiTextDataset('wikitext-103-v1', 'validation', config['block_size'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Diagnostic Metrics
    total_tokens = 0
    optimal_matches = 0
    total_log_likelihood_chosen = 0.0
    total_log_likelihood_optimal = 0.0
    
    print("\nRunning Router Optimality Diagnostic...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= config['eval_batches']: break
            
            x, y = x.to(device), y.to(device)
            B, T = x.shape
            
            # 1. Get Logits for ALL streams
            # We need to hack the forward pass slightly to get individual stream outputs
            # Or use the return_stream_outputs flag if we modified the model (which we did for explainability)
            # Let's assume return_stream_outputs is available or minimal modification needed
            
            # Since KolosisS forward doesn't return per-stream logits for next token, 
            # we need to execute the final layer for each stream feature.
            # But the final layer is usually shared (ln_f + head).
            # The fusion happens BEFORE the final layer.
            
            # Step A: Get stream features pre-fusion
            # We need to manually run parts of the model to get per-stream losses.
            # This is tricky without modifying the class. 
            # Let's inspect the model structure dynamically or use hooks if needed.
            
            # Actually, `fusion_gate` returns `stream_features` in `info` if requested.
            # Let's rely on that.
            
            _, _, info = model(x, return_stream_outputs=True)
            stream_features = info['stream_features'] # [B, T, 4, n_embd]
            gate_weights = info['gate_weights']       # [B, T, 4]
            
            # Step B: Project each stream feature to logits
            # KolosisS usually has self.ln_f and self.head
            # We need to apply ln_f and head to EACH stream separately
            
            n_streams = stream_features.shape[2]
            per_stream_losses = []
            
            for k in range(n_streams):
                feat_k = stream_features[:, :, k, :] # [B, T, n_embd]
                
                # Apply Final Layer Norm
                feat_k = model.ln_f(feat_k)
                
                # Apply Head
                logits_k = model.head(feat_k) # [B, T, vocab_size]
                
                # Calculate Loss for this stream
                loss_k = F.cross_entropy(logits_k.view(-1, config['vocab_size']), y.view(-1), reduction='none')
                loss_k = loss_k.view(B, T)
                per_stream_losses.append(loss_k)
                
            per_stream_losses = torch.stack(per_stream_losses, dim=-1) # [B, T, 4]
            
            # Step C: Determine Choices
            optimal_indices = torch.argmin(per_stream_losses, dim=-1) # [B, T]
            chosen_indices = torch.argmax(gate_weights, dim=-1)       # [B, T]
            
            # Step D: Calculate Metrics
            matches = (optimal_indices == chosen_indices).sum().item()
            optimal_matches += matches
            total_tokens += (B * T)
            
            # Accumulate Losses (for PPL calculation)
            # Gather optimal and chosen losses
            loss_optimal = torch.gather(per_stream_losses, -1, optimal_indices.unsqueeze(-1)).squeeze(-1)
            loss_chosen = torch.gather(per_stream_losses, -1, chosen_indices.unsqueeze(-1)).squeeze(-1)
            
            total_log_likelihood_optimal += loss_optimal.sum().item()
            total_log_likelihood_chosen += loss_chosen.sum().item()
            
            if i % 10 == 0:
                print(f"Batch {i}/{config['eval_batches']} processed...")

    # Final Calculation
    optimality_rate = (optimal_matches / total_tokens) * 100
    
    avg_loss_chosen = total_log_likelihood_chosen / total_tokens
    avg_loss_optimal = total_log_likelihood_optimal / total_tokens
    
    ppl_chosen = math.exp(avg_loss_chosen)
    ppl_optimal = math.exp(avg_loss_optimal)
    
    potential_gain = ppl_chosen - ppl_optimal
    
    print("\n" + "="*50)
    print("ROUTER OPTIMALITY DIAGNOSTIC (KOLOSIS-S)")
    print("="*50)
    print(f"Optimality Rate:        {optimality_rate:.2f}%")
    print(f"Current PPL (Router):   {ppl_chosen:.2f}")
    print(f"Oracle PPL (Optimal):   {ppl_optimal:.2f}")
    print(f"Potential Gain:         {potential_gain:.2f} PPL")
    print("="*50)
    
    # Verdict
    print("\nVERDICT:")
    if potential_gain >= 0.3:
        print("✅ RL RECOMMENDED (Gain >= 0.3 PPL)")
    else:
        print("❌ RL NOT RECOMMENDED (Gain too small)")
        
    if optimality_rate > 80:
        print("❌ Router is already near-optimal (>80%)")
    elif optimality_rate < 65:
        print("✅ Router is underfitting (<65%) - Strong candidate for RL")
    else:
        print("⚠️ Router is decent (65-80%) - Marginal gains expected")

if __name__ == "__main__":
    measure_optimality()
