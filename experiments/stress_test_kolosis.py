"""
Stress Test for Kolosis Models
Rigorous check for data leakage (cheating) and mode collapse.
"""
import torch
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_networks.kolosis.kolosis_s import KolosisS
from neural_networks.kolosis.kolosis_x import KolosisX

def check_causality_perturbation(model, model_name, seq_len=16):
    """
    Checks if changing input at t+1 affects output at t.
    If it does, the model is cheating (seeing the future).
    """
    print(f"\n[{model_name}] Checking Causality (Perturbation)...")
    model.eval()
    
    B = 1
    x = torch.randint(0, 100, (B, seq_len))
    
    # Run 1: Original input
    with torch.no_grad():
        out = model(x)
        y1 = out[0]
    
    # Perturb the LAST token
    x_perturbed = x.clone()
    x_perturbed[0, -1] = (x_perturbed[0, -1] + 1) % 100
    
    # Run 2: Perturbed input
    with torch.no_grad():
        out = model(x_perturbed)
        y2 = out[0]
        
    # Check differences at all positions
    # We expect output at t to be SAME for both runs if t < last_token_idx
    # The output at the LAST position (predicting next token) SHOULD change (maybe)
    # But outputs at positions 0..T-2 MUST NOT change when x[T-1] changes.
    
    diff = (y1 - y2).abs().max(dim=-1).values.squeeze()
    
    # diff[t] is the change in output at step t
    # If we changed input at T-1, then outputs at 0..T-2 should be IDENTICAL.
    
    leakage_found = False
    for t in range(seq_len - 1):
        if diff[t] > 1e-5:
            print(f"  ❌ LEAKAGE DETECTED at step {t}! Output changed when step {seq_len-1} input changed.")
            print(f"     Diff: {diff[t].item()}")
            leakage_found = True
            
    if not leakage_found:
        print("  ✅ Passed perturbation test. No future info leaked.")
    else:
        print("  ⚠️  MODEL IS CHEATING!")

def check_causality_gradient(model, model_name, seq_len=16):
    """
    Checks if gradients flow from output at t to input at t+1.
    This is the most rigorous mathematical check.
    """
    print(f"\n[{model_name}] Checking Causality (Gradient)...")
    model.train() # Gradients need training mode usually, or just requires_grad
    
    # Embeddings are usually discrete, so we can't diff wrt input indices directly.
    # But we can check if the model's internal attention allows flow.
    # A proxy is to check if the output at t depends on the embedding of t+1.
    
    # We'll hook the embedding layer to get gradients wrt embeddings
    
    B = 1
    x = torch.randint(0, 100, (B, seq_len))
    
    # Get embeddings
    if hasattr(model, 'token_emb'):
        emb = model.token_emb(x)
    else:
        print("  Skipping gradient check (structure unclear)")
        return

    emb.retain_grad()
    
    # We can't easily run forward from embeddings because the model takes indices.
    # We'll skip this for now and rely on perturbation, which is effectively the same for discrete inputs.
    # Actually, perturbation is better for "black box" checking.
    print("  (Skipping gradient check in favor of perturbation test)")

def check_router_collapse(model, model_name):
    """
    Checks if the router/gate always outputs the same weights (collapse).
    """
    if model_name == "Kolosis-S":
        # Kolosis-S has a static gate (MLP), but it should vary by input
        print(f"\n[{model_name}] Checking Fusion Gate Variance...")
    else:
        print(f"\n[{model_name}] Checking Meta-Router Variance...")
        
    model.eval()
    B = 4
    T = 32
    x = torch.randint(0, 100, (B, T))
    
    with torch.no_grad():
        _, _, info = model(x, return_stream_outputs=True)
        
    weights = info['gate_weights'] # (B, T, n_streams)
    
    # Calculate variance across Batch and Time
    var = weights.var(dim=[0, 1]) # Variance per stream
    mean = weights.mean(dim=[0, 1])
    
    print("  Stream Weights (Mean ± Std):")
    for i in range(weights.shape[-1]):
        print(f"    Stream {i}: {mean[i]:.4f} ± {var[i].sqrt():.4f}")
        
    if var.sum() < 1e-5:
        print("  ⚠️  WARNING: Router has collapsed! (Zero variance)")
    else:
        print("  ✅ Router is active and dynamic.")

def main():
    print("="*60)
    print("STRESS TEST: VULNERABILITY CHECK")
    print("="*60)
    
    config = {
        'vocab_size': 1000,
        'n_embd': 64,
        'block_size': 32,
        'n_layer': 2,
        'dropout': 0.0
    }
    
    # 1. Test Kolosis-S
    model_s = KolosisS(**config)
    check_causality_perturbation(model_s, "Kolosis-S")
    check_router_collapse(model_s, "Kolosis-S")
    
    # 2. Test Kolosis-X
    model_x = KolosisX(**config)
    check_causality_perturbation(model_x, "Kolosis-X")
    check_router_collapse(model_x, "Kolosis-X")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
