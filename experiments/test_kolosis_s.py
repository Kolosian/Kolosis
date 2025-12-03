"""
Test script for Kolosis-S (Streamlined)
Verifies parameter count and functionality.
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_networks.kolosis.kolosis_s import KolosisS

def test_kolosis_s():
    print("="*60)
    print("TESTING KOLOSIS-S (STREAMLINED)")
    print("="*60)
    
    # Config similar to WikiText experiments
    config = {
        'vocab_size': 50257,  # GPT-2 vocab size
        'n_embd': 768,        # GPT-2 small size
        'block_size': 128,
        'n_layer': 2,
        'dropout': 0.1
    }
    
    print(f"Config: {config}")
    
    model = KolosisS(**config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model created successfully!")
    print(f"Total Parameters: {n_params:,}")
    
    # Compare with V2 estimate
    # V2 would be: 4 streams * (embeddings + layers + heads) + fusion
    # Approx: 4 * (38M + 7M + 38M) = ~332M (huge!)
    # Kolosis-S: ~80M (mostly embeddings) + ~10M layers
    
    print(f"\nTesting forward pass...")
    x = torch.randint(0, 1000, (2, 32))
    y = torch.randint(0, 1000, (2, 32))
    
    logits, loss, info = model(x, y, return_stream_outputs=True)
    
    print(f"✅ Forward pass successful!")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    print(f"\nFusion Weights:")
    weights = info['gate_weights'].mean(dim=[0,1])
    print(f"  Symbol:   {weights[0]:.4f}")
    print(f"  Temporal: {weights[1]:.4f}")
    print(f"  Semantic: {weights[2]:.4f}")
    print(f"  Concept:  {weights[3]:.4f}")
    
    print("\n" + "="*60)
    print("✅ KOLOSIS-S IS READY")
    print("="*60)

if __name__ == "__main__":
    test_kolosis_s()
