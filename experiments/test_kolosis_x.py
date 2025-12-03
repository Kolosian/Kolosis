"""
Test script for Kolosis-X (Experimental)
Verifies parameter count, unsupervised losses, and meta-routing.
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_networks.kolosis.kolosis_x import KolosisX

def test_kolosis_x():
    print("="*60)
    print("TESTING KOLOSIS-X (EXPERIMENTAL)")
    print("="*60)
    
    config = {
        'vocab_size': 50257,
        'n_embd': 768,
        'block_size': 128,
        'n_layer': 2,
        'dropout': 0.1
    }
    
    print(f"Config: {config}")
    
    model = KolosisX(**config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Model created successfully!")
    print(f"Total Parameters: {n_params:,}")
    
    # Kolosis-S was ~188M. Kolosis-X should be similar or slightly less due to shared backbone
    # but slightly more due to meta-router and stream heads.
    # Let's see.
    
    print(f"\nTesting forward pass...")
    x = torch.randint(0, 1000, (2, 32))
    y = torch.randint(0, 1000, (2, 32))
    
    logits, loss, info = model(x, y, return_stream_outputs=True)
    
    print(f"✅ Forward pass successful!")
    print(f"Logits shape: {logits.shape}")
    print(f"Total Loss: {loss.item():.4f}")
    
    print(f"\nLoss Breakdown:")
    print(f"  Main Task: {info['main_loss']:.4f}")
    print(f"  Auxiliary: {info['aux_loss']:.4f}")
    print(f"  Unsupervised: {info['unsup_loss']:.4f}")
    print(f"  Diversity: {info['diversity_loss']:.4f}")
    
    print(f"\nMeta-Router Weights (Avg):")
    weights = info['gate_weights'].mean(dim=[0,1])
    print(f"  Temporal: {weights[0]:.4f}")
    print(f"  Semantic: {weights[1]:.4f}")
    print(f"  Concept:  {weights[2]:.4f}")
    
    print(f"\nTesting Dynamic Stream Addition...")
    from neural_networks.kolosis.kolosis_x import UnsupervisedStream
    
    class CausalStream(UnsupervisedStream):
        def unsupervised_loss(self, features, **kwargs):
            return torch.tensor(0.1) # Dummy loss
            
    new_stream = model.add_stream(CausalStream)
    print(f"✅ Added CausalStream. Total streams: {len(model.streams)}")
    
    # Test forward with new stream
    logits, loss, info = model(x, y, return_stream_outputs=True)
    print(f"✅ Forward pass with new stream successful!")
    print(f"New Router Weights shape: {info['gate_weights'].shape}")
    
    print("\n" + "="*60)
    print("✅ KOLOSIS-X IS READY")
    print("="*60)

if __name__ == "__main__":
    test_kolosis_x()
