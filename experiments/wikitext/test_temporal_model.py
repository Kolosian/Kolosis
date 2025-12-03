"""
Quick test script for Kolosis V2 Minimal + Temporal Single Head
Run this in Colab to verify the model works before starting full training.
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_networks.kolosis.kolosis_v2_minimal_temporal_single_head import KolosisV2MinimalTemporalSingleHead

def test_model():
    print("="*60)
    print("TESTING KOLOSIS V2 MINIMAL + TEMPORAL SINGLE HEAD")
    print("="*60)
    
    # Create model
    config = {
        'vocab_size': 100,
        'n_embd': 64,
        'block_size': 32,
        'n_layer': 2,
        'dropout': 0.1
    }
    
    print(f'\nCreating model with config:')
    for k, v in config.items():
        print(f'  {k}: {v}')
    
    try:
        model = KolosisV2MinimalTemporalSingleHead(**config)
        print('\n✅ Model created successfully!')
    except Exception as e:
        print(f'\n❌ Model creation failed: {e}')
        return False
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params:,}')
    
    # Test forward pass
    print('\nTesting forward pass...')
    try:
        x = torch.randint(0, 100, (2, 16))
        y = torch.randint(0, 100, (2, 16))
        
        logits, loss = model(x, y)
        
        print('✅ Forward pass successful!')
        print(f'  Logits shape: {logits.shape}')
        print(f'  Loss: {loss.item():.4f}')
    except Exception as e:
        print(f'❌ Forward pass failed: {e}')
        return False
    
    # Test fusion weights
    print('\nTesting fusion weights...')
    try:
        fusion = model.get_fusion_weights()
        print('✅ Fusion weights retrieved!')
        print(f'  Concept:  {fusion["concept"]:.4f} ({fusion["concept"]*100:.1f}%)')
        print(f'  Semantic: {fusion["semantic"]:.4f} ({fusion["semantic"]*100:.1f}%)')
        print(f'  Temporal: {fusion["temporal"]:.4f} ({fusion["temporal"]*100:.1f}%)')
    except Exception as e:
        print(f'❌ Fusion weights failed: {e}')
        return False
    
    # Test temporal stats
    print('\nTesting temporal stats...')
    try:
        temporal_stats = model.get_temporal_stats()
        print('✅ Temporal stats retrieved!')
        print(f'  Layer 0:')
        print(f'    Fast:   γ={temporal_stats[0]["gamma_fast"]:.4f}, α={temporal_stats[0]["alpha_fast"]:.4f}')
        print(f'    Medium: γ={temporal_stats[0]["gamma_medium"]:.4f}, α={temporal_stats[0]["alpha_medium"]:.4f}')
        print(f'    Slow:   γ={temporal_stats[0]["gamma_slow"]:.4f}, α={temporal_stats[0]["alpha_slow"]:.4f}')
    except Exception as e:
        print(f'❌ Temporal stats failed: {e}')
        return False
    
    # Test generation
    print('\nTesting generation...')
    try:
        context = torch.randint(0, 100, (1, 5))
        generated = model.generate(context, max_new_tokens=10)
        print('✅ Generation successful!')
        print(f'  Generated shape: {generated.shape}')
    except Exception as e:
        print(f'❌ Generation failed: {e}')
        return False
    
    print('\n' + '='*60)
    print('✅ ALL TESTS PASSED!')
    print('='*60)
    print('\nModel is ready for training.')
    print('You can now run: python experiments/wikitext/train_kolosis_v2_temporal_single_head.py')
    
    return True

if __name__ == '__main__':
    success = test_model()
    exit(0 if success else 1)
